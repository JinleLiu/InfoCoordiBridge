from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import hashlib
import math
import numpy as np

from .association import EntitySeed
from .config import ICAConfig
from .geometry import circular_mean, normalize_angle
from .schemas import FusedEntity, Observation


@dataclass
class ReliabilityBank:
    """Keeps EMA-style reliability terms r_k^(alpha) across frames."""

    values: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def get(self, modality: str, attribute: str, default: float = 1.0) -> float:
        return float(self.values.get(modality, {}).get(attribute, default))

    def update(self, modality: str, attribute: str, score: float, beta: float) -> None:
        self.values.setdefault(modality, {})
        prev = self.values[modality].get(attribute, 1.0)
        self.values[modality][attribute] = float(beta * prev + (1.0 - beta) * score)



def _make_diag_cov(obs: Observation, attribute: str) -> np.ndarray:
    if attribute == "position":
        if obs.covariance is not None:
            cov = np.asarray(obs.covariance, dtype=float)
            if cov.ndim == 2 and cov.shape[0] >= 2:
                return cov[:2, :2]
            if cov.ndim == 1 and len(cov) >= 2:
                return np.diag(np.abs(cov[:2]))
        sigma = max(0.2, 2.0 * (1.0 - obs.score))
        return np.diag([sigma**2, sigma**2])
    if attribute == "velocity":
        sigma = max(0.3, 2.5 * (1.0 - obs.score))
        return np.diag([sigma**2, sigma**2])
    if attribute == "size":
        sigma = max(0.15, 1.5 * (1.0 - obs.score))
        return np.diag([sigma**2, sigma**2, sigma**2])
    if attribute == "yaw":
        sigma = max(0.10, 0.5 * (1.0 - obs.score))
        return np.diag([sigma**2])
    sigma = max(0.2, 2.0 * (1.0 - obs.score))
    return np.diag([sigma**2])



def _attribute_vector(obs: Observation, attribute: str) -> Optional[np.ndarray]:
    if attribute == "position" and obs.position is not None:
        return np.asarray(obs.position, dtype=float)[:2]
    if attribute == "velocity" and obs.velocity is not None:
        return np.asarray(obs.velocity, dtype=float)[:2]
    if attribute == "size" and obs.size is not None:
        return np.asarray(obs.size, dtype=float)[:3]
    if attribute == "yaw" and obs.yaw is not None:
        return np.asarray([float(obs.yaw)], dtype=float)
    return None



def _uncertainty_factor(obs: Observation, attribute: str) -> float:
    cov = _make_diag_cov(obs, attribute)
    trace = float(np.trace(cov))
    return float(1.0 / max(1e-4, trace))



def _consistency_factor(x: np.ndarray, x_seed: np.ndarray, cov: np.ndarray) -> float:
    try:
        inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
        dx = (x - x_seed).reshape(-1, 1)
        val = float((dx.T @ inv @ dx).item())
        return float(math.exp(-val))
    except Exception:
        return 1.0



def _weighted_normalize(items: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
    weights = np.array([max(0.0, float(w)) for _, w in items], dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(items), dtype=float)
    weights = weights / weights.sum()
    return [(obj, float(w)) for (obj, _), w in zip(items, weights.tolist())]



def covariance_intersection(mu_a: np.ndarray, cov_a: np.ndarray,
                            mu_b: np.ndarray, cov_b: np.ndarray,
                            omega: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    omega = float(np.clip(omega, 0.05, 0.95))
    inv_a = np.linalg.inv(cov_a + 1e-6 * np.eye(cov_a.shape[0]))
    inv_b = np.linalg.inv(cov_b + 1e-6 * np.eye(cov_b.shape[0]))
    cov = np.linalg.inv(omega * inv_a + (1.0 - omega) * inv_b)
    mu = cov @ (omega * inv_a @ mu_a + (1.0 - omega) * inv_b @ mu_b)
    return mu, cov



def weighted_information_fusion(vectors: List[np.ndarray], covs: List[np.ndarray], weights: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    assert len(vectors) == len(covs) == len(weights) and len(vectors) > 0
    dim = vectors[0].shape[0]
    info = np.zeros((dim, dim), dtype=float)
    rhs = np.zeros((dim,), dtype=float)
    for x, cov, w in zip(vectors, covs, weights):
        inv = np.linalg.inv(cov + 1e-6 * np.eye(dim))
        info += float(w) * inv
        rhs += float(w) * (inv @ x)
    cov = np.linalg.inv(info + 1e-6 * np.eye(dim))
    mu = cov @ rhs
    return mu, cov



def _fuse_continuous_attribute(observations: List[Observation], attribute: str,
                               cfg: ICAConfig,
                               reliability_bank: ReliabilityBank,
                               x_seed: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, float], List[str]]:
    supported: List[Tuple[Observation, np.ndarray, np.ndarray, float]] = []
    ambiguity_flags: List[str] = []
    consistency_scores: Dict[str, float] = {}
    for obs in observations:
        x = _attribute_vector(obs, attribute)
        if x is None:
            continue
        cov = _make_diag_cov(obs, attribute)
        prior = cfg.prior(attribute, obs.modality)
        rel = reliability_bank.get(obs.modality, attribute, 1.0)
        conf = max(1e-4, float(obs.score))
        unc_factor = _uncertainty_factor(obs, attribute)
        cons = 1.0
        if x_seed is not None and x_seed.shape[0] == x.shape[0]:
            cons = _consistency_factor(x, x_seed, cov)
        w = prior * rel * conf * unc_factor * cons
        supported.append((obs, x, cov, w))
        consistency_scores[obs.modality] = cons
    if not supported:
        return None, None, consistency_scores, ambiguity_flags

    # Update reliability according to pass / suspicious / veto from residual gate.
    for obs, x, cov, _ in supported:
        if x_seed is None or x_seed.shape[0] != x.shape[0]:
            score = 1.0
        else:
            residual = float(np.linalg.norm(x - x_seed))
            if residual <= cfg.ambiguity_residual_threshold_m:
                score = 1.0
            elif residual <= 2.0 * cfg.ambiguity_residual_threshold_m:
                score = 0.5
            else:
                score = 0.0
        reliability_bank.update(obs.modality, attribute, score, cfg.reliability_beta)

    # Handle correlated-source fusion using Covariance Intersection.
    correlated_modalities = {"bevfusion", "lidar", "camera"}
    correlated = [(obs, x, cov, w) for obs, x, cov, w in supported if obs.modality in correlated_modalities]
    independent = [(obs, x, cov, w) for obs, x, cov, w in supported if obs.modality not in correlated_modalities]

    fused_vectors: List[np.ndarray] = []
    fused_covs: List[np.ndarray] = []
    fused_weights: List[float] = []

    if cfg.use_covariance_intersection and len(correlated) >= 2:
        correlated_sorted = sorted(correlated, key=lambda t: t[3], reverse=True)
        obs_a, x_a, cov_a, w_a = correlated_sorted[0]
        obs_b, x_b, cov_b, w_b = correlated_sorted[1]
        omega = float(np.clip(w_a / max(1e-6, w_a + w_b), 0.15, 0.85))
        mu_ci, cov_ci = covariance_intersection(x_a, cov_a, x_b, cov_b, omega)
        fused_vectors.append(mu_ci)
        fused_covs.append(cov_ci)
        fused_weights.append(max(w_a, w_b))
        for obs, x, cov, w in correlated_sorted[2:]:
            fused_vectors.append(x)
            fused_covs.append(cov)
            fused_weights.append(w)
    else:
        for obs, x, cov, w in correlated:
            fused_vectors.append(x)
            fused_covs.append(cov)
            fused_weights.append(w)

    for obs, x, cov, w in independent:
        fused_vectors.append(x)
        fused_covs.append(cov)
        fused_weights.append(w)

    mu, cov = weighted_information_fusion(fused_vectors, fused_covs, fused_weights)

    if x_seed is not None and x_seed.shape[0] == mu.shape[0]:
        max_res = max(float(np.linalg.norm(x - mu)) for _, x, _, _ in supported)
        if max_res > cfg.ambiguity_residual_threshold_m:
            ambiguity_flags.append(f"ambiguous_{attribute}")
    return mu, cov, consistency_scores, ambiguity_flags



def _fuse_categorical(observations: List[Observation], attribute: str,
                      cfg: ICAConfig, reliability_bank: ReliabilityBank) -> Tuple[Any, float, List[str]]:
    weighted_votes: Dict[str, float] = {}
    for obs in observations:
        if attribute == "class":
            candidates = obs.class_probs or {obs.class_name: obs.class_confidence}
        else:
            value = obs.attrs.get(attribute)
            if value is None:
                continue
            candidates = {str(value): 1.0}
        prior = cfg.prior(attribute, obs.modality)
        rel = reliability_bank.get(obs.modality, attribute, 1.0)
        conf = max(1e-4, float(obs.score))
        unc_factor = _uncertainty_factor(obs, attribute)
        w = prior * rel * conf * unc_factor
        for cls_name, cls_p in candidates.items():
            weighted_votes[str(cls_name)] = weighted_votes.get(str(cls_name), 0.0) + w * float(cls_p)
    if not weighted_votes:
        return (None, 0.0, [])
    best = max(weighted_votes.items(), key=lambda kv: kv[1])
    total = max(1e-6, sum(weighted_votes.values()))
    conf = float(best[1] / total)
    flags = []
    if conf < cfg.class_confidence_threshold:
        flags.append(f"ambiguous_{attribute}")
    return best[0], conf, flags



def _derive_entity_id(seed: EntitySeed) -> str:
    track_ids = [obs.track_id for obs in seed.observations if obs.track_id]
    if track_ids:
        # majority vote
        counts: Dict[str, int] = {}
        for tid in track_ids:
            counts[tid] = counts.get(tid, 0) + 1
        return max(counts.items(), key=lambda kv: kv[1])[0]
    sig = "|".join(sorted(f"{o.modality}:{o.sensor_name}:{o.local_id}" for o in seed.observations))
    return "Ent_" + hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]



def fuse_seed(seed: EntitySeed, cfg: ICAConfig, reliability_bank: ReliabilityBank) -> FusedEntity:
    geometry_best = seed.best_geometry()
    seed_pos = np.asarray(geometry_best.position[:2], dtype=float) if geometry_best is not None and geometry_best.position is not None else None
    seed_vel = np.asarray(geometry_best.velocity[:2], dtype=float) if geometry_best is not None and geometry_best.velocity is not None else None
    seed_size = np.asarray(geometry_best.size[:3], dtype=float) if geometry_best is not None and geometry_best.size is not None else None

    pos_mu, pos_cov, pos_cons, pos_flags = _fuse_continuous_attribute(seed.observations, "position", cfg, reliability_bank, seed_pos)
    vel_mu, vel_cov, vel_cons, vel_flags = _fuse_continuous_attribute(seed.observations, "velocity", cfg, reliability_bank, seed_vel)
    size_mu, size_cov, size_cons, size_flags = _fuse_continuous_attribute(seed.observations, "size", cfg, reliability_bank, seed_size)

    yaw_values = [float(obs.yaw) for obs in seed.observations if obs.yaw is not None]
    yaw_weights = []
    for obs in seed.observations:
        if obs.yaw is not None:
            yaw_weights.append(cfg.prior("yaw", obs.modality) * reliability_bank.get(obs.modality, "yaw", 1.0) * max(1e-4, obs.score))
    yaw = circular_mean(yaw_values, yaw_weights) if yaw_values else None

    category, cls_conf, cls_flags = _fuse_categorical(seed.observations, "class", cfg, reliability_bank)

    semantic_attrs: Dict[str, Any] = {}
    fine_type, fine_conf, fine_flags = _fuse_categorical(seed.observations, "fine_type", cfg, reliability_bank)
    if fine_type is not None:
        semantic_attrs["fine_type"] = fine_type
        semantic_attrs["fine_type_confidence"] = fine_conf
    color, color_conf, color_flags = _fuse_categorical(seed.observations, "color", cfg, reliability_bank)
    if color is not None:
        semantic_attrs["color"] = color
        semantic_attrs["color_confidence"] = color_conf
    intent, intent_conf, intent_flags = _fuse_categorical(seed.observations, "intent", cfg, reliability_bank)
    if intent is not None:
        semantic_attrs["intent"] = intent
        semantic_attrs["intent_confidence"] = intent_conf

    entity_id = _derive_entity_id(seed)
    position = np.array([0.0, 0.0, 0.0], dtype=float)
    velocity = np.array([0.0, 0.0, 0.0], dtype=float)
    size = np.array([4.5, 1.8, 1.6], dtype=float)
    if pos_mu is not None:
        position[:2] = pos_mu[:2]
    elif geometry_best is not None and geometry_best.position is not None:
        position[:2] = geometry_best.position[:2]
    if vel_mu is not None:
        velocity[:2] = vel_mu[:2]
    elif geometry_best is not None and geometry_best.velocity is not None:
        velocity[:2] = geometry_best.velocity[:2]
    if size_mu is not None:
        size[: min(3, len(size_mu))] = size_mu[: min(3, len(size_mu))]
    elif geometry_best is not None and geometry_best.size is not None:
        size[:3] = geometry_best.size[:3]

    covariance = None
    if pos_cov is not None and vel_cov is not None:
        covariance = np.zeros((4, 4), dtype=float)
        covariance[:2, :2] = pos_cov[:2, :2]
        covariance[2:4, 2:4] = vel_cov[:2, :2]
    elif pos_cov is not None:
        covariance = pos_cov

    ambiguity_flags = sorted(set(pos_flags + vel_flags + size_flags + cls_flags + fine_flags + color_flags + intent_flags))
    provenance_modalities = sorted(set(obs.modality for obs in seed.observations))
    conflict_resolved = {}
    if len(set(obs.class_name for obs in seed.observations if obs.class_name != "unknown")) > 1:
        conflict_resolved["class"] = [obs.class_name for obs in seed.observations if obs.class_name != "unknown"]
    if len(seed.observations) > 1 and position[:2].tolist() != [0.0, 0.0]:
        centers = [obs.position[:2].tolist() for obs in seed.observations if obs.position is not None]
        if len(centers) >= 2:
            conflict_resolved["geometry"] = centers

    provenance = {
        "source_modalities": provenance_modalities,
        "single_source_retained": len(provenance_modalities) == 1,
        "camera_semantic_attached": any(obs.modality == "camera" for obs in seed.observations),
        "conflict_resolved": conflict_resolved,
        "consistency_scores": {**pos_cons, **vel_cons},
    }
    fusion_lineage = [f"{obs.modality}:{obs.sensor_name}:{obs.local_id}" for obs in seed.observations]
    source_observation_ids = [obs.local_id for obs in seed.observations]

    extras = {}
    if semantic_attrs.get("intent") is not None:
        extras["intent"] = semantic_attrs["intent"]
    if semantic_attrs.get("fine_type") is not None:
        extras["fine_type"] = semantic_attrs["fine_type"]

    return FusedEntity(
        entity_id=entity_id,
        category=str(category) if category is not None else "unknown",
        category_confidence=float(cls_conf),
        position_bev_m=position,
        velocity_bev_mps=velocity,
        yaw_rad=yaw,
        size_m=size,
        covariance=covariance,
        ambiguity_flags=ambiguity_flags,
        provenance=provenance,
        fusion_lineage=fusion_lineage,
        source_observation_ids=source_observation_ids,
        semantic_attributes=semantic_attrs,
        raw_observations=list(seed.observations),
        extras=extras,
    )
