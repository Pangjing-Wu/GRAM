"""Method metadata written verbatim into every experiment config."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import json


DEFAULT_GRAM_VARIANT = "main_adaptive_mixture"

GRAM_VARIANTS = {
    DEFAULT_GRAM_VARIANT: {
        "role": "main",
        "graph_neighbors": 50,
        "margin_graph_rank": 64,
        "gradient_graph_rank": 64,
        "kernel_weights": "adaptive",
        "active_kernel_components": ("margin", "gradient", "identity"),
        "initial_kernel_weights": (1.0 / 3.0,) * 3,
        "observation_noise_variance": 0.1,
        "weight_regularization": 0.1,
        "minimum_queried_for_weight_learning": 30,
        "require_both_status_classes_for_weight_learning": True,
        "inference": "gaussian_surrogate",
        "acquisition": "repair_plus_informative_value",
        "iv_beta": 1.0,
        "informative_value_pool": "all_unverified_including_query",
        "informative_value_normalization": "mean_over_unverified",
    },
}
_GRAM_MAIN = GRAM_VARIANTS[DEFAULT_GRAM_VARIANT]

GRAM_VARIANTS.update(
    {
        "abl_fixed_uniform_weights": {
            **_GRAM_MAIN,
            "role": "adaptive_weight_ablation",
            "kernel_weights": "fixed",
            "kernel_weight_parameterization": "fixed",
            "marginal_likelihood_reduction": "unused",
            "weight_optimizer": "unused",
            "weight_update_schedule": "fixed_for_all_query_batches",
        },
        "abl_no_identity": {
            **_GRAM_MAIN,
            "role": "identity_ablation",
            "active_kernel_components": ("margin", "gradient"),
            "initial_kernel_weights": (0.5, 0.5, 0.0),
        },
        "abl_no_gradient": {
            **_GRAM_MAIN,
            "role": "gradient_ablation",
            "active_kernel_components": ("margin", "identity"),
            "initial_kernel_weights": (0.5, 0.0, 0.5),
        },
        "abl_no_margin": {
            **_GRAM_MAIN,
            "role": "margin_ablation",
            "active_kernel_components": ("gradient", "identity"),
            "initial_kernel_weights": (0.0, 0.5, 0.5),
        },
    }
)
# Vary trajectory neighbors only; main supplies k=50.
GRAM_VARIANTS.update(
    {
        f"abl_k{k}": {
            **_GRAM_MAIN,
            "role": "graph_neighbors_ablation",
            "diagnostic_neighbors": _GRAM_MAIN["graph_neighbors"],
            "graph_neighbors": k,
        }
        for k in (10, 25, 100, 250)
    }
)
GRAM_VARIANTS.update(
    {
        f"abl_{name}": {
            **_GRAM_MAIN,
            "role": "acquisition_ablation",
            "acquisition": acquisition,
            **({"iv_beta": 0.0} if name == "risk_only" else {}),
        }
        for name, acquisition in (
            ("risk_only", "posterior_risk"),
            ("iv_only", "informative_value"),
            ("variance_only", "posterior_variance"),
            ("local_only", "repair_plus_local_iv"),
        )
    }
)
# beta=0 is risk_only and beta=1 is the main setting. IV keeps its raw units.
GRAM_VARIANTS.update(
    {
        f"abl_beta{str(beta).replace('.', 'p')}": {
            **_GRAM_MAIN,
            "role": "iv_beta_sensitivity",
            "iv_beta": float(beta),
        }
        for beta in (0.1, 10, 100, 1000, 10000)
    }
)
GRAM_ABLATION_VARIANTS = tuple(
    variant for variant in GRAM_VARIANTS if variant != DEFAULT_GRAM_VARIANT
)

# Cross beta with one existing setting at a time. Keep historical names so
# completed single-factor experiments are reused in place.
GRAM_GRID_BETAS = (0.0, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
GRAM_GRID_NEIGHBORS = (10, 25, 50, 100, 250)
# Empirical beta=1 winners on macro-averaged unverified AUPRC area over the
# common 0.001--0.20 budget window. Evidence: docs/gram_k_selection.md.
GRAM_DATASET_NEIGHBORS = {
    "cifar10": 25,
    "cifar100": 100,
    "atis": 250,
    "qnli": 10,
    "adult": 10,
    "letter": 250,
}
GRAM_GRID_KERNELS = {
    "adaptive": DEFAULT_GRAM_VARIANT,
    "fixed_uniform_weights": "abl_fixed_uniform_weights",
    "no_identity": "abl_no_identity",
    "no_gradient": "abl_no_gradient",
    "no_margin": "abl_no_margin",
}
GRAM_GRID_ACQUISITIONS = {
    "repair_iv": "repair_plus_informative_value",
    "local_only": "repair_plus_local_iv",
    "risk_only": "posterior_risk",
    "iv_only": "informative_value",
    "variance_only": "posterior_variance",
}


def gram_grid_settings(reference_neighbors: int = 50) -> dict:
    """Keep the k sweep; anchor all other controls at one fixed reference k."""
    if reference_neighbors not in GRAM_GRID_NEIGHBORS:
        raise ValueError(f"unsupported reference neighbors: {reference_neighbors}")
    return {
        DEFAULT_GRAM_VARIANT: (reference_neighbors, "adaptive", "repair_iv"),
        **{f"abl_k{k}": (k, "adaptive", "repair_iv")
           for k in GRAM_GRID_NEIGHBORS if k != reference_neighbors},
        **{variant: (reference_neighbors, kernel, "repair_iv")
           for kernel, variant in GRAM_GRID_KERNELS.items() if kernel != "adaptive"},
        **{f"abl_{acquisition}": (reference_neighbors, "adaptive", acquisition)
           for acquisition in GRAM_GRID_ACQUISITIONS if acquisition != "repair_iv"},
    }


GRAM_GRID_SETTINGS = gram_grid_settings()


def _gram_config_key(config: dict) -> str:
    effective = {key: value for key, value in config.items() if key != "role"}
    effective.setdefault("diagnostic_neighbors", config["graph_neighbors"])
    return json.dumps(effective, sort_keys=True)


_gram_config_names = {
    _gram_config_key(config): name for name, config in GRAM_VARIANTS.items()
}
_GRAM_GRID_LOOKUP = {}
_gram_grid_structures = tuple(dict.fromkeys(
    setting for reference in GRAM_GRID_NEIGHBORS
    for setting in gram_grid_settings(reference).values()
))
for _beta, (_k, _kernel, _acquisition) in product(
    GRAM_GRID_BETAS, _gram_grid_structures,
):
    _config = {
        **GRAM_VARIANTS[GRAM_GRID_KERNELS[_kernel]],
        "role": "hyperparameter_grid",
        "graph_neighbors": _k,
        "diagnostic_neighbors": _GRAM_MAIN["graph_neighbors"],
        "acquisition": GRAM_GRID_ACQUISITIONS[_acquisition],
        "iv_beta": _beta,
    }
    # beta=0 makes both repair + IV policies identical to risk-only. The other
    # controls do not use beta, so run them once in their default setting.
    if _config["acquisition"] == "posterior_risk" or (
        _beta == 0 and _acquisition in ("repair_iv", "local_only")
    ):
        _config.update(acquisition="posterior_risk", iv_beta=0.0)
    elif _acquisition in ("iv_only", "variance_only"):
        _config["iv_beta"] = 1.0
    _key = _gram_config_key(_config)
    if _key not in _gram_config_names:
        _suffix = {
            "posterior_risk": "risk_only",
            "informative_value": "iv_only",
            "posterior_variance": "variance_only",
        }.get(_config["acquisition"])
        if _suffix is None:
            _suffix = f"{_acquisition}_beta{_beta:g}".replace(".", "p")
        _name = f"grid_{_kernel}_k{_k}_{_suffix}"
        GRAM_VARIANTS[_name] = _config
        _gram_config_names[_key] = _name
    _GRAM_GRID_LOOKUP[(_beta, _k, _kernel, _acquisition)] = _gram_config_names[_key]


def gram_grid_variants(
    *, betas=GRAM_GRID_BETAS, neighbors=GRAM_GRID_NEIGHBORS,
    kernels=tuple(GRAM_GRID_KERNELS), acquisitions=tuple(GRAM_GRID_ACQUISITIONS),
    reference_neighbors: int = 50,
) -> tuple[str, ...]:
    """Cross beta with the single-factor settings that match the filters.

    K controls use the full kernel and repair+IV. Kernel/acquisition controls
    stay at reference_neighbors, rather than crossing all values of k.
    """
    axes = (tuple(betas), tuple(neighbors), tuple(kernels), tuple(acquisitions))
    for values, supported in zip(axes, (
        GRAM_GRID_BETAS, GRAM_GRID_NEIGHBORS,
        GRAM_GRID_KERNELS, GRAM_GRID_ACQUISITIONS,
    )):
        if not values or any(value not in supported for value in values):
            raise ValueError(f"grid values {values!r} must be drawn from {tuple(supported)!r}")
    settings = set(gram_grid_settings(reference_neighbors).values())
    variants = tuple(dict.fromkeys(
        _GRAM_GRID_LOOKUP[key] for key in product(*axes) if key[1:] in settings
    ))
    if not variants:
        raise ValueError(
            "no single-factor settings match these filters; vary only one of "
            f"kernels or acquisitions at reference k={reference_neighbors}, "
            "or select adaptive kernels and repair_iv for the k sweep"
        )
    return variants


GRAM_GRID_VARIANTS = gram_grid_variants()

ACTIVE_LABEL_CORRECTION = {
    "neighbors": 10,
    "diffusion_steps": 10,
    "diffusion_step_size": 0.1,
    "contribution_step_size": 0.1,
    "distance_pairs": 100_000,
}

METHOD_CONFIGS = {
    "aum_b": {
        "source": "https://arxiv.org/abs/2001.10528",
        "implementation": "AUM trajectory plus verification-stratified isotonic calibration",
        "variant": "verification_adapted",
        "quantile_bins": 10,
    },
    "el2n_b": {
        "source": "AMADE",
        "base_score": "el2n",
        "variant": "aum_b_style_stratified_verification_calibration",
        "quantile_bins": 10,
    },
    "forgetting_b": {
        "source": "https://arxiv.org/abs/1812.05159",
        "base_score": "forgetting_events_with_never_learned_ranked_last",
        "variant": "aum_b_style_stratified_verification_calibration",
        "quantile_bins": 10,
    },
    "early_loss_b": {
        "source": "https://www.vldb.org/pvldb/vol17/p1159-chai.pdf",
        "base_score": "mean_cross_entropy_over_configured_early_epochs",
        "variant": "non_iterative_early_loss_with_verification_calibration",
        "quantile_bins": 10,
    },
    "cleanlab_b": {
        "source": "https://github.com/cleanlab/cleanlab",
        "base_score": "one_minus_normalized_margin_label_quality",
        "predicted_probabilities": "shared_task_model_training_set_inference",
        "variant": "verification_adapted_score_calibration",
        "quantile_bins": 10,
    },
    "knn_label_disagreement_b": {
        "source": "AMADE",
        "base_score": "knn_label_disagreement",
        "variant": "aum_b_style_stratified_verification_calibration",
        "quantile_bins": 10,
        "neighbors": 10,
    },
    "moderate_b": {
        "source": "AMADE",
        "base_score": "moderate",
        "variant": "aum_b_style_stratified_verification_calibration",
        "quantile_bins": 10,
    },
    "robust_alc": {
        "source": "https://github.com/kremerj/relabeling",
        "implementation": "multiclass expected-gradient-length acquisition and forward noise correction",
        "global_detection_score": "one_minus_noise_adjusted_posterior_of_original_noisy_label_on_unverified_samples",
        "variant": "shared_backbone_multiclass_adaptation",
        "detection_checkpoint_training": "previous_weights_warm_start_fresh_optimizer",
    },
    "robust_alc_frozen": {
        "source": "https://github.com/kremerj/relabeling",
        "implementation": (
            "frozen shared backbone with verification-updated noise transition, "
            "posterior, and expected-gradient-length acquisition"
        ),
        "global_detection_score": (
            "one_minus_noise_adjusted_posterior_of_original_noisy_label_on_"
            "unverified_samples"
        ),
        "variant": "frozen_backbone_compute_control",
        "detection_checkpoint_training": "shared_task_model_once",
    },
    "dalc": {
        "source": "https://github.com/lilylisy/mlj21DALC",
        "implementation": "DALC-E high-entropy oracle source plus low-entropy pseudo source and forward correction",
        "variant": "shared_backbone_multiclass_adaptation",
        "pseudo_to_oracle_ratio": 1,
        "detection_checkpoint_training": "previous_weights_warm_start_fresh_optimizer",
    },
    "dalc_frozen": {
        "source": "https://github.com/lilylisy/mlj21DALC",
        "implementation": (
            "frozen shared backbone with DALC-E oracle and pseudo sources plus "
            "verification-updated noise posterior"
        ),
        "variant": "frozen_backbone_compute_control",
        "pseudo_to_oracle_ratio": 1,
        "detection_checkpoint_training": "shared_task_model_once",
    },
    "active_label_cleaning": {
        "source": "https://www.nature.com/articles/s41467-022-28818-3",
        "source_code": "microsoft/InnerEye-DeepLearning",
        "source_commit": "1606729c7a16e1bfeb269694314212b6e2737939",
        "acquisition": "cross_entropy_minus_clipped_annotation_difficulty",
        "global_detection_score": "label_correctness_cross_entropy_component_only",
        "variant": "adapted_one_update_per_requested_budget_checkpoint",
        "detection_checkpoint_training": "previous_weights_warm_start_fresh_optimizer",
    },
    "active_label_correction": {
        "source": "https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3670_ECCV_2022_paper.php",
        "acquisition": "robust_parameter_update_plus_entropy_propagation",
        "global_detection_score": "one_minus_robust_classifier_noisy_label_confidence",
        "variant": "shared_backbone_native_mechanism",
        "detection_checkpoint_training": (
            "previous_weights_and_contribution_state_warm_start_fresh_optimizer"
        ),
        **ACTIVE_LABEL_CORRECTION,
    },
    "graph_label_propagation": {
        "implementation": "clamped kNN label propagation with maximum Bernoulli-uncertainty query",
        "variant": "verification_adapted",
        "neighbors": 10,
        "alpha": 0.9,
    },
    "cleannet": {
        "source": "https://github.com/kuanghuei/clean-net",
        "implementation": "cross-modal class-reference cosine relevance plus shared verified binary head",
        "variant": "cross_modal_shared_backbone_reference_head_adaptation",
        "supported_modalities": ("image", "text", "tabular"),
        "restriction": "at_most_B_verified_references",
    },
    "misdetect_b": {
        "source": "https://www.vldb.org/pvldb/vol17/p1159-chai.pdf",
        "implementation": "early-loss pools, last-layer self-influence proxy, verified override, binary detector",
        "variant": "verification_adapted",
    },
    "ours": {
        "feedback": "binary_mislabel_status_only",
        "global_detection_score": "graph_gp_posterior",
        "kernel": "trace_normalized_margin_gradient_heat_kernel_mixture",
        "kernel_normalization": "N_over_trace",
        "kernel_weight_parameterization": "softmax_simplex",
        "gradient_trajectory": "last_layer_per_sample_gradient_norm_by_epoch",
        "gaussian_target_encoding": "two_times_mislabel_status_minus_one",
        "mean_function": "ranked_joint_diagnostics_mapped_to_minus_one_plus_one",
        "marginal_likelihood_reduction": "mean_over_queried_samples",
        "weight_optimizer": "L-BFGS-B_on_anchored_softmax_logits",
        "weight_update_schedule": "once_after_each_completed_query_batch",
        "default_variant": DEFAULT_GRAM_VARIANT,
        "variants": GRAM_VARIANTS,
    },
}


def resolve_gram_variant(method: str, gram_variant: str | None) -> str | None:
    """Resolve and validate the GRAM variant selected for one experiment."""
    if method != "ours":
        if gram_variant is not None:
            raise ValueError("--gram_variant is only valid when --method=ours")
        return None
    selected = DEFAULT_GRAM_VARIANT if gram_variant is None else gram_variant
    if selected not in GRAM_VARIANTS:
        raise ValueError(
            f"unknown GRAM variant {selected!r}; choose from {tuple(GRAM_VARIANTS)}"
        )
    return selected


def method_config(method: str, gram_variant: str | None = None) -> dict:
    """Return the effective, fully materialized method configuration."""
    config = deepcopy(METHOD_CONFIGS.get(method, {}))
    if method != "ours":
        if gram_variant is not None:
            raise ValueError("GRAM variants cannot be applied to baseline methods")
        return config
    selected = resolve_gram_variant(method, gram_variant)
    config.pop("variants")
    config["variant"] = selected
    config.update(deepcopy(GRAM_VARIANTS[selected]))
    return config
