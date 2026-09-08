"""Method metadata written verbatim into every experiment config."""

from __future__ import annotations

from copy import deepcopy


DEFAULT_GRAM_VARIANT = "main_adaptive_mixture_uncertainty"

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
        "acquisition": "posterior_variance",
    },
    "abl_fixed_uniform_weights": {
        "role": "adaptive_weight_ablation",
        "kernel_weight_parameterization": "fixed",
        "marginal_likelihood_reduction": "unused",
        "weight_optimizer": "unused",
        "weight_update_schedule": "fixed_for_all_query_batches",
        "graph_neighbors": 50,
        "margin_graph_rank": 64,
        "gradient_graph_rank": 64,
        "kernel_weights": "fixed",
        "active_kernel_components": ("margin", "gradient", "identity"),
        "initial_kernel_weights": (1.0 / 3.0,) * 3,
        "observation_noise_variance": 0.1,
        "inference": "gaussian_surrogate",
        "acquisition": "posterior_variance",
    },
    "abl_no_identity": {
        "role": "identity_ablation",
        "graph_neighbors": 50,
        "margin_graph_rank": 64,
        "gradient_graph_rank": 64,
        "kernel_weights": "adaptive",
        "active_kernel_components": ("margin", "gradient"),
        "initial_kernel_weights": (0.5, 0.5, 0.0),
        "observation_noise_variance": 0.1,
        "weight_regularization": 0.1,
        "minimum_queried_for_weight_learning": 30,
        "require_both_status_classes_for_weight_learning": True,
        "inference": "gaussian_surrogate",
        "acquisition": "posterior_variance",
    },
    "abl_margin_only": {
        "role": "gradient_and_identity_ablation",
        "kernel_weight_parameterization": "fixed",
        "marginal_likelihood_reduction": "unused",
        "weight_optimizer": "unused",
        "weight_update_schedule": "fixed_for_all_query_batches",
        "graph_neighbors": 50,
        "margin_graph_rank": 64,
        "gradient_graph_rank": 64,
        "kernel_weights": "fixed",
        "active_kernel_components": ("margin",),
        "initial_kernel_weights": (1.0, 0.0, 0.0),
        "observation_noise_variance": 0.1,
        "inference": "gaussian_surrogate",
        "acquisition": "posterior_variance",
    },
    "abl_gradient_only": {
        "role": "margin_and_identity_ablation",
        "kernel_weight_parameterization": "fixed",
        "marginal_likelihood_reduction": "unused",
        "weight_optimizer": "unused",
        "weight_update_schedule": "fixed_for_all_query_batches",
        "graph_neighbors": 50,
        "margin_graph_rank": 64,
        "gradient_graph_rank": 64,
        "kernel_weights": "fixed",
        "active_kernel_components": ("gradient",),
        "initial_kernel_weights": (0.0, 1.0, 0.0),
        "observation_noise_variance": 0.1,
        "inference": "gaussian_surrogate",
        "acquisition": "posterior_variance",
    },
}

# Vary only the trajectory graphs; keep the diagnostic prior at the main k.
# k=50 is already covered by DEFAULT_GRAM_VARIANT and need not be rerun.
GRAM_VARIANTS.update(
    {
        f"abl_k{k}": {
            **GRAM_VARIANTS[DEFAULT_GRAM_VARIANT],
            "role": "graph_neighbors_ablation",
            "diagnostic_neighbors": GRAM_VARIANTS[DEFAULT_GRAM_VARIANT]["graph_neighbors"],
            "graph_neighbors": k,
        }
        for k in (10, 25, 100, 250)
    }
)

# Acquisition-only ablations use the current adaptive mixture and Gaussian GP.
# beta=0 is pure posterior-mean acquisition; sigma is the latent standard deviation.
GRAM_VARIANTS.update(
    {
        f"abl_ucb_beta{str(beta).replace('.', 'p')}": {
            **GRAM_VARIANTS[DEFAULT_GRAM_VARIANT],
            "role": "acquisition_ablation",
            "acquisition": "latent_ucb",
            "ucb_beta": float(beta),
        }
        for beta in (0, 0.5, 1, 2)
    }
)

# The dedicated ablation launcher runs every variant except the main setting.
GRAM_ABLATION_VARIANTS = tuple(
    variant for variant in GRAM_VARIANTS if variant != DEFAULT_GRAM_VARIANT
)

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
