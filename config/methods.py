"""Method metadata written verbatim into every experiment config."""

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
        "variant": "shared_backbone_multiclass_adaptation",
    },
    "dalc": {
        "source": "https://github.com/lilylisy/mlj21DALC",
        "implementation": "DALC-E high-entropy oracle source plus low-entropy pseudo source and forward correction",
        "variant": "shared_backbone_multiclass_adaptation",
        "pseudo_to_oracle_ratio": 1,
    },
    "active_label_cleaning": {
        "source": "https://www.nature.com/articles/s41467-022-28818-3",
        "source_code": "microsoft/InnerEye-DeepLearning",
        "source_commit": "1606729c7a16e1bfeb269694314212b6e2737939",
        "acquisition": "cross_entropy_minus_clipped_annotation_difficulty",
        "global_detection_score": "label_correctness_cross_entropy_component_only",
        "variant": "adapted_one_update_per_requested_budget_checkpoint",
    },
    "active_label_correction": {
        "source": "https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/3670_ECCV_2022_paper.php",
        "acquisition": "robust_parameter_update_plus_entropy_propagation",
        "global_detection_score": "one_minus_robust_classifier_noisy_label_confidence",
        "variant": "shared_backbone_native_mechanism",
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
        "implementation": "class-reference cosine relevance plus shared verified binary head",
        "variant": "shared_backbone_reference_head_adaptation",
        "restriction": "image_only_and_at_most_B_verified_references",
    },
    "misdetect_b": {
        "source": "https://www.vldb.org/pvldb/vol17/p1159-chai.pdf",
        "implementation": "early-loss pools, last-layer self-influence proxy, verified override, binary detector",
        "variant": "verification_adapted",
    },
    "ours": {
        "feedback": "binary_mislabel_status_only",
        "global_detection_score": "graph_gp_posterior",
    },
}
