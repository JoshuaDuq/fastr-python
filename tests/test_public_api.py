from eegfmri_fastr import api, config, pipeline


def test_high_level_api_reexports_existing_objects_by_identity() -> None:
    assert api.ConfigurationError is config.ConfigurationError
    assert api.CorrectionConfig is config.CorrectionConfig
    assert api.CorrectionSummary is pipeline.CorrectionSummary
    assert api.PipelineInputError is pipeline.PipelineInputError
    assert api.load_config is config.load_config
    assert api.run_correction is pipeline.run_correction


def test_high_level_api_declares_only_supported_names() -> None:
    assert api.__all__ == [
        "ConfigurationError",
        "CorrectionConfig",
        "CorrectionSummary",
        "PipelineInputError",
        "load_config",
        "run_correction",
    ]


def test_existing_facades_declare_public_boundaries() -> None:
    assert config.__all__ == [
        "ConfigurationError",
        "CorrectionConfig",
        "DiagnosticsConfig",
        "InputConfig",
        "OutputConfig",
        "ProcessingConfig",
        "QualityControlConfig",
        "TimingConfig",
        "TrimConfig",
        "load_config",
    ]
    assert pipeline.__all__ == [
        "CorrectionSummary",
        "PipelineInputError",
        "run_correction",
    ]
