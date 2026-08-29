from pathlib import Path

import pytest
import yaml

from mri_correction.config import (
    ConfigurationError,
    CorrectionConfig,
    load_config,
)


def valid_document() -> str:
    return """
input:
  raw_vhdr: data/raw.vhdr
  fmri_metadata: metadata/bold.json
output:
  vhdr: output/corrected.vhdr
timing:
  marker_type: Volume
  marker_description: volume-start
processing:
  method: acquisition_group_fastr
  interpolation_factor: 10
  neighbor_count: 20
  search_radius_samples: 3
  lowpass_hz: 100.0
  output_sampling_rate_hz: 1000.0
  channel_batch_size: 8
  reference_channel: EEG 001
  line_noise_frequencies_hz: [60.0]
"""


def test_load_config_resolves_paths_relative_to_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert isinstance(config, CorrectionConfig)
    assert config.input.raw_vhdr == (tmp_path / "data/raw.vhdr").resolve()
    assert config.input.fmri_metadata == (tmp_path / "metadata/bold.json").resolve()
    assert config.output.vhdr == (tmp_path / "output/corrected.vhdr").resolve()
    assert config.timing.marker_type == "Volume"
    assert config.processing.reference_channel == "EEG 001"
    assert config.processing.line_noise_frequencies_hz == (60.0,)


def test_config_is_immutable(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")
    config = load_config(config_path)

    with pytest.raises(AttributeError):
        config.processing.neighbor_count = 10


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("input", "raw_vhdr"),
        ("input", "fmri_metadata"),
        ("output", "vhdr"),
        ("timing", "marker_type"),
        ("timing", "marker_description"),
        ("processing", "method"),
        ("processing", "interpolation_factor"),
        ("processing", "neighbor_count"),
        ("processing", "search_radius_samples"),
        ("processing", "lowpass_hz"),
        ("processing", "output_sampling_rate_hz"),
        ("processing", "channel_batch_size"),
        ("processing", "reference_channel"),
        ("processing", "line_noise_frequencies_hz"),
    ],
)
def test_config_rejects_missing_required_fields(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    document = {
        "input": {
            "raw_vhdr": "data/raw.vhdr",
            "fmri_metadata": "metadata/bold.json",
        },
        "output": {"vhdr": "output/corrected.vhdr"},
        "timing": {"marker_type": "Volume", "marker_description": "start"},
        "processing": {
            "method": "acquisition_group_fastr",
            "interpolation_factor": 10,
            "neighbor_count": 20,
            "search_radius_samples": 3,
            "lowpass_hz": 100.0,
            "output_sampling_rate_hz": 1000.0,
            "channel_batch_size": 8,
            "reference_channel": "EEG 001",
            "line_noise_frequencies_hz": [60.0],
        },
    }
    del document[section][field]
    config_path = tmp_path / "config.yml"

    import yaml

    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=f"{section}.{field}"):
        load_config(config_path)


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document() + "unexpected: true\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="unexpected"):
        load_config(config_path)


def test_config_accepts_an_explicit_empty_line_noise_list(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    document = valid_document().replace(
        "  line_noise_frequencies_hz: [60.0]\n",
        "  line_noise_frequencies_hz: []\n",
    )
    config_path.write_text(document, encoding="utf-8")

    config = load_config(config_path)

    assert config.processing.line_noise_frequencies_hz == ()


@pytest.mark.parametrize(
    "value",
    ["60.0", [0.0], [500.0], [60.0, 60.0], [True]],
)
def test_config_rejects_invalid_line_noise_frequencies(
    tmp_path: Path,
    value: object,
) -> None:
    document = yaml.safe_load(valid_document())
    document["processing"]["line_noise_frequencies_hz"] = value
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="line_noise_frequencies_hz"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "analyzer"),
        ("interpolation_factor", 0),
        ("neighbor_count", 3),
        ("search_radius_samples", -1),
        ("lowpass_hz", 0.0),
        ("output_sampling_rate_hz", -1.0),
        ("channel_batch_size", 0),
        ("reference_channel", ""),
        ("interpolation_factor", True),
        ("lowpass_hz", True),
    ],
)
def test_config_rejects_invalid_processing_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    default_values = {
        "method": "acquisition_group_fastr",
        "interpolation_factor": 10,
        "neighbor_count": 20,
        "search_radius_samples": 3,
        "lowpass_hz": 100.0,
        "output_sampling_rate_hz": 1000.0,
        "channel_batch_size": 8,
        "reference_channel": "EEG 001",
    }
    default_value = default_values[field]
    default_text = (
        str(default_value).lower()
        if isinstance(default_value, bool)
        else str(default_value)
    )
    replacement_text = (
        str(value).lower() if isinstance(value, bool) else str(value)
    )
    document = valid_document().replace(
        f"  {field}: {default_text}\n",
        f"  {field}: {replacement_text}\n",
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(document, encoding="utf-8")

    with pytest.raises(ConfigurationError, match=field):
        load_config(config_path)


def test_config_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text("- not a mapping\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="mapping"):
        load_config(config_path)


def document_with_trim(trim: str | None) -> str:
    document = valid_document()
    if trim is None:
        return document
    return document + trim


def test_trim_defaults_to_no_trimming(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(document_with_trim(None), encoding="utf-8")

    config = load_config(config_path)

    assert config.trim.mode == "none"


def test_trim_section_is_parsed(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        document_with_trim("trim:\n  mode: first_to_last_volume\n"),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.trim.mode == "first_to_last_volume"


def test_trim_accepts_an_empty_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(document_with_trim("trim: {}\n"), encoding="utf-8")

    config = load_config(config_path)

    assert config.trim.mode == "none"


def test_trim_rejects_an_unknown_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        document_with_trim("trim:\n  mode: everything\n"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match=r"trim\.mode"):
        load_config(config_path)


def test_trim_rejects_unknown_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        document_with_trim("trim:\n  margin_samples: 10\n"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="trim"):
        load_config(config_path)


def test_adaptive_window_defaults_to_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.processing.adaptive_window is False
    assert config.processing.local_neighbor_count == 20


def test_adaptive_window_can_be_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        valid_document() + "  adaptive_window: true\n  local_neighbor_count: 20\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.processing.adaptive_window is True
    assert config.processing.local_neighbor_count == 20


def test_residual_gate_defaults_to_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.processing.residual_gate is False


def test_residual_gate_can_be_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        valid_document() + "  residual_gate: true\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.processing.residual_gate is True


def test_residual_gate_rejects_a_non_boolean(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        valid_document() + "  residual_gate: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="residual_gate"):
        load_config(config_path)


def test_template_high_pass_defaults_to_one_hertz(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.processing.template_high_pass_hz == 1.0


def test_template_high_pass_can_be_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        valid_document() + "  template_high_pass_hz: 0.0\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.processing.template_high_pass_hz == 0.0


def test_template_high_pass_rejects_a_negative_cutoff(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        valid_document() + "  template_high_pass_hz: -1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="template_high_pass_hz"):
        load_config(config_path)


def test_optional_quality_control_defaults_are_explicit(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.quality_control.block_seconds == 30.0
    assert config.quality_control.mains_frequency_hz == 60.0
    assert config.quality_control.mains_exclusion_hz == 1.0


def test_optional_diagnostics_defaults_preserve_current_output(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.diagnostics.psd_max_frequency_hz == 100.0
    assert config.diagnostics.psd_n_fft is None


def test_optional_sections_default_each_missing_field(tmp_path: Path) -> None:
    document = valid_document() + """
quality_control:
  block_seconds: 15.0
diagnostics:
  psd_n_fft: 4096
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(document, encoding="utf-8")

    config = load_config(config_path)

    assert config.quality_control.block_seconds == 15.0
    assert config.quality_control.mains_frequency_hz == 60.0
    assert config.quality_control.mains_exclusion_hz == 1.0
    assert config.diagnostics.psd_max_frequency_hz == 100.0
    assert config.diagnostics.psd_n_fft == 4096


def test_custom_quality_control_and_diagnostic_values_are_loaded(
    tmp_path: Path,
) -> None:
    document = valid_document() + """
quality_control:
  block_seconds: 15.0
  mains_frequency_hz: 50.0
  mains_exclusion_hz: 0.5
diagnostics:
  psd_max_frequency_hz: 80.0
  psd_n_fft: 4096
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(document, encoding="utf-8")

    config = load_config(config_path)

    assert config.quality_control.block_seconds == 15.0
    assert config.quality_control.mains_frequency_hz == 50.0
    assert config.quality_control.mains_exclusion_hz == 0.5
    assert config.diagnostics.psd_max_frequency_hz == 80.0
    assert config.diagnostics.psd_n_fft == 4096


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("quality_control", "block_seconds", 0.0),
        ("quality_control", "block_seconds", -1.0),
        ("quality_control", "block_seconds", True),
        ("quality_control", "mains_frequency_hz", 0.0),
        ("quality_control", "mains_frequency_hz", float("nan")),
        ("quality_control", "mains_exclusion_hz", -0.1),
        ("quality_control", "mains_exclusion_hz", True),
        ("diagnostics", "psd_max_frequency_hz", 0.0),
        ("diagnostics", "psd_max_frequency_hz", float("inf")),
        ("diagnostics", "psd_n_fft", 0),
        ("diagnostics", "psd_n_fft", 12.5),
        ("diagnostics", "psd_n_fft", True),
    ],
)
def test_optional_sections_reject_invalid_values(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    document = yaml.safe_load(valid_document())
    document[section] = {field: value}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=field):
        load_config(config_path)


@pytest.mark.parametrize("section", ["quality_control", "diagnostics"])
def test_optional_sections_reject_unknown_fields(
    tmp_path: Path,
    section: str,
) -> None:
    document = yaml.safe_load(valid_document())
    document[section] = {"unexpected": True}
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=f"{section}.*unexpected"):
        load_config(config_path)


@pytest.mark.parametrize("section", ["quality_control", "diagnostics"])
def test_optional_sections_must_be_mappings(
    tmp_path: Path,
    section: str,
) -> None:
    document = yaml.safe_load(valid_document())
    document[section] = []
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=section):
        load_config(config_path)


def test_robustness_thresholds_default_to_current_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.processing.residual_gate_mad_multiplier == 8.0
    assert config.processing.residual_gate_ratio == 8.0
    assert config.processing.residual_gate_max_fraction == 0.02
    assert config.processing.adaptive_improvement_ratio == 0.85


def test_custom_robustness_thresholds_are_loaded(tmp_path: Path) -> None:
    document = yaml.safe_load(valid_document())
    document["processing"].update(
        {
            "residual_gate_mad_multiplier": 6.0,
            "residual_gate_ratio": 5.0,
            "residual_gate_max_fraction": 0.1,
            "adaptive_improvement_ratio": 0.9,
        }
    )
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    config = load_config(config_path)

    assert config.processing.residual_gate_mad_multiplier == 6.0
    assert config.processing.residual_gate_ratio == 5.0
    assert config.processing.residual_gate_max_fraction == 0.1
    assert config.processing.adaptive_improvement_ratio == 0.9


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("residual_gate_mad_multiplier", 0.0),
        ("residual_gate_mad_multiplier", -1.0),
        ("residual_gate_mad_multiplier", True),
        ("residual_gate_ratio", 0.0),
        ("residual_gate_ratio", float("nan")),
        ("residual_gate_max_fraction", 0.0),
        ("residual_gate_max_fraction", 1.1),
        ("residual_gate_max_fraction", True),
        ("adaptive_improvement_ratio", 0.0),
        ("adaptive_improvement_ratio", 1.1),
        ("adaptive_improvement_ratio", float("inf")),
    ],
)
def test_robustness_thresholds_reject_invalid_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    document = yaml.safe_load(valid_document())
    document["processing"][field] = value
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigurationError, match=field):
        load_config(config_path)


def test_processing_rejects_unknown_robustness_threshold(tmp_path: Path) -> None:
    document = yaml.safe_load(valid_document())
    document["processing"]["robustness_threshold"] = 4.0
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="robustness_threshold"):
        load_config(config_path)


def test_residual_flag_settings_default_to_the_qc_module_values(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.quality_control.residual_mad_multiplier == 6.0
    assert config.quality_control.residual_minimum_channels == 4


def test_residual_flag_settings_are_configurable(tmp_path: Path) -> None:
    document = valid_document() + """
quality_control:
  residual_mad_multiplier: 4.5
  residual_minimum_channels: 8
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(document, encoding="utf-8")

    config = load_config(config_path)

    assert config.quality_control.residual_mad_multiplier == 4.5
    assert config.quality_control.residual_minimum_channels == 8


def test_residual_minimum_channels_must_be_at_least_one(tmp_path: Path) -> None:
    document = valid_document() + """
quality_control:
  residual_minimum_channels: 0
"""
    config_path = tmp_path / "config.yml"
    config_path.write_text(document, encoding="utf-8")

    with pytest.raises(
        ConfigurationError,
        match="residual_minimum_channels must be a positive integer",
    ):
        load_config(config_path)


def test_residual_obs_defaults_to_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(valid_document(), encoding="utf-8")

    config = load_config(config_path)

    assert config.processing.residual_obs is False
    assert config.processing.residual_obs_rank == 4


def test_residual_obs_can_be_enabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        valid_document() + "  residual_obs: true\n  residual_obs_rank: 6\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.processing.residual_obs is True
    assert config.processing.residual_obs_rank == 6


def test_residual_obs_rejects_a_non_boolean(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        valid_document() + "  residual_obs: 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="residual_obs"):
        load_config(config_path)


def test_residual_obs_rank_must_be_a_positive_integer(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        valid_document() + "  residual_obs_rank: 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="residual_obs_rank"):
        load_config(config_path)
