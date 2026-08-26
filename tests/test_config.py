from pathlib import Path

import pytest

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
