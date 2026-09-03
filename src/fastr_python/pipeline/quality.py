"""Assemble pipeline residual-quality measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..fastr import AcquisitionGeometry
from ..quality.residuals import (
    ResidualQcDefaults,
    block_residual_uv,
    flag_blocks,
    flag_channel_blocks,
    slice_harmonics,
    volume_harmonic_spectrum,
)


@dataclass(frozen=True, slots=True)
class _BlockResidualMeasurement:
    """One residual measurement, kept apart from the report written about it.

    The automatic channel-failure policy has to measure the same quantity the
    sidecar reports, on data the sidecar has not been written for yet, and then
    measure it again on a single retried channel. Holding the numbers and the
    block geometry that produced them makes those three measurements the same
    measurement rather than three that happen to agree.
    """

    residuals_uv: np.ndarray
    harmonics_hz: tuple[float, ...]
    block_seconds: float
    volumes_per_block: int


def _measure_residual_qc(
    corrected: np.ndarray,
    *,
    channel_names: list[str],
    non_eeg_indices: frozenset[int],
    output_rate: float,
    acquisition: AcquisitionGeometry,
    threshold_uv: float,
    block_seconds: float,
    mains_frequency_hz: float,
    mains_exclusion_hz: float,
    volume_spectrum_max_hz: float,
    mad_multiplier: float = ResidualQcDefaults.MAD_MULTIPLIER,
    minimum_channels: int = ResidualQcDefaults.MINIMUM_CHANNELS,
    report_channel_outliers: bool = True,
) -> dict[str, object]:
    """Measure residual gradient artifact across the whole corrected recording.

    The existing sidecar reports alignment correlations and amplitude fits, and
    neither moves when a correction fails: on this cohort both stayed healthy
    through blocks carrying twenty microvolts of residual artifact.
    """
    measurement = _measure_block_residuals(
        corrected,
        output_rate=output_rate,
        acquisition=acquisition,
        block_seconds=block_seconds,
        mains_frequency_hz=mains_frequency_hz,
        mains_exclusion_hz=mains_exclusion_hz,
    )
    return _residual_qc_report(
        measurement,
        corrected,
        channel_names=channel_names,
        non_eeg_indices=non_eeg_indices,
        output_rate=output_rate,
        acquisition=acquisition,
        threshold_uv=threshold_uv,
        mains_frequency_hz=mains_frequency_hz,
        mains_exclusion_hz=mains_exclusion_hz,
        volume_spectrum_max_hz=volume_spectrum_max_hz,
        mad_multiplier=mad_multiplier,
        minimum_channels=minimum_channels,
        report_channel_outliers=report_channel_outliers,
    )


def _measure_block_residuals(
    corrected: np.ndarray,
    *,
    output_rate: float,
    acquisition: AcquisitionGeometry,
    block_seconds: float,
    mains_frequency_hz: float,
    mains_exclusion_hz: float,
) -> _BlockResidualMeasurement:
    """Measure acquisition-locked residual amplitude over whole-volume blocks."""
    repetition_time = acquisition.repetition_time_seconds
    harmonics = slice_harmonics(
        groups_per_volume=acquisition.groups_per_volume,
        repetition_time_seconds=repetition_time,
        nyquist_hz=output_rate / 2.0,
        mains_hz=mains_frequency_hz,
        exclusion_hz=mains_exclusion_hz,
    )
    # A block boundary falling mid-volume splits one acquisition across two
    # blocks, so round the requested length to whole volumes.
    volumes_per_block = max(1, round(block_seconds / repetition_time))
    aligned_block_seconds = volumes_per_block * repetition_time
    return _BlockResidualMeasurement(
        residuals_uv=block_residual_uv(
            np.asarray(corrected) * 1e6,
            sampling_rate=output_rate,
            harmonics=harmonics,
            block_seconds=aligned_block_seconds,
        ),
        harmonics_hz=harmonics,
        block_seconds=float(aligned_block_seconds),
        volumes_per_block=int(volumes_per_block),
    )


def _residual_qc_report(
    measurement: _BlockResidualMeasurement,
    corrected: np.ndarray,
    *,
    channel_names: list[str],
    non_eeg_indices: frozenset[int],
    output_rate: float,
    acquisition: AcquisitionGeometry,
    threshold_uv: float,
    mains_frequency_hz: float,
    mains_exclusion_hz: float,
    volume_spectrum_max_hz: float,
    mad_multiplier: float,
    minimum_channels: int,
    report_channel_outliers: bool,
) -> dict[str, object]:
    """Serialize one residual measurement, and the flags derived from it."""
    repetition_time = acquisition.repetition_time_seconds
    residuals = measurement.residuals_uv
    flagged = flag_blocks(
        residuals,
        mad_multiplier=mad_multiplier,
        minimum_channels=minimum_channels,
        floor_uv=threshold_uv,
    )
    channel_flags = (
        flag_channel_blocks(
            residuals,
            mad_multiplier=mad_multiplier,
            floor_uv=threshold_uv,
        )
        if report_channel_outliers
        else np.zeros(residuals.shape, dtype=bool)
    )
    for index in non_eeg_indices:
        channel_flags[index] = False
    flagged_channel_blocks = {
        name: [int(block) for block in np.flatnonzero(channel_flags[index])]
        for index, name in enumerate(channel_names)
        if np.any(channel_flags[index])
    }
    maximum_spectrum_frequency = min(
        volume_spectrum_max_hz,
        float(np.nextafter(output_rate / 2.0, 0.0)),
    )
    eeg_indices = [
        index for index in range(len(channel_names)) if index not in non_eeg_indices
    ]
    volume_spectrum = volume_harmonic_spectrum(
        np.asarray(corrected)[eeg_indices] * 1e6,
        sampling_rate=output_rate,
        repetition_time_seconds=repetition_time,
        maximum_frequency_hz=maximum_spectrum_frequency,
        mains_frequency_hz=mains_frequency_hz,
        mains_exclusion_hz=mains_exclusion_hz,
    )
    if residuals.shape[1] == 0:
        worst_block = [-1] * residuals.shape[0]
        worst_uv = [0.0] * residuals.shape[0]
    else:
        worst_block = [int(index) for index in residuals.argmax(axis=1)]
        worst_uv = [float(value) for value in residuals.max(axis=1)]
    return {
        "block_seconds": measurement.block_seconds,
        "volumes_per_block": measurement.volumes_per_block,
        "harmonics_hz": [float(value) for value in measurement.harmonics_hz],
        "mains_frequency_hz": float(mains_frequency_hz),
        "mains_exclusion_hz": float(mains_exclusion_hz),
        "floor_uv": float(threshold_uv),
        "mad_multiplier": float(mad_multiplier),
        "minimum_channels": int(minimum_channels),
        "channel_names": list(channel_names),
        "block_residual_uv": [[float(v) for v in row] for row in residuals],
        "worst_block_index": worst_block,
        "worst_block_uv": worst_uv,
        "flagged_blocks": [bool(value) for value in flagged],
        "flagged_block_count": int(flagged.sum()),
        "report_channel_outliers": bool(report_channel_outliers),
        "flagged_channel_blocks_by_channel": flagged_channel_blocks,
        "flagged_channel_block_count": int(channel_flags.sum()),
        "volume_harmonic_spectrum": [
            asdict(measurement) for measurement in volume_spectrum
        ],
    }
