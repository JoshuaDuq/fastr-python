"""Template construction and correlation primitives for FASTR."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, firwin, sosfiltfilt, upfirdn

from .fastr_types import FastrInputError, _ArtifactEpoch, _TemplateWindow
from .fastr_validation import validate_sampling_rate

_INTERPOLATION_HALF_SPAN = 4
_INTERPOLATION_WINDOW = ("kaiser", 5.0)
_TEMPLATE_MEAN_CHUNK = 128


def _make_template_high_pass(
    cutoff_hz: float | None,
    *,
    sampling_rate: float | None,
) -> np.ndarray | None:
    """Build the stage-2 template high-pass, or None when it is disabled."""
    if cutoff_hz is None:
        return None
    if isinstance(cutoff_hz, bool) or not isinstance(cutoff_hz, Real):
        raise FastrInputError("template high-pass cutoff must be a finite number")
    cutoff = float(cutoff_hz)
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise FastrInputError("template high-pass cutoff must be a finite number")
    if cutoff == 0.0:
        return None
    rate = validate_sampling_rate(sampling_rate)
    if cutoff >= 0.5 * rate:
        raise FastrInputError(
            "template high-pass cutoff must stay below the input Nyquist frequency"
        )
    return butter(2, cutoff, btype="high", fs=rate, output="sos")


def _template_estimate_signal(
    channel: np.ndarray,
    template_filter: np.ndarray | None,
) -> np.ndarray:
    if template_filter is None:
        return channel
    return sosfiltfilt(template_filter, channel)


def _make_interpolation_filter(factor: int) -> np.ndarray:
    """Build the band-limited filter implied by FASTR's interpolation step."""
    if factor == 1:
        return np.ones(1)
    taps = firwin(
        2 * factor * _INTERPOLATION_HALF_SPAN + 1,
        1.0 / factor,
        window=_INTERPOLATION_WINDOW,
    )
    for phase in range(factor):
        branch = taps[phase::factor]
        branch /= branch.sum()
    return taps


def _interpolate(channel: np.ndarray, taps: np.ndarray, factor: int) -> np.ndarray:
    delay = (taps.size - 1) // 2
    upsampled = upfirdn(taps, channel.astype(np.float64, copy=False), up=factor)
    return upsampled[delay : delay + channel.size * factor]


def _extract_epochs(
    signal: np.ndarray,
    fine_triggers: np.ndarray,
    samples_before: int,
    samples_after: int,
) -> np.ndarray:
    offsets = np.arange(-samples_before, samples_after + 1)
    return signal[fine_triggers[:, np.newaxis] + offsets]


def _place_epochs(
    sample_count: int,
    starts: np.ndarray,
    epochs: np.ndarray,
) -> np.ndarray:
    """Lay fitted epochs out, each owning the grid up to the next epoch start."""
    placed = np.zeros(sample_count)
    length = epochs.shape[1]
    ends = np.append(starts[1:], starts[-1] + length)
    for index, start in enumerate(starts):
        written = min(ends[index], start + length) - start
        placed[start : start + written] = epochs[index, :written]
    return placed


def _make_templates(
    signal: np.ndarray,
    fine_triggers: np.ndarray,
    window: _TemplateWindow,
    epoch: _ArtifactEpoch,
) -> np.ndarray:
    """Average each target's chosen epochs, summing each residue class once."""
    epochs = _extract_epochs(
        signal,
        fine_triggers,
        epoch.samples_before,
        epoch.samples_after,
    )
    if not window.summed_contiguous:
        return _mean_selected_epochs(epochs, window.indices)
    neighbor_count = window.indices.shape[1]
    templates = np.empty_like(epochs)
    for residue in range(window.stride):
        class_epochs = epochs[residue :: window.stride]
        totals = np.zeros((class_epochs.shape[0] + 1, epoch.length))
        np.cumsum(class_epochs, axis=0, out=totals[1:])
        targets = np.flatnonzero(window.indices[:, 0] % window.stride == residue)
        starts = window.run_starts[targets]
        summed = totals[starts + window.run_length] - totals[starts]
        if window.contains_target:
            summed = summed - epochs[targets]
        templates[targets] = summed / neighbor_count
    return templates


def _mean_selected_epochs(epochs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Average arbitrary neighbour sets without materialising the full gather."""
    templates = np.empty_like(epochs)
    for start in range(0, indices.shape[0], _TEMPLATE_MEAN_CHUNK):
        stop = min(start + _TEMPLATE_MEAN_CHUNK, indices.shape[0])
        chosen = indices[start:stop]
        selected = epochs[np.clip(chosen, 0, epochs.shape[0] - 1)]
        valid = chosen >= 0
        selected = np.where(valid[..., np.newaxis], selected, 0.0)
        counts = np.maximum(valid.sum(axis=1, keepdims=True), 1)
        templates[start:stop] = selected.sum(axis=1) / counts
    return templates


def _template_residual(epochs: np.ndarray, templates: np.ndarray) -> np.ndarray:
    energies = np.sum(templates**2, axis=1)
    amplitudes = np.divide(
        np.sum(epochs * templates, axis=1),
        energies,
        out=np.ones(epochs.shape[0]),
        where=energies > 0.0,
    )
    return epochs - amplitudes[:, np.newaxis] * templates


def _fit_group_shifts(
    signal: np.ndarray,
    fine_triggers: np.ndarray,
    window: _TemplateWindow,
    epoch: _ArtifactEpoch,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    templates = _make_templates(signal, fine_triggers, window, epoch)
    shifts = np.empty(fine_triggers.size, dtype=np.int64)
    correlations = np.empty(fine_triggers.size, dtype=np.float64)
    for index, trigger in enumerate(fine_triggers):
        searched = signal[
            trigger - epoch.samples_before - search_radius : trigger
            + epoch.samples_after
            + search_radius
            + 1
        ]
        scores = _correlate(
            sliding_window_view(searched, epoch.length),
            templates[index],
        )
        best = int(np.argmax(scores))
        shifts[index] = best - search_radius
        correlations[index] = scores[best]
    return shifts, correlations


def _correlate(candidates: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Pearson correlation of every candidate epoch against one template."""
    centered = candidates - candidates.mean(axis=-1, keepdims=True)
    reference = template - template.mean()
    norms = np.sqrt(np.sum(centered**2, axis=-1) * np.sum(reference**2))
    return np.divide(
        centered @ reference,
        norms,
        out=np.zeros(candidates.shape[0]),
        where=norms > 0.0,
    )


def _fit_channel_noise(
    signal: np.ndarray,
    fitted_triggers: np.ndarray,
    window: _TemplateWindow,
    epoch: _ArtifactEpoch,
) -> tuple[np.ndarray, np.ndarray]:
    templates = _make_templates(signal, fitted_triggers, window, epoch)
    epochs = _extract_epochs(
        signal,
        fitted_triggers,
        epoch.samples_before,
        epoch.samples_after,
    )
    energies = np.sum(templates**2, axis=1)
    amplitudes = np.divide(
        np.sum(epochs * templates, axis=1),
        energies,
        out=np.ones(fitted_triggers.size),
        where=energies > 0.0,
    )

    noise = _place_epochs(
        signal.size,
        fitted_triggers - epoch.samples_before,
        amplitudes[:, np.newaxis] * templates,
    )
    return noise, amplitudes
