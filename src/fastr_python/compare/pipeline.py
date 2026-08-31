"""Compare existing uncorrected and FASTR folders."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from .config import CompareConfig
from .pairs import RecordingPair, pair_recordings
from .plots import (
    AlignmentError,
    align_to_fastr,
    load_vhdr,
    metrics_row,
    plot_psd,
)


def run_comparison(config: CompareConfig) -> list[dict]:
    """Compare paired uncorrected and FASTR recordings and write summaries."""
    pairs = pair_recordings(config)
    rows: list[dict] = []
    fig_root = config.paths.output_root / "figures"
    for pair in pairs:
        traces = _load_traces(pair)
        if "Uncorrected" not in traces or "FASTR" not in traces:
            continue
        dest = fig_root / pair.bids_id
        plot_psd(
            traces,
            title=f"Average PSD {pair.bids_id} run {pair.idx_run}",
            output=dest / f"psd_run{pair.idx_run}_avg.png",
            max_hz=config.plot.psd_max_hz,
        )
        rows.append(metrics_row(pair, traces, max_hz=config.plot.psd_max_hz))
        print(f"compared {pair.bids_id} run {pair.idx_run} {pair.key}")
        for raw in traces.values():
            raw.close()
    _write_summary(config.paths.output_root, rows)
    return rows


def _load_traces(pair: RecordingPair) -> dict[str, object]:
    traces: dict[str, object] = {}
    try:
        fastr = load_vhdr(pair.fastr_vhdr)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"failed to load FASTR {pair.fastr_vhdr}: {error}")
        return traces
    try:
        uncorrected = load_vhdr(pair.uncorrected_vhdr)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"failed to load uncorrected {pair.uncorrected_vhdr}: {error}")
        fastr.close()
        return traces
    try:
        aligned, cropped_fastr = align_to_fastr(uncorrected, fastr)
    except AlignmentError as error:
        print(f"skip {pair.fastr_vhdr.name}: {error}")
        fastr.close()
        uncorrected.close()
        return traces
    traces["FASTR"] = cropped_fastr
    traces["Uncorrected"] = aligned
    if fastr is not cropped_fastr:
        fastr.close()
    if uncorrected is not aligned:
        uncorrected.close()
    return traces


def _write_summary(output_root: Path, rows: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "compare_summary.json"
    csv_path = output_root / "compare_summary.csv"
    json_path.write_text(json.dumps(rows, indent=2))
    if not rows:
        return
    keys = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
