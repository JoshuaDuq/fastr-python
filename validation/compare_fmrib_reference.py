"""Compare matching MAT outputs from the MATLAB and Python FASTR runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from fastr_python.validation.matlab import ComparisonError, compare_arrays


def compare_mat_files(
    matlab_path: Path,
    python_path: Path,
    *,
    repetition_time_seconds: float,
    ecg_channel: str,
) -> dict[str, object]:
    """Validate two runner outputs and calculate their comparison report."""
    matlab = loadmat(matlab_path, simplify_cells=True)
    python = loadmat(python_path, simplify_cells=True)
    matlab_names = tuple(str(name) for name in np.atleast_1d(matlab["channel_names"]))
    python_names = tuple(str(name) for name in np.atleast_1d(python["channel_names"]))
    if matlab_names != python_names:
        raise ComparisonError("MATLAB and Python channel names differ")
    if ecg_channel not in matlab_names:
        raise ComparisonError(f"ECG channel is absent: {ecg_channel!r}")
    matlab_rate = float(matlab["sampling_rate"])
    python_rate = float(python["sampling_rate"])
    if matlab_rate != python_rate:
        raise ComparisonError("MATLAB and Python sampling rates differ")
    matlab_raw = np.asarray(matlab["raw_data"], dtype=np.float64) * 1e-6
    python_raw = np.asarray(python["raw_data"], dtype=np.float64) * 1e-6
    if not np.array_equal(matlab_raw, python_raw):
        raise ComparisonError("MATLAB and Python raw spans differ")
    return compare_arrays(
        raw=matlab_raw,
        matlab=np.asarray(matlab["corrected_data"], dtype=np.float64) * 1e-6,
        python=np.asarray(python["corrected_data"], dtype=np.float64) * 1e-6,
        sampling_rate=matlab_rate,
        repetition_time_seconds=repetition_time_seconds,
        ecg_index=matlab_names.index(ecg_channel),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matlab_mat", type=Path)
    parser.add_argument("python_mat", type=Path)
    parser.add_argument("--repetition-time-seconds", type=float, required=True)
    parser.add_argument("--ecg-channel", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = compare_mat_files(
        arguments.matlab_mat,
        arguments.python_mat,
        repetition_time_seconds=arguments.repetition_time_seconds,
        ecg_channel=arguments.ecg_channel,
    )
    serialized = json.dumps(report, indent=2) + "\n"
    if arguments.output is None:
        print(serialized, end="")
    else:
        arguments.output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
