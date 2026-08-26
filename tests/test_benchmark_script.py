import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def test_benchmark_defaults_to_all_channels_including_ecg() -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_sub0001.py"
    spec = importlib.util.spec_from_file_location("benchmark_sub0001", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    available = ["Fz", "ECG", "Cz"]

    assert module._select_channels(available, None) == available


def test_sub0001_benchmark_script_exposes_reproducible_cli() -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_sub0001.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--raw-vhdr" in result.stdout
    assert "--fmri-json" in result.stdout
    assert "--output" in result.stdout


def test_benchmark_output_records_runtime(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "benchmark_sub0001.py"
    spec = importlib.util.spec_from_file_location("benchmark_sub0001", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module.run_benchmark = lambda arguments: {}
    output = tmp_path / "benchmark.json"

    module.main(
        [
            "--raw-vhdr",
            "raw",
            "--analyzer-vhdr",
            "analyzer",
            "--fmri-json",
            "timing",
            "--output",
            str(output),
        ]
    )

    assert json.loads(output.read_text(encoding="utf-8"))["runtime_seconds"] >= 0
