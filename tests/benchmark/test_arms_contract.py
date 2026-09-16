import sys

import pytest

from benchmark.arms import run_measured


def test_run_measured_reports_the_exit_code_and_output():
    result = run_measured([sys.executable, "-c", "print('hello')"])
    assert result.exit_code == 0
    assert result.output.strip() == "hello"


def test_run_measured_reports_a_failure_without_raising():
    result = run_measured([sys.executable, "-c", "raise SystemExit(3)"])
    assert result.exit_code == 3


def test_run_measured_times_the_command():
    result = run_measured([sys.executable, "-c", "import time; time.sleep(0.2)"])
    assert result.wall_clock_seconds >= 0.2


def test_run_measured_attributes_memory_to_the_child_that_used_it():
    small = run_measured([sys.executable, "-c", "pass"])
    big = run_measured(
        [
            sys.executable,
            "-c",
            "x = bytearray(200_000_000); x[::4096] = b'\\x01' * (len(x)//4096)",
        ]
    )
    assert big.peak_memory_bytes > small.peak_memory_bytes + 100_000_000


def test_run_measured_does_not_inherit_a_previous_childs_high_water_mark():
    run_measured(
        [
            sys.executable,
            "-c",
            "x = bytearray(200_000_000); x[::4096] = b'\\x01' * (len(x)//4096)",
        ]
    )
    after = run_measured([sys.executable, "-c", "pass"])
    assert after.peak_memory_bytes < 100_000_000


def test_run_measured_reads_output_larger_than_a_pipe_buffer():
    result = run_measured([sys.executable, "-c", "print('x' * 200_000)"])
    assert result.exit_code == 0
    assert len(result.output) > 200_000


def test_run_measured_surfaces_a_missing_command():
    with pytest.raises(FileNotFoundError):
        run_measured(["definitely-not-a-real-command-42"])


def test_an_arm_error_carries_what_the_failed_attempt_cost():
    """A killed arm is diagnosable only from the clock and memory it reached."""
    from benchmark.arms import ArmError, MeasuredRun

    cost = MeasuredRun(
        wall_clock_seconds=900.0,
        peak_memory_bytes=14_000_000_000,
        exit_code=-9,
        output="",
    )
    error = ArmError("killed", cost=cost)
    assert error.cost is cost
    assert error.cost.exit_code == -9


def test_an_arm_error_without_a_subprocess_carries_no_cost():
    from benchmark.arms import ArmError

    assert ArmError("no matlab on the path").cost is None
