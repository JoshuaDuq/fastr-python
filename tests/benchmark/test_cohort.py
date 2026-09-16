import json
from pathlib import Path

import pytest

from benchmark.cohort import (
    CohortPaths,
    RunSpec,
    load_manifest,
    mean_framewise_displacement,
    select_cohort,
    write_manifest,
)

CONFOUNDS_HEADER = "csf\tframewise_displacement\ttrans_x\n"


def _write_confounds(path: Path, displacements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = "".join(f"0.1\t{value}\t0.2\n" for value in displacements)
    path.write_text(CONFOUNDS_HEADER + rows, encoding="utf-8")


def _build_cohort(root: Path, *, participants: int, runs: int) -> CohortPaths:
    source = root / "source_data"
    confounds = root / "fmriprep"
    protocol = root / "bids" / "fmri"
    for index in range(participants):
        participant = f"sub-{index:04d}"
        trimmed = source / participant / "eeg" / "original_trimmed_5khz"
        untrimmed = source / participant / "eeg" / "original_untrimmed_5khz"
        trimmed.mkdir(parents=True, exist_ok=True)
        untrimmed.mkdir(parents=True, exist_ok=True)
        for suffix in (".vhdr", ".vmrk", ".eeg"):
            (untrimmed / f"BaselineEEG_x{suffix}").write_bytes(b"rest")
        (trimmed / "BaselineEEG_x_first_to_last_volume.vhdr").touch()
        for run in range(1, runs + 1):
            stem = f"ThermalPainEEGFMRI_run{run}_{participant}_t"
            (trimmed / f"{stem}_first_to_last_volume.vhdr").touch()
            for suffix in (".vhdr", ".vmrk", ".eeg"):
                (untrimmed / f"{stem}{suffix}").write_bytes(stem.encode())
            (protocol / participant / "func").mkdir(parents=True, exist_ok=True)
            (
                protocol
                / participant
                / "func"
                / f"{participant}_task-thermalactive_run-{run:02d}_bold.json"
            ).touch()
            _write_confounds(
                confounds
                / participant
                / "func"
                / f"{participant}_task-thermalactive_run-{run:02d}"
                "_desc-confounds_timeseries.tsv",
                ["n/a", str(0.1 * run), str(0.1 * run + 0.02)],
            )
    return CohortPaths(
        source_root=source, confounds_root=confounds, protocol_root=protocol
    )


def test_mean_framewise_displacement_skips_the_unset_first_volume(tmp_path):
    path = tmp_path / "confounds.tsv"
    _write_confounds(path, ["n/a", "0.20", "0.40"])
    assert mean_framewise_displacement(path) == pytest.approx(0.30)


def test_mean_framewise_displacement_rejects_a_file_without_the_column(tmp_path):
    path = tmp_path / "confounds.tsv"
    path.write_text("csf\ttrans_x\n0.1\t0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="framewise_displacement"):
        mean_framewise_displacement(path)


def test_select_cohort_takes_three_motion_strata_per_participant(tmp_path):
    paths = _build_cohort(tmp_path, participants=3, runs=6)
    selected = select_cohort(paths, runs_per_participant=3)
    assert len(selected) == 9
    for participant in ("sub-0000", "sub-0001", "sub-0002"):
        runs = [run for run in selected if run.participant == participant]
        assert [run.motion_stratum for run in runs] == ["low", "middle", "high"]
        assert [run.run_index for run in runs] == [1, 3, 6]


def test_select_cohort_excludes_baseline_recordings(tmp_path):
    paths = _build_cohort(tmp_path, participants=2, runs=4)
    selected = select_cohort(paths, runs_per_participant=3)
    assert all("Baseline" not in run.raw_vhdr.name for run in selected)


def test_select_cohort_refuses_a_participant_with_too_few_runs(tmp_path):
    paths = _build_cohort(tmp_path, participants=1, runs=2)
    with pytest.raises(ValueError, match="sub-0000"):
        select_cohort(paths, runs_per_participant=3)


def test_select_cohort_is_deterministic(tmp_path):
    paths = _build_cohort(tmp_path, participants=3, runs=6)
    assert select_cohort(paths, runs_per_participant=3) == select_cohort(
        paths, runs_per_participant=3
    )


def test_manifest_round_trips_through_disk(tmp_path):
    paths = _build_cohort(tmp_path, participants=2, runs=5)
    selected = select_cohort(paths, runs_per_participant=3)
    manifest = tmp_path / "manifest.json"
    write_manifest(selected, manifest)
    assert load_manifest(manifest) == selected


def test_manifest_records_a_hash_for_every_selected_recording(tmp_path):
    paths = _build_cohort(tmp_path, participants=1, runs=3)
    manifest = tmp_path / "manifest.json"
    write_manifest(select_cohort(paths, runs_per_participant=3), manifest)
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(recorded["runs"]) == 3
    assert all(len(row["vhdr_sha256"]) == 64 for row in recorded["runs"])
    assert all(len(row["vmrk_sha256"]) == 64 for row in recorded["runs"])
    assert all(row["eeg_bytes"] > 0 for row in recorded["runs"])


def test_run_spec_carries_the_declared_marker_block_for_sub_0000_run_1(tmp_path):
    paths = _build_cohort(tmp_path, participants=1, runs=3)
    selected = select_cohort(paths, runs_per_participant=3)
    assert all(run.marker_block is None for run in selected)
    assert RunSpec.__slots__
