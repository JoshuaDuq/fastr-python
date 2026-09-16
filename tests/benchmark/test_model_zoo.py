import hashlib
import json
from pathlib import Path

import pytest

from benchmark.arms.model_zoo import (
    ModelExport,
    ModelZooError,
    discover_exports,
    fetch,
)

EXPORT_ROOT = "artifacts/exports/masterthesis/phase_1"
PROVENANCE_ROOT = "masterthesis_guide/experiments/phase_1"


def _pointer(oid: str, size: int) -> str:
    return (
        f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {size}\n"
    )


def _make_clone(root: Path, models: dict[str, dict]) -> Path:
    for name, spec in models.items():
        directory = root / EXPORT_ROOT / name / f"holdout_{name}"
        directory.mkdir(parents=True)
        for stem, (oid, size) in spec["exports"].items():
            (directory / f"{stem}.ts").write_text(_pointer(oid, size), encoding="utf-8")
        provenance = root / PROVENANCE_ROOT / f"holdout_{name}" / "provenance"
        provenance.mkdir(parents=True)
        (provenance / "facet_train_config.resolved.json").write_text(
            json.dumps({"training": spec["training"]}), encoding="utf-8"
        )
    return root


TRAINING = {"chunk_size": 512, "target_type": "artifact", "trigger_aligned": True}


def test_discovery_reads_each_models_own_training_settings(tmp_path):
    _make_clone(
        tmp_path,
        {"dpae": {"exports": {"dpae": ("a" * 64, 1024)}, "training": TRAINING}},
    )
    exports = discover_exports(tmp_path)
    assert len(exports) == 1
    assert exports[0].name == "dpae"
    assert exports[0].chunk_size_samples == 512
    assert exports[0].output_type == "artifact"
    assert exports[0].size_bytes == 1024


def test_discovery_prefers_a_cpu_trace_over_a_cuda_one(tmp_path):
    _make_clone(
        tmp_path,
        {
            "demucs": {
                "exports": {"demucs": ("b" * 64, 10), "demucs_cpu": ("c" * 64, 20)},
                "training": TRAINING,
            }
        },
    )
    assert discover_exports(tmp_path)[0].repository_path.endswith("demucs_cpu.ts")


def test_discovery_skips_an_empty_export(tmp_path):
    _make_clone(
        tmp_path,
        {
            "st_gnn": {"exports": {"st_gnn": ("d" * 64, 0)}, "training": TRAINING},
            "dpae": {"exports": {"dpae": ("e" * 64, 99)}, "training": TRAINING},
        },
    )
    assert [export.name for export in discover_exports(tmp_path)] == ["dpae"]


def test_discovery_refuses_a_clone_without_exports(tmp_path):
    with pytest.raises(ModelZooError, match="no FACETpy model exports"):
        discover_exports(tmp_path)


def test_discovery_refuses_a_model_with_no_recorded_training_run(tmp_path):
    directory = tmp_path / EXPORT_ROOT / "mystery" / "holdout_mystery"
    directory.mkdir(parents=True)
    (directory / "mystery.ts").write_text(_pointer("f" * 64, 5), encoding="utf-8")
    with pytest.raises(ModelZooError, match="no resolved training configuration"):
        discover_exports(tmp_path)


def test_discovery_refuses_a_file_that_is_not_a_pointer(tmp_path):
    directory = tmp_path / EXPORT_ROOT / "plain" / "holdout_plain"
    directory.mkdir(parents=True)
    (directory / "plain.ts").write_text("not a pointer", encoding="utf-8")
    with pytest.raises(ModelZooError, match="not a Git LFS pointer"):
        discover_exports(tmp_path)


def _export(payload: bytes) -> ModelExport:
    return ModelExport(
        name="cached",
        repository_path="unused",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        chunk_size_samples=512,
        output_type="artifact",
        trigger_aligned=True,
    )


def test_fetch_returns_a_verified_copy_without_downloading(tmp_path):
    payload = b"already here"
    export = _export(payload)
    (tmp_path / "cached.ts").write_bytes(payload)
    assert fetch(export, tmp_path) == tmp_path / "cached.ts"


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, size: int) -> bytes:
        chunk, self._payload = self._payload[:size], self._payload[size:]
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_fetch_rejects_a_download_that_does_not_match_its_declared_hash(
    tmp_path, monkeypatch
):
    export = _export(b"expected")
    monkeypatch.setattr(
        "benchmark.arms.model_zoo.urllib.request.urlopen",
        lambda url: _FakeResponse(b"tampered"),
    )
    with pytest.raises(ModelZooError, match="not the declared"):
        fetch(export, tmp_path)
    assert not list(tmp_path.glob("*.partial"))


def test_fetch_replaces_a_cached_copy_that_no_longer_matches(tmp_path, monkeypatch):
    payload = b"expected"
    export = _export(payload)
    (tmp_path / "cached.ts").write_bytes(b"stale")
    monkeypatch.setattr(
        "benchmark.arms.model_zoo.urllib.request.urlopen",
        lambda url: _FakeResponse(payload),
    )
    assert fetch(export, tmp_path).read_bytes() == payload
