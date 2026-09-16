import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark.arms import ArmError
from benchmark.arms.facetpy_deep import BLUEPRINT_ALIASES, FacetpyDeepArm
from benchmark.arms.model_zoo import ModelExport, discover_exports, fetch
from benchmark.cohort import RunSpec
from benchmark.config import MatchedSettings

ROOT = Path(__file__).parents[2]
INTERPRETER = ROOT / ".local" / "facetpy-env" / "bin" / "python"
SOURCE = ROOT / ".local" / "facetpy-src"
CACHE = ROOT / ".local" / "facetpy-models"

pytestmark = pytest.mark.skipif(
    not INTERPRETER.is_file() or not SOURCE.is_dir(),
    reason="the FACETpy environment and clone are not installed",
)

DEMO_SETTINGS = MatchedSettings(
    marker_description="volume-start", reference_channel="Cz", neighbor_count=20
)


@pytest.fixture(scope="module")
def demo(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("demo")
    subprocess.run(
        [
            str(Path(sys.executable).parent / "fastr-python"),
            "demo",
            "--output-dir",
            str(directory),
        ],
        check=True,
        capture_output=True,
    )
    return directory


def _spec(demo: Path) -> RunSpec:
    return RunSpec(
        participant="sub-demo",
        run_index=1,
        raw_vhdr=demo / "demo.vhdr",
        protocol_json=demo / "demo_bold.json",
        mean_framewise_displacement=0.1,
        motion_stratum="low",
        marker_block=None,
    )


def _arm(export: ModelExport) -> FacetpyDeepArm:
    return FacetpyDeepArm(
        DEMO_SETTINGS,
        interpreter=INTERPRETER,
        export=export,
        checkpoint=fetch(export, CACHE),
    )


def _export(name: str) -> ModelExport:
    for export in discover_exports(SOURCE):
        if export.name == name:
            return export
    pytest.skip(f"{name} is not in this FACETpy clone")


def test_a_model_that_runs_produces_a_recording_and_records_its_contract(
    demo, tmp_path
):
    result = _arm(_export("dpae")).correct(_spec(demo), output_directory=tmp_path)
    assert result.corrected_vhdr.is_file()
    assert result.volume_count > 0
    assert result.cost.exit_code == 0
    metadata = json.loads(result.corrected_vhdr.with_suffix(".meta.json").read_text())
    assert metadata["channel_group"] == 1
    assert metadata["chunk_size_samples"] > 0


def test_the_arm_is_named_for_the_model_it_ran():
    assert _arm(_export("dpae")).name == "ml_dpae"


def test_an_export_name_resolves_to_the_blueprint_it_came_from():
    assert _arm(_export("ic_unet")).blueprint == "ic_u_net"
    assert _arm(_export("dpae")).blueprint == "dpae"
    assert set(BLUEPRINT_ALIASES) >= {"ic_unet", "dhct_gan_v2"}


def test_a_model_that_cannot_take_this_recording_fails_loudly(demo, tmp_path):
    """Screening reports a model that will not run; it does not skip it."""
    with pytest.raises(ArmError, match="failed on demo"):
        _arm(_export("st_gnn")).correct(_spec(demo), output_directory=tmp_path)
