import numpy as np
import pytest

from fastr_python.fastr import (
    FastrInputError,
    adaptive_noise_cancel,
    fmrib_lms,
)

MATLAB_REFS = np.fromstring(
    """0.2,0.56040665155630132,0.86946740149426616,1.0849071543857249,
    1.1768311771816327,1.1317801073872009,0.95456391853555977,
    0.66762647560946475,0.308026496733949,-0.077556691426585653,
    -0.43911691662337793,-0.73006290193195156,-0.91349584024284192,
    -0.96717350459981533,-0.88648938761174145,-0.68506403503760882,
    -0.39286892776661864,-0.052137230524682034,0.28838473581440144,
    0.58006686960361509,0.7810078723605578,0.8617356876216884,
    0.80922391294776597,0.62868025005305206,0.34286462784212046,
    -0.010970990346059167,-0.38610717741408768,-0.73266614824586807,
    -1.0043516636824963,-1.1647076491426309,-1.1920485421857623,
    -1.0823941156729591,-0.8500115645889198,-0.525491446739067,
    -0.151617805849635,0.22240868931825514,0.54761288513288386,
    0.78185378478485401,0.89550548359394477,0.87544178378642001""",
    sep=",",
)
MATLAB_DESIRED = np.fromstring(
    """1.0800000000000001,1.1978290556278954,1.2438394581232317,
    1.2052088767513967,1.0765526275161159,0.86119948383903755,
    0.57146639871205818,0.2278569584707982,-0.1427532769154008,
    -0.50974922072935447,-0.84192278792917563,-1.1107597600501624,
    -1.2934710463064696,-1.375401222667322,-1.3515234735015835,
    -1.2268438286093484,-1.0156740048487491,-0.73987335313255387,
    -0.42628789750993723,-0.1037116129177047,0.20025062200916915,
    0.46203434262463083,0.66434225328689656,0.7974961820028692,
    0.85978376200744044,0.85680402102309716,0.79994745061101891,
    0.70425669121525392,0.58598908618230938,0.46023232289690214,
    0.33890568325105186,0.22941520628577161,0.13413064903229449,
    0.050729736400366099,-0.026672028044611738,-0.10580202420385929,
    -0.1942031846208756,-0.29722502647229992,-0.41635731279193872,
    -0.54818187039446364""",
    sep=",",
)
MATLAB_ERROR = np.fromstring(
    """0,0,0,0,1.0765526275161159,0.77181773886582161,
    0.40294861488177813,0.032977504580313677,-0.3002559403643546,
    -0.58951832686493466,-0.83804193481577105,-1.0383317031360542,
    -1.1661541413727743,-1.1937366745869076,-1.1135026882919696,
    -0.95058882076143003,-0.74925319036425297,-0.54667247509226979,
    -0.35913921190264542,-0.18658202952094285,-0.024261657271256426,
    0.12879037215532418,0.26877868002040051,0.39246650052873516,
    0.50265968542396844,0.60736976746639959,0.71252163749090058,
    0.81343199859352433,0.89063187789209197,0.91498949188514,
    0.86388117874750625,0.73887900882171287,0.56673779434827898,
    0.37971034176748614,0.19525682493681457,0.012952807329124266,
    -0.17477690221538539,-0.36976530233834126,-0.56090857319092702,
    -0.72732779666276304""",
    sep=",",
)
MATLAB_NOISE = np.fromstring(
    """0,0,0,0,0,0.089381744973215924,0.16851778383028007,
    0.19487945389048453,0.15750266344895381,0.079769106135580248,
    -0.0038808531134045259,-0.072428056914108307,-0.12731690493369538,
    -0.18166454808041435,-0.23802078520961378,-0.27625500784791834,
    -0.26642081448449623,-0.19320087804028413,-0.067148685607291786,
    0.082870416603238156,0.22451227928042558,0.33324397046930665,
    0.39556357326649605,0.40502968147413404,0.35712407658347201,
    0.24943425355669752,0.087425813120118351,-0.10917530737827047,
    -0.30464279170978259,-0.45475716898823781,-0.52497549549645439,
    -0.50946380253594126,-0.43260714531598443,-0.32898060536712004,
    -0.22192885298142631,-0.11875483153298355,-0.01942628240549021,
    0.072540275866041365,0.1445512603989883,0.17914592626829939""",
    sep=",",
)


def test_fmrib_lms_matches_matlab_fixture() -> None:
    error, noise = fmrib_lms(
        MATLAB_REFS,
        MATLAB_DESIRED,
        filter_order=4,
        step_size=0.01,
    )

    np.testing.assert_allclose(error, MATLAB_ERROR, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(noise, MATLAB_NOISE, rtol=1e-13, atol=1e-13)


def test_adaptive_noise_cancel_rejects_zero_variance_reference() -> None:
    with pytest.raises(FastrInputError, match="reference variance"):
        adaptive_noise_cancel(
            np.sin(np.arange(2_000) / 11.0)[np.newaxis, :],
            np.zeros((1, 2_000)),
            sampling_rate=500.0,
            filter_order=50,
            excluded_channels=(),
        )


def test_adaptive_noise_cancel_leaves_excluded_channels_untouched() -> None:
    samples = np.arange(2_000)
    artifact = np.sin(2.0 * np.pi * samples / 50.0)[np.newaxis, :]
    corrected = np.cos(2.0 * np.pi * samples / 73.0)[np.newaxis, :]

    result = adaptive_noise_cancel(
        corrected,
        artifact,
        sampling_rate=500.0,
        filter_order=50,
        excluded_channels=(0,),
    )

    np.testing.assert_array_equal(result.data, corrected)
    assert np.isnan(result.reference_scales[0])
    assert np.isnan(result.step_sizes[0])


def test_adaptive_noise_cancel_bypasses_flat_channels() -> None:
    corrected = np.vstack([np.zeros(2_000), np.sin(np.arange(2_000) / 11.0)])
    artifact = np.vstack([np.zeros(2_000), np.sin(np.arange(2_000) / 7.0)])

    result = adaptive_noise_cancel(
        corrected,
        artifact,
        sampling_rate=500.0,
        filter_order=50,
        excluded_channels=(),
    )

    np.testing.assert_array_equal(result.data[0], corrected[0])
    assert np.isnan(result.reference_scales[0])
    assert np.isfinite(result.reference_scales[1])


@pytest.mark.parametrize(
    ("reference", "desired", "filter_order", "step_size", "message"),
    [
        (np.ones(10), np.ones(11), 2, 0.1, "equal length"),
        (np.ones((1, 10)), np.ones(10), 2, 0.1, "one-dimensional"),
        (np.ones(10), np.ones(10), True, 0.1, "filter order"),
        (np.ones(10), np.ones(10), 10, 0.1, "shorter"),
        (np.ones(10), np.ones(10), 2, 0.0, "step size"),
    ],
)
def test_fmrib_lms_rejects_invalid_inputs(
    reference: np.ndarray,
    desired: np.ndarray,
    filter_order: object,
    step_size: object,
    message: str,
) -> None:
    with pytest.raises(FastrInputError, match=message):
        fmrib_lms(
            reference,
            desired,
            filter_order=filter_order,
            step_size=step_size,
        )
