import numpy as np
import pytest

from nect.src.sampling.methods import equidistant, golden_angle, hybrid_golden_angle


@pytest.mark.parametrize(
    "nprojs, nrevs, radians",
    [(13, 1, True), (13, 1, False), (100, 2, False)],
    ids=["Radians", "Degrees", "100 projections and 2 revolutions"],
)
def test_equidistant_sampling(nprojs: int, nrevs: int, radians: bool):
    """Checks that the correct number of angles is reveived and that they are equidistantly spaced.
    Also checks the boundary values.

    Args:
        nprojs (int): How many projections per revolution.
        nrevs (int): Number of revolutions.
        radians (bool): Whether to return angles in radians or degrees.
    """
    angles = equidistant(nproj=nprojs, nrevs=nrevs, radians=radians)
    assert len(angles) == nprojs * nrevs
    assert angles[0] == 0
    assert np.min(angles) >= 0
    if radians:
        assert np.max(angles) <= 2 * np.pi * nrevs
    else:
        assert np.max(angles) <= 360 * nrevs
    diffs = np.diff(angles)
    assert np.allclose(diffs, diffs[0])


@pytest.mark.parametrize(
    "nprojs, radians",
    [(13, True), (13, False), (17, False)],
    ids=["Radians", "Degrees", "17 projections and 2 revolutions"],
)
def test_golden_angle_sampling(nprojs, radians):
    angles = golden_angle(nprojs=nprojs, radians=radians)
    assert len(angles) == nprojs
    assert angles[0] == 0
    assert np.min(angles) >= 0

    if nprojs == 13 and radians:
        diffs = np.diff(angles)
        assert np.allclose(diffs, 137.5 * np.pi / 180)
    else:
        diffs = np.diff(angles)
        assert np.allclose(diffs, 137.5)


@pytest.mark.parametrize(
    "nprojs, nrevs, radians",
    [(13, 1, True), (13, 1, False), (13, 2, False), (13, 2, False)],
    ids=[
        "Radians",
        "Degrees",
        "13 projections, 2 revolutions",
        "17 projections, 2 revolutions",
    ],
)
def test_hybrid_golden_angle_sampling(nprojs, nrevs, radians):
    angles = hybrid_golden_angle(nproj=nprojs, nrevs=nrevs, radians=radians)
    assert angles.shape == (nrevs, nprojs)
    assert np.min(angles) >= 0
    if radians:
        assert np.max(angles) <= 2 * np.pi
    else:
        assert np.max(angles) <= 360
    diffs = np.diff(angles)
    for revolution in range(nrevs):
        assert np.allclose(diffs[revolution], diffs[revolution][0])
