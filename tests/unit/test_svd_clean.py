import numpy as np
import pytest

from omc3.harpy import clean


@pytest.mark.basic
def test_unitarity(u_mat):
    assert np.all(np.linalg.norm(u_mat, axis=0) == 1)
    assert np.all(np.linalg.norm(u_mat, axis=1) == 1)


@pytest.mark.basic
def test_remove_dominant_elements_none(u_mat, s_mat):
    u, s, u_mask = clean._remove_dominant_elements(u_mat, s_mat, None)
    assert np.all(u == u_mat)
    assert np.all(s == s_mat)
    assert np.all(u_mask)
    assert u_mask.shape == u_mat.shape


@pytest.mark.basic
def test_remove_dominant_element(u_mat, s_mat):
    u, s, u_mask = clean._remove_dominant_elements(u_mat, s_mat, 0.9, num_iter=1)
    assert ~u_mask[0, 0]
    assert np.isclose(s[0], np.sqrt(3))
    assert u[0, 0] == 0
    u_mask[0, 0] = True
    assert np.all(u_mask)


@pytest.mark.basic
def test_remove_dominant_elements_iterations(u_mat, s_mat):
    u, s, u_mask = clean._remove_dominant_elements(u_mat, s_mat, 0.8, num_iter=2)
    assert ~u_mask[2, 1]
    assert np.isclose(s[1], np.sqrt(1.6))
    assert u[1, 1] == 0
    assert u[2, 1] == 0
    u_mask[0, 0], u_mask[1, 1], u_mask[2, 1] = True, True, True
    assert np.all(u_mask)


@pytest.mark.basic
def test_remove_dominant_elements_only_max(u_mat, s_mat):
    u, s, u_mask = clean._remove_dominant_elements(u_mat, s_mat, 0.35, num_iter=1)
    assert ~u_mask[2, 3]
    assert np.isclose(u[3, 3], np.sqrt(0.4/0.55))
    assert u[2, 3] == 0
    u_mask[0, 0], u_mask[1, 1], u_mask[2, 3], u_mask[3, 2] = True, True, True, True
    assert np.all(u_mask)


@pytest.fixture
def u_mat():
    return np.sqrt(np.array([[0.97, 0.01, 0.01, 0.01],
                             [0.01, 0.65, 0.2,  0.14],
                             [0.01, 0.25, 0.29, 0.45],
                             [0.01, 0.09, 0.5,  0.4]]))


@pytest.fixture
def s_mat():
    return np.array([10, 4, 3, 1])
