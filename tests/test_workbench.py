import src.workbench as workbench
import pytest
from numpy import nan

EPS = 1E-03


def eps(x, xref):
    """
    Calculate absolute deviation.
    :param x: value
    :param xref: ground truth
    :return: |x - xref|
    """
    if xref == 0:
        ValueError('xref cannot be zero')
    else:
        return abs(x-xref)


@pytest.mark.parametrize('x, xref, result',
                        [
                            (3, 2, 3-2),
                            (5, 10, 10-5),
                            (1.9995, 2.0000, 2.0000-1.9995),
                        ])
def test_eps(x, xref, result):
    assert eps(x, xref) == result


@pytest.mark.parametrize('k, M, p_ratio, T_ratio, A_ratio, pA_ratio, nu, mu',
                         [
                             (1.4, 0.50, 0.84302, 0.95238, 1.33984, 1.12951, nan, nan),
                             (1.4, 2.00, 0.12780, 0.55556, 1.68750, 0.21567, 26.37976, 30.0000)
                         ])
def test_isentropic(k, M, p_ratio, T_ratio, A_ratio, pA_ratio, nu, mu):
    assert eps( workbench.isentropic('T', k=k, M=M), T_ratio ) < EPS
    assert eps( workbench.isentropic('p', k=k, M=M), p_ratio ) < EPS
    assert eps( workbench.isentropic('A', k=k, M=M), A_ratio ) < EPS
    if M < 1:
        assert eps( workbench.isentropic('M', k=k, A_ratio=A_ratio, regime='subsonic'), M) < EPS
    elif M > 1:
        assert eps( workbench.isentropic('M', k=k, A_ratio=A_ratio, regime='supersonic'), M) < EPS


@pytest.mark.parametrize('k, Ms, Msl, p_ratio, T_ratio, vel_ratio, pt_ratio, pt2_to_p1',
                         [
                             (1.4, 1.50, 0.70109, 2.45833, 1.32022, 0.69444, 0.92979, 3.41327),
                             (1.4, 2.00, 0.57735, 4.50000, 1.68750, 1.25000, 0.72087, 5.64044),
                             (1.4, 3.00, 0.47519, 10.33333, 2.67901, 2.22222, 0.32834, 12.06096),
                         ])
def test_nshock(k, Ms, Msl, p_ratio, T_ratio, vel_ratio, pt_ratio, pt2_to_p1):
    assert eps( workbench.nshock('M', Ms, k), Msl ) < EPS
    assert eps( workbench.nshock('p', Ms, k), p_ratio ) < EPS
    assert eps( workbench.nshock('pt', Ms, k), pt_ratio ) < EPS


@pytest.mark.parametrize('k, M, T_ratio, p_ratio, pt_ratio, v_ratio, fLmax_to_D, Smax_to_R',
                         [
                             (1.4, 0.50, 1.14286, 2.13809, 1.33984, 0.53452, 1.06906, 0.29255),
                             (1.4, 0.75, 1.07865, 1.38478, 1.06242, 0.77894, 0.12728, 0.06055),
                             (1.4, 2.00, 0.66667, 0.40825, 1.68750, 1.63299, 0.30500, 0.52325),
                             (1.4, 3.00, 0.42857, 0.21822, 4.23457, 1.96396, 0.52216, 1.44328),
                         ])
def test_fanno(k, M, T_ratio, p_ratio, pt_ratio, v_ratio, fLmax_to_D, Smax_to_R):
    assert eps( workbench.fanno('T', M, k), T_ratio ) < EPS
    assert eps( workbench.fanno('p', M, k), p_ratio ) < EPS
    assert eps( workbench.fanno('pt', M, k), pt_ratio ) < EPS
    assert eps( workbench.fanno('fld', M, k), fLmax_to_D ) < EPS
    assert eps( workbench.fanno('fld', fLmax_to_D, k, inv=True), M ) < EPS


