import src.workbench as wb
import pytest
from pytest import approx
from numpy import nan

EPS = 1E-05
EPS_M = 5E-3

@pytest.mark.parametrize('input, output',
                        [
                            ({'k': 1.4}, 1.4),
                            ({'M': 3}, 1.4),
                            ({'k': 1.1}, 1.1),
                        ])
def test_get_k(input, output):
    assert wb.get_k(input) == output


@pytest.mark.parametrize('k, M, p_ratio, T_ratio, A_ratio, pA_ratio, nu, mu',
                         [
                             (1.4, 0.50, 0.84302, 0.95238, 1.33984, 1.12951, nan, nan),
                             (1.4, 2.00, 0.12780, 0.55556, 1.68750, 0.21567, 26.37976, 30.0000)
                         ])
def test_isentropic(k, M, p_ratio, T_ratio, A_ratio, pA_ratio, nu, mu):
    assert wb.isentropic('T', k=k, M=M) == approx(T_ratio, abs=EPS)
    assert wb.isentropic('p', k=k, M=M) == approx(p_ratio, abs=EPS)
    assert wb.isentropic('A', k=k, M=M) == approx(A_ratio, abs=EPS)

    if M < 1:
        assert wb.isentropic('M', k=k, A_ratio=A_ratio, regime='subsonic') == approx(M, abs=EPS)
    elif M > 1:
        assert wb.isentropic('M', k=k, A_ratio=A_ratio, regime='supersonic') == approx(M, abs=EPS)


@pytest.mark.parametrize('k, Ms, Msl, p_ratio, T_ratio, vel_ratio, pt_ratio, pt2_to_p1',
                         [
                             (1.4, 1.50, 0.70109, 2.45833, 1.32022, 0.69444, 0.92979, 3.41327),
                             (1.4, 2.00, 0.57735, 4.50000, 1.68750, 1.25000, 0.72087, 5.64044),
                             (1.4, 3.00, 0.47519, 10.33333, 2.67901, 2.22222, 0.32834, 12.06096),
                         ])
def test_nshock(k, Ms, Msl, p_ratio, T_ratio, vel_ratio, pt_ratio, pt2_to_p1):
    assert wb.nshock('M', Ms=Ms, k=k) == approx(Msl, abs=EPS)
    assert wb.nshock('p', Ms=Ms, k=k) == approx(p_ratio, abs=EPS)
    assert wb.nshock('pt', Ms=Ms, k=k) == approx(pt_ratio, abs=EPS)


@pytest.mark.parametrize('k, M, T_ratio, p_ratio, pt_ratio, v_ratio, fLmax_to_D, Smax_to_R',
                         [
                             (1.4, 0.20, 1.19048, 5.45545, 2.96352, 0.21822, 14.53327, 1.08638),
                             (1.4, 0.25, 1.18519, 4.35465, 2.40271, 0.27217, 8.48341, 0.87660),
                             (1.4, 0.50, 1.14286, 2.13809, 1.33984, 0.53452, 1.06906, 0.29255),
                             (1.4, 0.75, 1.07865, 1.38478, 1.06242, 0.77894, 0.12728, 0.06055),
                             (1.4, 1.20, 0.93168, 0.80436, 1.03044, 1.15828, 0.03364, 0.02999),
                             (1.4, 1.80, 0.72816, 0.47407, 1.43898, 1.53598, 0.24189, 0.36394),
                             (1.4, 2.00, 0.66667, 0.40825, 1.68750, 1.63299, 0.30500, 0.52325),
                             (1.4, 3.00, 0.42857, 0.21822, 4.23457, 1.96396, 0.52216, 1.44328),
                         ])
def test_fanno(k, M, T_ratio, p_ratio, pt_ratio, v_ratio, fLmax_to_D, Smax_to_R):
    assert wb.fanno('T', M=M, k=k) == approx(T_ratio, abs=EPS)
    assert wb.fanno('p', M=M, k=k) == approx(p_ratio, abs=EPS)
    assert wb.fanno('pt', M=M, k=k) == approx(pt_ratio, abs=EPS)
    assert wb.fanno('fld', M=M, k=k) == approx(fLmax_to_D, abs=EPS)
    if M < 1:
        assert wb.fanno('M', fld=fLmax_to_D, k=k, regime='subsonic') == approx(M, abs=EPS)
    elif M > 1:
        assert wb.fanno('M', fld=fLmax_to_D, k=k, regime='supersonic') == approx(M, abs=EPS*2)


@pytest.mark.parametrize('k, M, Tt_ratio, T_ratio, p_ratio, pt_ratio, v_ratio, Smax_to_R',
                         [
                             # (1.4, 0.18, 0.14324, 0.17078, 2.29586, 1.24059, 0.07439, 7.01694),
                             # (1.4, 0.19, 0.15814, 0.18841, 2.28454, 1.23765, 0.08247, 6.66813),
                             # (1.4, 0.20, 0.17355, 0.20661, 2.27273, 1.23460, 0.09091, 6.34018),
                             (1.4, 0.96, 0.99883, 1.01205, 1.04793, 1.00078, 0.96577, 0.00488),
                             (1.4, 0.97, 0.99935, 1.00929, 1.03571, 1.00044, 0.97450, 0.00271),
                             # (1.4, 0.98, 0.99971, 1.00636, 1.02365, 1.00019, 0.98311, 0.00119),
                             (1.4, 1.10, 0.99392, 0.96031, 0.89087, 1.00486, 1.07795, 0.02618),
                             (1.4, 1.11, 0.99275, 0.95577, 0.88075, 1.00588, 1.08518, 0.03135),
                             (1.4, 1.12, 0.99148, 0.95115, 0.87078, 1.00699, 1.09230, 0.03692),
                             (1.4, 2.50, 0.71006, 0.37870, 0.24615, 2.22183, 1.53846, 1.99676),
                             (1.4, 1.50, 0.90928, 0.75250, 0.57831, 1.12155, 1.30120, 0.44758),
                             (1.4, 1.51, 0.90676, 0.74732, 0.57250, 1.12649, 1.30536, 0.46169),
                             (1.4, 1.52, 0.90424, 0.74215, 0.56676, 1.13153, 1.30945, 0.47589),
                             (1.4, 2.50, 0.71006, 0.37870, 0.24615, 2.22183,  1.53846, 1.99676),
                         ])
def test_rayleigh(k, M, Tt_ratio, T_ratio, p_ratio, pt_ratio, v_ratio, Smax_to_R):
    assert wb.rayleigh('Tt', M=M, k=k) == approx(Tt_ratio, abs=EPS)
    assert wb.rayleigh('T', M=M, k=k) == approx(T_ratio, abs=EPS)
    assert wb.rayleigh('p', M=M, k=k) == approx(p_ratio, abs=EPS)
    assert wb.rayleigh('pt', M=M, k=k) == approx(pt_ratio, abs=EPS)
    # assert wb.rayleigh('v', M=M, k=k) == approx(v_ratio, abs=EPS)
    # assert wb.rayleigh('s', M=M, k=k) == approx(Smax_to_R, abs=EPS)
    if M < 1:
        assert wb.rayleigh('M', Tt=Tt_ratio, k=k, regime='subsonic') == approx(M, abs=EPS_M)
        assert wb.rayleigh('M', T=T_ratio, k=k, regime='subsonic') == approx(M, abs=EPS_M)
        assert wb.rayleigh('M', p=p_ratio, k=k, regime='subsonic') == approx(M, abs=EPS_M)
        assert wb.rayleigh('M', pt=pt_ratio, k=k, regime='subsonic') == approx(M, abs=EPS_M)
    elif M > 1:
        assert wb.rayleigh('M', Tt=Tt_ratio, k=k, regime='supersonic') == approx(M, abs=EPS_M)
        assert wb.rayleigh('M', T=T_ratio, k=k, regime='supersonic') == approx(M, abs=EPS_M)
        assert wb.rayleigh('M', p=p_ratio, k=k, regime='supersonic') == approx(M, abs=EPS_M)
        assert wb.rayleigh('M', pt=pt_ratio, k=k, regime='supersonic') == approx(M, abs=EPS_M)



