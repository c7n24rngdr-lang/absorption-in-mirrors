"""Scalar neutron optics and reproducible graded Ni/Ti designs.

Lengths: angstrom; SLD: angstrom^-2; q argument: incident NORMAL k,
not reflectometry Q=2*k. exp(+ikz), rho=rho_real-i*rho_abs.
No magnetic potential, interface roughness, or diffuse scattering.
"""
from __future__ import annotations
import numpy as np

# Bulk values used by neutron_fields_configurable.ipynb (capture only).
SLD = {
    'air': 0j,
    'Ni': (9.407764749850172 - 1j * .0011404490352502582) * 1e-6,
    'Ti': (-1.9248657571990726 - 1j * .0009673155188374778) * 1e-6,
    'Si': (2.0737423003838087 - 1j * .000023757942260997373) * 1e-6,
    # Be values from the comparison cells; less precise than the Ni/Ti table.
    'Be': (9.620 - 1j * 2.6e-6) * 1e-6,
}
KC_NI = np.sqrt(4 * np.pi * SLD['Ni'].real)


def graded_profile(m: float, pairs: int, padding: float = .2) -> np.ndarray:
    """Chirped quarter-wave design; rows from illuminated surface to substrate.

    s_j = [s_min^4 + j/(N-1)*(s_max^4-s_min^4)]^(1/4),
    d_material = pi/(2*sqrt((s_j*kc_Ni)^2-4*pi*rho_material)).
    This is an explicitly specified engineering design, NOT a claimed
    implementation of the Hayter-Mook algorithm used in Kolevatov's paper.
    The fourth-power chirp allocates more bilayers to high-k weak contrast.
    Padding moves the terminal Bragg band beyond the nominal usable edge.
    Returns [one-based pair index, d_Ni_nm, d_Ti_nm].
    """
    if not np.isfinite(m) or m <= 1.02 or pairs < 2 or padding < 0:
        raise ValueError('Require m>1.02, pairs>=2, padding>=0.')
    s = np.linspace(1.02**4, (m + padding)**4, pairs)**.25
    rho = np.array([SLD['Ni'].real, SLD['Ti'].real])
    d_nm = np.pi / (2 * np.sqrt((KC_NI*s[:, None])**2 - 4*np.pi*rho)) / 10
    return np.column_stack((np.arange(1, pairs+1), d_nm))


def profile_layers(profile, high='Ni', absorption=True):
    """Return thickness_A, complex SLD; append vacuum and semi-infinite Si."""
    a = np.asarray(profile, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not len(a):
        raise ValueError('Expected nonempty (N,3) [pair, high_nm, Ti_nm].')
    if not np.isfinite(a).all() or np.any(a[:, 1:] <= 0):
        raise ValueError('Thicknesses must be finite and positive.')
    d = np.r_[0., a[:, 1:].ravel()*10, 0.]
    rho = np.r_[SLD['air'], np.tile([SLD[high], SLD['Ti']], len(a)), SLD['Si']]
    return d, rho if absorption else rho.real.astype(complex)


def _inputs(k_perp, d, rho):
    q, d, rho = np.asarray(k_perp, float), np.asarray(d, float), np.asarray(rho, complex)
    if q.ndim != 1 or not len(q) or np.any(q <= 0) or not np.isfinite(q).all():
        raise ValueError('k_perp must be a finite positive one-dimensional array.')
    if d.ndim != 1 or len(d) < 2 or rho.shape != d.shape:
        raise ValueError('Need ambient and substrate with matching thickness/SLD arrays.')
    if not np.isfinite(d).all() or not np.isfinite(rho).all():
        raise ValueError('Nonfinite layer input.')
    if d[0] != 0 or d[-1] != 0 or np.any(d[1:-1] <= 0):
        raise ValueError('End media have zero dummy thickness; finite layers have d>0.')
    if rho[0].imag != 0 or np.any(rho.imag > 0):
        raise ValueError('Need lossless ambient and passive rho=rho_real-i*rho_abs.')
    return q, d, rho-rho[0].real


def reflectivity(k_perp, d, rho):
    """Parratt recursion, O(n_k) memory; exact k_j=0 is explicitly rejected."""
    q, d, rho = _inputs(k_perp, d, rho)
    rr = np.zeros(len(q), complex)
    kr = np.sqrt(q*q-4*np.pi*rho[-1])
    for j in range(len(d)-2, -1, -1):
        kl = np.sqrt(q*q-4*np.pi*rho[j])
        if np.any(kl == 0) or np.any(kr == 0):
            raise ValueError('Exact k_j=0: use a limiting/boundary-value solution.')
        r = (kl-kr)/(kl+kr)
        load = rr*np.exp(2j*kr*d[j+1])
        rr = (r+load)/(1+r*load)
        kr = kl
    return abs(rr)**2


def solve_fields(k_perp, d, rho):
    """Stable amplitudes at opposite layer ends, currents, analytic absorption.

    psi_j(x)=A_j*exp(ik_j*x)+B_j*exp(ik_j*(d_j-x)).
    Based on continuity (Kolevatov Eqs. 9-20; Aksenov Eqs. 7-10).
    Explicitly reject the degenerate two-exponential basis at exact k_j=0.
    """
    q, d, rho = _inputs(k_perp, d, rho)
    k = np.sqrt(q[:, None]**2-4*np.pi*rho)
    if np.any(k == 0):
        raise ValueError('Exact k_j=0: use a limiting/boundary-value solution.')
    p = np.exp(1j*k*d)
    r = (k[:, :-1]-k[:, 1:])/(k[:, :-1]+k[:, 1:])
    t = 2*k[:, :-1]/(k[:, :-1]+k[:, 1:])
    effective = np.zeros_like(k)
    for j in range(len(d)-2, -1, -1):
        load = effective[:, j+1]*p[:, j+1]**2
        effective[:, j] = (r[:, j]+load)/(1+r[:, j]*load)
    A, B = np.zeros_like(k), np.zeros_like(k)
    A[:, 0], B[:, 0] = 1, effective[:, 0]
    for j in range(len(d)-1):
        load = effective[:, j+1]*p[:, j+1]**2
        A[:, j+1] = t[:, j]*A[:, j]*p[:, j]/(1+r[:, j]*load)
        B[:, j+1] = effective[:, j+1]*p[:, j+1]*A[:, j+1]
    kj, aj, bj, dj = k[:, 1:-1], A[:, 1:-1], B[:, 1:-1], d[None, 1:-1]
    beta = kj.imag
    attenuation = np.broadcast_to(dj, beta.shape).copy()
    np.divide(-np.expm1(-2*beta*dj), 2*beta, out=attenuation, where=beta != 0)
    integral = (abs(aj)**2+abs(bj)**2)*attenuation + 2*(aj*bj.conj()).real*np.exp(-beta*dj)*dj*np.sinc(kj.real*dj/np.pi)
    absorption = -4*np.pi*rho.imag[None, 1:-1]*integral/q[:, None]
    psi_l, psi_r = A+B*p, A*p+B
    deriv_l, deriv_r = 1j*k*(A-B*p), 1j*k*(A*p-B)
    jl = (psi_l.conj()*deriv_l).imag/q[:, None]
    jr = (psi_r.conj()*deriv_r).imag/q[:, None]
    return dict(r=B[:, 0], k=k, A=A, B=B, p=p, integral=integral,
                absorption=absorption, R=abs(B[:, 0])**2,
                T=k[:, -1].real/q*abs(A[:, -1])**2,
                current_loss=jl[:, 1:-1]-jr[:, 1:-1],
                jump_psi=psi_r[:, :-1]-psi_l[:, 1:],
                jump_derivative=deriv_r[:, :-1]-deriv_l[:, 1:])
