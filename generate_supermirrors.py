#!/usr/bin/env python3
"""Generate and validate graded m=3/m=5 Ni/Ti supermirrors.

Run: python generate_supermirrors.py
Defaults: 300/1600 bilayers; no roughness or diffuse loss; bulk capture SLD.
Outputs use profile_data_GRB.npy's [pair, Ni_nm, Ti_nm] layout.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from supermirror import KC_NI, SLD, graded_profile, profile_layers, reflectivity, solve_fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent/'supermirrors')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    report = {'model': 'capture only, bulk nuclear SLD, sharp interfaces, vacuum/Ni-Ti/Si',
              'algorithm': 'fourth-power chirp of quarter-wave bilayers; s_min=1.02, s_max=m+0.2',
              'kc_Ni_A_inverse': float(KC_NI), 'Qc_Ni_A_inverse': float(2*KC_NI),
              'columns': ['bilayer_1_based', 'd_Ni_nm', 'd_Ti_nm'],
              'order': 'incident surface to substrate; each pair is Ni then Ti',
              'SLD_A_inverse_squared': {k: {'real': v.real, 'absorption': -v.imag} for k, v in SLD.items()},
              'designs': {}}
    for column, (m, pairs, minimum) in enumerate([(3, 300, .90), (5, 1600, .85)]):
        data = graded_profile(m, pairs)
        d, rho = profile_layers(data)
        # Dense nominal-band grid PLUS the interleaved shifted grid, no angular averaging.
        # The lower limit is finite; this does not claim a continuous-interval proof.
        scan_m = np.linspace(.02, m, 24001)
        scan_r = reflectivity(scan_m*KC_NI, d, rho)
        if scan_r.min() < minimum:
            raise RuntimeError(f'm={m} dense-grid minimum R={scan_r.min():.6f} < {minimum}')
        name = f'Ni_Ti_supermirror_m{m}.npy'
        np.save(args.output_dir/name, data, allow_pickle=False)
        np.testing.assert_array_equal(np.load(args.output_dir/name, allow_pickle=False), data)
        x = np.linspace(.02, m+.7, 5001)
        r = reflectivity(x*KC_NI, d, rho)
        lossless = reflectivity(x*KC_NI, d, rho.real.astype(complex))
        # Chunk full wave calculation to bound memory for the m=5 coating.
        curves = []
        checks = {'flux_balance': 0., 'layer_current_vs_integral': 0., 'psi_continuity': 0., 'derivative_continuity': 0., 'independent_reflectivity': 0.}
        for start in range(0, len(x), 128):
            f = solve_fields(x[start:start+128]*KC_NI, d, rho)
            pn, pt = f['absorption'][:, ::2].sum(1), f['absorption'][:, 1::2].sum(1)
            if min(f['R'].min(), f['T'].min(), f['absorption'].min()) < -1e-10:
                raise AssertionError('Negative physical probability')
            curves.append(np.c_[f['T'], pn, pt])
            checks['flux_balance'] = max(checks['flux_balance'], float(abs(f['R']+f['T']+pn+pt-1).max()))
            checks['layer_current_vs_integral'] = max(checks['layer_current_vs_integral'], float(abs(f['absorption']-f['current_loss']).max()))
            checks['psi_continuity'] = max(checks['psi_continuity'], float(abs(f['jump_psi']).max()))
            checks['derivative_continuity'] = max(checks['derivative_continuity'], float(abs(f['jump_derivative']).max()))
            checks['independent_reflectivity'] = max(checks['independent_reflectivity'], float(abs(f['R']-r[start:start+128]).max()))
        assert max(checks.values()) < 1e-8, checks
        curves = np.vstack(curves)
        np.savez_compressed(args.output_dir/f'm{m}_response.npz', k_perp=x*KC_NI, Q=2*x*KC_NI, m_coordinate=x, R=r, R_no_absorption=lossless, T=curves[:, 0], P_Ni=curves[:, 1], P_Ti=curves[:, 2])
        metrics = dict(file=name, bilayers=pairs, finite_layers=2*pairs,
                       total_thickness_um=float(data[:, 1:].sum()/1000),
                       thickness_range_nm={'Ni': [float(data[:,1].min()), float(data[:,1].max())], 'Ti': [float(data[:,2].min()), float(data[:,2].max())]},
                       validation_points=len(scan_m), validated_m_range=[.02, m],
                       required_minimum_R=minimum, minimum_R=float(scan_r.min()),
                       m_at_minimum_R=float(scan_m[scan_r.argmin()]),
                       R_at_nominal_m=float(scan_r[-1]), Q_nominal_A_inverse=float(2*m*KC_NI),
                       checks=checks)
        report['designs'][str(m)] = metrics
        axes[0,column].plot(x, lossless, color='gray', lw=.8, label='R, capture disabled')
        axes[0,column].plot(x, r, lw=.8, label='R, capture included')
        axes[0,column].plot(x, curves[:,0], lw=.8, label='T into Si')
        axes[0,column].plot(x, curves[:,1]+curves[:,2], lw=.8, label='Capture in coating')
        axes[0,column].axvline(m, ls='--', color='black', lw=.8)
        axes[0,column].set(title=f'm={m}: {pairs} graded Ni/Ti bilayers', xlabel=r'$k_\perp/k_{c,Ni}=Q/Q_{c,Ni}$', ylabel='Fraction of incident normal flux', ylim=(-.02,1.02))
        axes[0,column].legend(fontsize=8)
        axes[1,column].plot(data[:,0], data[:,1], label='Ni')
        axes[1,column].plot(data[:,0], data[:,2], label='Ti')
        axes[1,column].set(xlabel='Bilayer from illuminated surface', ylabel='Thickness (nm)', yscale='log')
        axes[1,column].legend()
        print(json.dumps(metrics, indent=2), flush=True)
    for ax in axes.flat: ax.grid(alpha=.2)
    fig.suptitle('Ideal graded supermirrors: nuclear potential, capture, zero roughness')
    fig.savefig(args.output_dir/'design_validation.png', dpi=180)
    fig.savefig(args.output_dir/'design_validation.pdf')
    (args.output_dir/'design_metadata.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
