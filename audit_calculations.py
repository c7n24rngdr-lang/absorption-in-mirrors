#!/usr/bin/env python3
"""Independent checks of the ORIGINAL notebooks plus new supermirror solver.

python audit_calculations.py [--execute-notebooks]
Full notebook execution writes copies/results under /tmp, not over inputs.
"""
import argparse
import ast
import contextlib
import io
import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from supermirror import KC_NI, profile_layers, reflectivity, solve_fields

ROOT = Path(__file__).resolve().parent

def notebook_functions(path):
    """Load reviewed function definitions, without running plotting/data cells."""
    ns = {'np': np, 'torch': torch, 'plt': plt, 'Path': Path,
          'WAVELENGTH_A': 4.52, 'DZ_A': .5, 'AIR_EXTENT_A': 300., 'SUBSTRATE_EXTENT_A': 100.}
    nb = json.loads(path.read_text())
    for cell in nb['cells']:
        if cell['cell_type'] != 'code': continue
        tree = ast.parse(''.join(cell['source']))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MATERIALS' for t in node.targets):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns


def boundary_reference(q, layers):
    """Independent psi/psi' propagation, including the analytic k=0 limit.

    A 2x2 characteristic matrix per layer (not scattering recursion).
    Used only on thin few-layer samples where matrices stay well conditioned.
    """
    d = layers[:, 0]
    rho = (layers[:, 1]-layers[0,1]-1j*layers[:,2])*1e-6
    k = np.sqrt(q[:,None]**2-4*np.pi*rho)
    matrix = np.broadcast_to(np.eye(2,dtype=complex),(len(q),2,2)).copy()
    for j in range(1,len(d)-1):
        kd = k[:,j]*d[j]
        co, sk = np.cos(kd), d[j]*np.sinc(kd/np.pi)
        one = np.empty_like(matrix)
        one[:,0,0]=co; one[:,0,1]=sk
        one[:,1,0]=-k[:,j]**2*sk; one[:,1,1]=co
        matrix=one@matrix
    incident=np.c_[np.ones(len(q)),1j*q]
    reflected=np.c_[np.ones(len(q)),-1j*q]
    out=np.c_[np.ones(len(q)),1j*k[:,-1]]
    lhs=np.stack((np.einsum('nij,nj->ni',matrix,reflected),-out),axis=-1)
    rhs=-np.einsum('nij,nj->ni',matrix,incident)
    rt=np.linalg.solve(lhs,rhs[...,None])[...,0]
    r,t=rt[:,0],rt[:,1]
    # States at successive interfaces, independent of opposite-boundary amplitudes.
    state=incident+r[:,None]*reflected
    states=[state.copy()]
    for j in range(1,len(d)-1):
        kd=k[:,j]*d[j]; co=np.cos(kd);sk=d[j]*np.sinc(kd/np.pi)
        state=np.c_[co*state[:,0]+sk*state[:,1],-k[:,j]**2*sk*state[:,0]+co*state[:,1]]
        states.append(state.copy())
    return r, k[:,-1].real/q*abs(t)**2, states, k


def run_checks(ns):
    q = np.linspace(.00213,.05117,37)
    ordinary=np.array([[0,0,0],[65,9.408,.00114],[113,-1.925,.000967],[58,9.408,.00114],[0,2.074,.0000238]])
    cases={'absorbing':ordinary,'lossless':ordinary*np.array([1,1,0]),
           'nonzero_incident':ordinary+np.array([0,1.23,0]),
           'single_interface':ordinary[[0,-1]],
           'uniform_vacuum':ordinary*np.array([1,0,0]),
           'strong_absorber':np.array([[0,0,0],[200,4,1.5],[0,2,.3]])}
    summary={}
    gx,gw=np.polynomial.legendre.leggauss(96)
    for name,layers in cases.items():
        lt=torch.tensor(layers,dtype=torch.float64)
        qt=torch.tensor(q,dtype=torch.float64)
        r,k,A,B,p,_=ns['stable_multilayer_fields'](qt,lt)
        ref_r,ref_t,states,ref_k=boundary_reference(q,layers)
        t=k[:,-1].real/qt*abs(A[:,-1])**2
        integrals=ns['integrate_layers_analytic'](A,B,k,lt).numpy().T
        field_error=0.; quadrature_error=0.
        for j in range(1,len(layers)-1):
            d=layers[j,0];x=(gx+1)*d/2
            state=states[j-1]
            # psi propagated from the independently obtained interface state.
            psi=state[:,0,None]*np.cos(ref_k[:,j,None]*x)+state[:,1,None]*x*np.sinc(ref_k[:,j,None]*x/np.pi)
            integral=(abs(psi)**2*gw).sum(1)*d/2
            quadrature_error=max(quadrature_error,float(abs(integral-integrals[:,j-1]).max()))
            z=layers[1:j,0].sum()+x
            evaluated=ns['evaluate_psi_squared_stable'](A,B,k,lt,z).numpy()
            field_error=max(field_error,float(abs(evaluated-abs(psi)**2).max()))
        with contextlib.redirect_stdout(io.StringIO()):
            diag=ns['flux_balance_diagnostics'](qt,r,k,A,B,p,lt)
        metrics={'amplitude_vs_boundary_matrix':float(abs(r.numpy()-ref_r).max()),
                 'transmission_vs_boundary_matrix':float(abs(t.numpy()-ref_t).max()),
                 'field_vs_boundary_matrix':field_error,
                 'integral_vs_Gauss_quadrature_A':quadrature_error,**diag['errors']}
        assert max(metrics.values())<1e-8,(name,metrics)
        if name=='lossless': assert np.max(abs(diag['L'].numpy()))<1e-14
        f=solve_fields(q,layers[:,0],(layers[:,1]-1j*layers[:,2])*1e-6)
        np.testing.assert_allclose(f['r'],r.numpy(),rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(f['integral'],integrals,rtol=1e-9,atol=1e-9)
        summary[name]=metrics
    if 'compute_absorption' in ns:
        # Direct normalization and a linear function whose trapezoid integral is exact.
        z=np.array([0.,.5,1.5,3.,4.]);v=np.array([1+2*z,3-z/2])
        got=ns['integrate_layers'](v,z,np.array([0.,1.,4.]))
        expected=np.array([[2.,2.75],[18.,5.25]])
        np.testing.assert_allclose(got,expected,atol=1e-14)
        ans=ns['compute_absorption'](got,np.array([.1,.2]),np.array([.01,.02]))
        np.testing.assert_allclose(ans,4*np.pi*1e-6*np.array([.1,.2])[:,None]*expected/np.array([.01,.02]))
        raw=ns['stable_multilayer_fields'](q,ordinary)[0]
        legacy=ns['reflec_new_stable'](q,ordinary,bkg=-7,scale=.3)
        np.testing.assert_allclose(legacy[3].numpy(),abs(raw.numpy())**2*10**.3+1e-7)
        summary['quadrature_absorption_wrapper']='passed'
    if 'make_layers' in ns:
        for stack in [[('air',0)], [('air',0),('Ni',-1),('Si',0)], [('air',1),('Si',0)], [('air',0),('unknown',10),('Si',0)]]:
            try: ns['make_layers'](stack)
            except ValueError: pass
            else: raise AssertionError('Invalid structure accepted')
        names,layers=ns['make_layers']([('air',0),('Ni',100),('Ti',700),('Ni',500),('Si',0)])
        with contextlib.redirect_stdout(io.StringIO()):
            s=ns['calculate_structure']([('air',0),('Ni',100),('Ti',700),('Ni',500),('Si',0)],q)
            ns['run_independent_checks']([s])
        summary['configurable_structure_and_direct_solver']='passed'
    # Record known edge failures rather than hiding them as passed physical tests.
    crit_layers=np.array([[0,0,0],[100,9.408,0],[0,2.074,0]])
    critical=np.array([np.sqrt(4*np.pi*9.408e-6)])
    rr,tt,_,_=boundary_reference(critical,crit_layers)
    r,k,A,B,p,_=ns['stable_multilayer_fields'](critical,crit_layers)
    calculated_t=(k[:,-1].real/k[:,0].real*abs(A[:,-1])**2).numpy()
    summary['known_exact_critical_limit']={'reference_R':float(abs(rr[0])**2),'reference_T':float(tt[0]),
                                        'notebook_R':float(abs(r.numpy()[0])**2),'notebook_T':float(calculated_t[0]),
                                        'note':'Two-exponential basis degenerates; eps is not the analytic limit.'}
    return summary


def characterize_inputs():
    x=np.linspace(.02,5.5,12001)
    result={}
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for path in sorted(ROOT.glob('*.npy')):
        a=np.load(path,allow_pickle=False)
        high='Be' if path.name.startswith('Be') else 'Ni'
        d,rho=profile_layers(a,high=high)
        r=reflectivity(x*KC_NI,d,rho)
        detail={'assumed_material':high+'/Ti','pairs':len(a),'thickness_um':float(a[:,1:].sum()/1000),
                'periodic':bool(np.allclose(a[:,1:],a[0,1:])),
                'R_at_m':{str(m):float(reflectivity(np.array([m*KC_NI]),d,rho)[0]) for m in [1.5,2,2.5,3,4,5]}}
        if path.name=='profile_data_GRB.npy':
            bad=np.flatnonzero((x>=1)&(r<.9));detail['first_R_below_90_percent_above_m1']=float(x[bad[0]])
        result[path.name]=detail
        ax.plot(x,r,lw=.7,label=path.stem)
    ax.set(xlabel=r'$k_\perp/k_{c,Ni}$',ylabel='Specular reflectivity',ylim=(-.02,1.02),title='Supplied profiles, capture only, sharp interfaces')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(ROOT/'analysis'/'supplied_profiles.png',dpi=180)
    return result


def execute_notebooks():
    import nbformat
    from nbclient import NotebookClient
    results={}
    for path in sorted(ROOT.glob('*.ipynb')):
        work=Path(tempfile.mkdtemp(prefix=path.stem+'-'))
        for data in ROOT.glob('*.npy'): shutil.copy2(data,work/data.name)
        nb=nbformat.read(path,as_version=4)
        NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':str(work)}}).execute()
        nbformat.write(nb,work/path.name)
        cells=[c for c in nb.cells if c.cell_type=='code' and c.source.strip()]
        assert cells and all(c.execution_count is not None for c in cells)
        outputs='\n'.join(o.text for c in cells for o in c.get('outputs',[]) if o.output_type=='stream')
        results[path.name]={'executed_cells':len(cells),'status':'passed','output_directory':str(work),'text_outputs':outputs}
        print('Full execution passed:',path.name,len(cells),flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-notebooks',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    (ROOT/'analysis').mkdir(exist_ok=True)
    from periodictable.nsf import neutron_sld
    material_data={}
    for material,density in [('Ni',8.902),('Ti',4.54),('Cr',7.19),('Si',2.33),('Be',1.848)]:
        real,capture,incoherent=neutron_sld(material,density=density,wavelength=2.4)
        material_data[material]={'density_g_cm3':density,'real_micro_A_inverse_squared':real,
                                'capture_micro_A_inverse_squared':capture,'incoherent_micro_A_inverse_squared':incoherent}
    (ROOT/'analysis'/'material_sld_check.json').write_text(json.dumps({'source':'periodictable 2.1.0 neutron_sld; wavelength=2.4 angstrom','materials':material_data},indent=2)+'\n')
    report={'independent_checks':{},'supplied_profiles':characterize_inputs(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(ROOT.glob('*.ipynb'))},
            'versions':{'numpy':np.__version__,'torch':torch.__version__}}
    for path in sorted(ROOT.glob('*.ipynb')):
        report['independent_checks'][path.name]=run_checks(notebook_functions(path))
        print('Independent checks completed:',path.name,flush=True)
    if args.execute_notebooks: report['notebook_execution']=execute_notebooks()
    (ROOT/'analysis'/'audit_results.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__': main()
