import sys, json, warnings
import numpy as np
REPO='/mnt/mib-dbia08_1/users/noa.ganeko/projects/cell-registration-prototype-gpu'
sys.path.insert(0, REPO); sys.path.insert(0, REPO+'/analysis/adaptive-registration-validation-20261111')
import common as C11
from src.registration_research import candidate_adapter as CA, cell_correspondence as CC
from src.registration_research.legacy import modules
E,_,V = modules(); import ot
he, mv = C11.DerivedSide('031-6_2','HE'), C11.DerivedSide('031-6_2','IF')
c = CA.from_benchmark_result(REPO+'/analysis/adaptive-registration-validation-20261111/runs/P6_2_orig__D/result.json')[0]
ov = CC.overlap_subset(c.M_final, c.t_final, he.points, he.extent, mv.points, mv.extent)
f, m = he.points[ov['fixed_idx']], ov['moving_mapped'][ov['moving_idx']]
cfg = V.UOT_CONFIGS['UOT conservative']
n = max(len(m), len(f)); a = np.ones(len(m))/n; b = np.ones(len(f))/n
from scipy.spatial.distance import cdist
M = cdist(m, f, 'sqeuclidean')/(2*cfg['cost_scale_um']**2)
plan, log = ot.unbalanced.sinkhorn_unbalanced(a, b, M, reg=0.1, reg_m=0.5, method=cfg['solver'], reg_type='kl', numItermax=2000, stopThr=1e-7, log=True)
print(len(m), len(f), {k:(v if not hasattr(v,'__len__') else (len(v), v[-3:] if len(v) else v)) for k,v in log.items() if k in ('err','niter')}, np.isfinite(plan).all(), plan.sum())
print('min cost row', np.sort(M.min(1))[-5:])
