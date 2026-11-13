import numpy as np
import pytest
from rgbd_lab.evaluation import align_se3,trajectory_metrics


def test_rigid_alignment_recovers_translation():
    xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    aligned,T=align_se3(xyz,xyz+[10,20,30])
    assert np.allclose(aligned,xyz+[10,20,30])
    assert np.isclose(np.linalg.det(T[:3,:3]),1.0)

def test_alignment_does_not_absorb_scale():
    xyz=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
    aligned,_=align_se3(xyz,xyz*2)
    assert not np.allclose(aligned,xyz*2)

def test_perfect_trajectory_rpe_zero():
    times=np.arange(6,dtype=float)
    p=np.repeat(np.eye(4)[None],6,axis=0)
    p[:,0,3]=np.arange(6)/10
    m=trajectory_metrics(times[1:5],p[1:5],times,p,maximum_gap=1,rpe_step=1)
    assert m['matched_frames']==4
    assert m['ate_rmse_m'] < 1e-10
    assert m['rpe_translation_rmse_m']<1e-10

def test_drift_has_positive_ate():
    times=np.arange(6,dtype=float)
    actual=np.repeat(np.eye(4)[None],6,axis=0)
    estimate=actual.copy(); estimate[:,0,3]=np.arange(6)*.1
    m=trajectory_metrics(times[1:5],estimate[1:5],times,actual,maximum_gap=1)
    assert m['ate_rmse_m']>.05

def test_not_enough_gt_rejected():
    t=np.array([1.,2.,3.]);poses=np.repeat(np.eye(4)[None],3,axis=0)
    with pytest.raises(ValueError): trajectory_metrics(np.array([1.]),poses[:1],t,poses)
