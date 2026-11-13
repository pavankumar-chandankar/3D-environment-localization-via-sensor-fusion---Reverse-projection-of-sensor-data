import cv2
import numpy as np
import pytest
from rgbd_lab.calibration import Calibration
from rgbd_lab.odometry import Tracking,estimate_motion,accumulate_poses


def test_flat_rgb_fails_cleanly():
    rgb=np.zeros((64,64,3),np.uint8)
    depth=np.full((64,64),1200,dtype=np.uint16)
    cal=Calibration(90,90,32,32,depth_factor=1000)
    result=estimate_motion(rgb,depth,rgb,cal)
    assert result.success is False and result.inliers==0

def test_orb_tracking_identical_textured_views():
    rng=np.random.default_rng(7)
    rgb=rng.integers(0,256,(200,280,3),np.uint8)
    depth=np.full((200,280),1800,np.uint16)
    cal=Calibration(210,210,140,100,depth_factor=1000)
    m=estimate_motion(rgb,depth,rgb.copy(),cal,min_inliers=12)
    assert m.success,m.reason
    assert m.inliers>=12 and m.reprojection_rmse_px<0.25
    assert np.linalg.norm(m.transform_current_from_previous - np.eye(4)) < .02

def test_malformed_frame_pair_rejected():
    rgb=np.zeros((20,30,3),np.uint8)
    with pytest.raises(ValueError): estimate_motion(rgb,np.zeros((30,20),np.uint16),rgb,Calibration(10,10,0,0))

def test_pose_accumulation_convention():
    T=np.eye(4);T[0,3]=-1.0 # camera-current from camera-prev moves -1 => camera world +1
    motion=Tracking(T,45,30,25,.25,True,'tracked')
    poses,ok=accumulate_poses([motion,motion])
    assert ok.tolist()==[True,True,True]
    assert np.allclose(poses[:,0,3],[0,1,2])

def test_failure_keeps_pose_and_is_reported():
    bad=Tracking(np.eye(4),0,0,0,float('nan'),False,'no descriptors')
    p,ok=accumulate_poses([bad])
    assert not ok[-1] and np.allclose(p[0],p[1])
