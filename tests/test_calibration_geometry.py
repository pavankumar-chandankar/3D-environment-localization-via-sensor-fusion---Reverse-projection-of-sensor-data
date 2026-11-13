from pathlib import Path
import numpy as np
import pytest
from rgbd_lab.calibration import Calibration, CAMERAS
from rgbd_lab.geometry import backproject, color_cloud, voxel_reduce, transform_points, write_ply


def test_tum_depth_convention():
    z = CAMERAS['fr1-default'].metric_depth(np.array([[0, 5000, 40000]], dtype=np.uint16))
    assert np.allclose(z, [[0, 1, 0]])

@pytest.mark.parametrize('field,value', [('fx',0),('fy',-2),('depth_factor',0),('max_depth_m',0.01),('cx',float('nan'))])
def test_invalid_calibration(field,value):
    d = dict(fx=100., fy=100., cx=2., cy=1., min_depth_m=0.1, max_depth_m=4.)
    d[field] = value
    with pytest.raises(ValueError): Calibration(**d)

def test_backprojection_center_and_depth(calibration):
    depth = np.zeros((3,5), dtype=np.uint16)
    depth[1,2] = 1000
    pts, pix = backproject(depth,calibration)
    assert pts.shape == (1,3)
    assert np.allclose(pts[0], [0,0,1]); assert pix.tolist() == [[1,2]]

def test_backprojection_homogeneous(calibration):
    depth = np.array([[2000,0],[0,2000]], dtype=np.uint16)
    pts,_ = backproject(depth,calibration)
    assert np.allclose(pts[0], [-0.04,-0.02,2])
    assert np.allclose(pts[1], [-0.02,0.0,2])

def test_color_index_correspondence(calibration):
    rgb = np.zeros((3,5,3),dtype=np.uint8)
    rgb[1,2] = [3,6,9]
    z = np.zeros((3,5),dtype=np.uint16); z[1,2]=1000
    xyz,col = color_cloud(rgb,z,calibration,stride=1)
    assert xyz.shape == (1,3)
    assert col.tolist() == [[3,6,9]]

def test_stride_rejects_zero(calibration):
    with pytest.raises(ValueError): backproject(np.ones((2,2),dtype=np.uint16),calibration,stride=0)

def test_broken_depth_shape(calibration):
    with pytest.raises(ValueError): backproject(np.ones((2,2,3),dtype=np.uint16),calibration)

def test_misaligned_rgb_depth(calibration):
    with pytest.raises(ValueError): color_cloud(np.zeros((3,3,3),np.uint8),np.ones((4,4),np.uint16),calibration)

def test_voxel_reduction_averages_colors():
    pts = np.array([[.01,.01,.01],[.02,.02,.02],[1.,1.,1.]])
    col = np.array([[0,20,30],[20,40,50],[200,210,220]],dtype=np.uint8)
    xyz,rgb = voxel_reduce(pts,col,.1)
    assert xyz.shape == (2,3)
    assert [10,30,40] in rgb.tolist()

def test_no_voxels_preserves_empty_shape():
    a,b = voxel_reduce(np.empty((0,3)),np.empty((0,3),np.uint8))
    assert a.shape == b.shape == (0,3)

def test_reprojection_to_world():
    T=np.eye(4);T[:3,3]=[1,2,3]
    assert np.allclose(transform_points(np.array([[1,1,1]]),T),[[2,3,4]])

def test_export_ply(tmp_path):
    dst=tmp_path/'cloud.ply'
    write_ply(dst,np.array([[1.,2.,3.]]),np.array([[20,30,40]],np.uint8))
    text=dst.read_text()
    assert 'element vertex 1' in text and '1.000000 2.000000 3.000000 20 30 40' in text
