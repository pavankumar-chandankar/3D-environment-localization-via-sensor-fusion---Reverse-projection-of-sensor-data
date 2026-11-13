from pathlib import Path
import numpy as np
import pytest
from rgbd_lab.association import read_index,pair_timestamps,load_recording,read_groundtruth,interpolate_poses


def test_greedy_one_to_one():
    a=[(1.,'a'),(1.003,'b')]
    b=[(1.002,'a'),(1.05,'b')]
    assert pair_timestamps(a,b,tolerance=.01) == [(1,0)]

def test_pair_orders_by_rgb_time():
    a=[(2.,'x'),(1.,'y')]; b=[(2.,'u'),(1.,'v')]
    assert len(pair_timestamps(a,b))==2

def test_no_association_beyond_tolerance():
    assert pair_timestamps([(1.,'a')],[(2.,'b')])==[]

def test_index_comment_ordering(tmp_path):
    file=tmp_path/'rgb.txt'; file.write_text('# comment\n2 rgb/z.png\n1 rgb/a.png\n')
    assert read_index(file) == [(1.,'rgb/a.png'),(2.,'rgb/z.png')]

def test_reject_path_escape(tmp_path):
    file=tmp_path/'rgb.txt';file.write_text('1 ../outside.png\n')
    with pytest.raises(ValueError,match='path'): read_index(file)

def test_reject_duplicate_timestamps(tmp_path):
    file=tmp_path/'rgb.txt';file.write_text('1 rgb/a.png\n1 rgb/b.png\n')
    with pytest.raises(ValueError,match='duplicate'): read_index(file)

def test_recording_end_to_end(recording):
    pairs=load_recording(recording)
    assert len(pairs)==4
    assert pairs[0].rgb_file.exists() and pairs[0].depth_file.exists()
    assert abs(pairs[0].rgb_t - pairs[0].depth_t) < .005

def test_read_groundtruth(recording):
    times,poses=read_groundtruth(recording/'groundtruth.txt')
    assert len(times)==4
    assert np.allclose(poses[0],np.eye(4))

def test_gt_interpolation():
    t=np.array([1.,3.]);poses=np.repeat(np.eye(4)[None],2,axis=0)
    poses[1,:3,3]=[2,0,0]
    out,valid=interpolate_poses(np.array([2.,4.]),t,poses,max_gap=2)
    assert valid.tolist()==[True,False]
    assert np.isclose(out[0,0,3],1)

def test_gap_does_not_extrapolate():
    t=np.array([1.,3.]);poses=np.repeat(np.eye(4)[None],2,axis=0)
    _,valid=interpolate_poses(np.array([2.]),t,poses,max_gap=.1)
    assert valid.tolist()==[False]
