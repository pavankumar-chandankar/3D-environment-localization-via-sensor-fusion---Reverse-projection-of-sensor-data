import json
from pathlib import Path
from rgbd_lab.cli import main


def test_inspect_reads_real_tum_format(recording,capsys):
    main(['inspect',str(recording)])
    j=json.loads(capsys.readouterr().out)
    assert j['pairs']==4 and j['image_resolution']==[220,160]
    assert j['groundtruth_interpolatable_frames']>=3

def test_cloud_exports_ply(recording,tmp_path,capsys):
    out=tmp_path/'out.ply'
    main(['cloud',str(recording),'--output',str(out),'--voxel','.1'])
    j=json.loads(capsys.readouterr().out)
    assert j['exported_points']>0
    assert out.read_text().startswith('ply\n')

def test_end_to_end_orb_track(recording,tmp_path,capsys):
    output=tmp_path/'results'
    main(['track',str(recording),'--max-frames','4','--output',str(output),'--min-inliers','10'])
    j=json.loads(capsys.readouterr().out)
    assert j['frames']==4 and j['tracked_transitions']==3,j
    assert (output/'trajectory.txt').is_file()
    assert (output/'tracking.csv').is_file()
    assert (output/'metrics.json').is_file()

def test_reconstruct_groundtruth_poses(recording,tmp_path,capsys):
    out=tmp_path/'fused.ply'
    main(['reconstruct',str(recording),'--pose-source','groundtruth','--every','1',
          '--output',str(out),'--max-frames','4'])
    j=json.loads(capsys.readouterr().out)
    assert out.exists() and j['points_exported']>0

def test_reconstruct_estimated_trajectory(recording,tmp_path,capsys):
    output=tmp_path/'track'
    main(['track',str(recording),'--output',str(output)])
    capsys.readouterr()
    out=tmp_path/'trajectory_map.ply'
    main(['reconstruct',str(recording),'--trajectory',str(output/'trajectory.txt'),
          '--output',str(out)])
    j=json.loads(capsys.readouterr().out)
    assert j['pose_source']=='estimated'
