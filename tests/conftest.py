from pathlib import Path
import cv2
import numpy as np
import pytest

@pytest.fixture
def calibration():
    from rgbd_lab.calibration import Calibration
    return Calibration(100.0, 100.0, 2.0, 1.0, depth_factor=1000, min_depth_m=0.1, max_depth_m=4)

@pytest.fixture
def recording(tmp_path: Path):
    """Tiny generated *test fixture* using the public TUM folder convention."""
    (tmp_path / 'rgb').mkdir(); (tmp_path / 'depth').mkdir()
    rng = np.random.default_rng(114)
    rgb = rng.integers(0, 256, size=(160, 220, 3), dtype=np.uint8)
    depth = np.full((160, 220), 8000, dtype=np.uint16)  # 1.6 m with TUM factor 5000
    rgb_index, depth_index, truth = ['# TUM rgb timestamps'], ['# depth timestamps'], ['# gt timestamps']
    for i in range(4):
        t = 1305000000 + 0.04 * i
        cv2.imwrite(str(tmp_path / 'rgb' / f'{i}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tmp_path / 'depth' / f'{i}.png'), depth)
        rgb_index.append(f'{t:.8f} rgb/{i}.png')
        depth_index.append(f'{t+0.003:.8f} depth/{i}.png')
        truth.append(f'{t:.8f} 0 0 0 0 0 0 1')
    (tmp_path / 'rgb.txt').write_text('\n'.join(rgb_index) + '\n')
    (tmp_path / 'depth.txt').write_text('\n'.join(depth_index) + '\n')
    (tmp_path / 'groundtruth.txt').write_text('\n'.join(truth) + '\n')
    return tmp_path
