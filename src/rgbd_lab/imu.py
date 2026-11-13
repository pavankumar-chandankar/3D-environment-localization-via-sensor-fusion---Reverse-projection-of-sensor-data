"""Optional TUM Kinect accelerometer diagnostics (not six-axis inertial fusion)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def read_accelerometer(path: Path) -> np.ndarray:
    """Read timestamp ax ay az, reject missing/nonfinite samples."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts)!=4:
            raise ValueError(f"accelerometer line {number}: expected t ax ay az")
        row = [float(v) for v in parts]
        if not np.isfinite(row).all():
            raise ValueError(f"accelerometer line {number}: nonfinite sample")
        rows.append(row)
    samples = np.asarray(rows, dtype=np.float64).reshape(-1,4)
    if len(samples) < 2 or np.any(np.diff(samples[:,0]) <= 0):
        raise ValueError("Need at least two ordered accelerometer samples")
    return samples

def accelerometer_summary(samples: np.ndarray) -> dict:
    if samples.ndim!=2 or samples.shape[1]!=4 or len(samples)<2:
        raise ValueError("Invalid accelerometer array")
    magnitude=np.linalg.norm(samples[:,1:],axis=1)
    return {"samples":int(len(samples)),
            "sample_rate_median_hz":float(1/np.median(np.diff(samples[:,0]))),
            "median_acceleration_magnitude":float(np.median(magnitude)),
            "p95_acceleration_magnitude":float(np.percentile(magnitude,95)),
            "note":"Exploratory accelerometer diagnostics only. No pose integration or 6-DoF IMU fusion."}
