import numpy as np
import pytest
from rgbd_lab.imu import read_accelerometer,accelerometer_summary

def test_accelerometer_diagnostics(tmp_path):
    file=tmp_path/'accelerometer.txt'
    file.write_text('# motion\n1.00 0 0 9.81\n1.01 0 0 9.8\n1.02 0 0 9.7\n')
    samples=read_accelerometer(file)
    result=accelerometer_summary(samples)
    assert result['samples']==3 and result['median_acceleration_magnitude']>9

def test_out_of_order_accelerometer_rejected(tmp_path):
    f=tmp_path/'bad.txt';f.write_text('2 0 0 9.8\n1 0 0 9.8\n')
    with pytest.raises(ValueError):read_accelerometer(f)

def test_nonfinite_accelerometer_rejected(tmp_path):
    f=tmp_path/'bad.txt';f.write_text('1 0 0 nan\n2 0 0 9.8\n')
    with pytest.raises(ValueError):read_accelerometer(f)
