#!/usr/bin/env python3
"""Run the same odometry configuration on multiple extracted TUM sequences."""
import argparse
import csv
import json
from pathlib import Path
from rgbd_lab.cli import main as track_command

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('recordings',nargs='+',type=Path)
    p.add_argument('--output',type=Path,default=Path('results/benchmark'))
    p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--stride-frames',type=int,default=1)
    p.add_argument('--camera',choices=['fr1-default','fr1-rgb-calibrated','fr2-default','fr3'],default='fr1-default')
    args=p.parse_args(argv)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for sequence in args.recordings:
        if not sequence.is_dir(): raise FileNotFoundError(sequence)
        target=args.output/sequence.name
        track_command(['track',str(sequence),'--camera',args.camera,'--output',str(target),
                       '--max-frames',str(args.max_frames),'--stride-frames',str(args.stride_frames)])
        result=json.loads((target/'metrics.json').read_text())
        rows.append(result)
    columns=['sequence','frames','tracked_transitions','failed_transitions','matched_frames',
             'ate_rmse_m','ate_median_m','rpe_translation_rmse_m']
    with (args.output/'summary.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    print('Saved',args.output/'summary.csv')

if __name__=='__main__':main()
