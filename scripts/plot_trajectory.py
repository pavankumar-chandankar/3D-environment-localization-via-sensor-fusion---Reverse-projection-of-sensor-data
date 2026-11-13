#!/usr/bin/env python3
"""Plot a TUM-format estimated trajectory with ground-truth rigid alignment."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from rgbd_lab.association import read_groundtruth,interpolate_poses
from rgbd_lab.evaluation import align_se3

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--estimated',type=Path,required=True)
    parser.add_argument('--groundtruth',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gt-max-gap',type=float,default=0.05)
    args=parser.parse_args(argv)
    et,ep=read_groundtruth(args.estimated)
    gt,gp=read_groundtruth(args.groundtruth)
    interp,mask=interpolate_poses(et,gt,gp,max_gap=args.gt_max_gap)
    if mask.sum()<3: raise ValueError('Not enough ground-truth samples for alignment')
    aligned,_=align_se3(ep[mask,:3,3],interp[mask,:3,3])
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(interp[mask,0,3],interp[mask,1,3],label='Motion capture',linewidth=1.5)
    ax.plot(aligned[:,0],aligned[:,1],label='RGB-D odometry',linewidth=1.5)
    ax.set(xlabel='World X [m]',ylabel='World Y [m]',title='Top-view camera trajectory')
    ax.axis('equal'); ax.grid(alpha=.3); ax.legend()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(args.output,dpi=160);plt.close(fig)
    print(args.output)

if __name__=='__main__':main()
