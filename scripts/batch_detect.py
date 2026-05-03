"""Batch runner that uses the CLI to process a set of images and collect outputs."""
import argparse
import os
import subprocess
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--min-area', type=int, default=5000)
    p.add_argument('--out-dir', default='output/batch')
    args = p.parse_args()

    cmd = ['python', 'scripts/geoai_cli.py', 'batch', '--folder', args.input_dir, '--min-area', str(args.min_area), '--outdir', args.out_dir]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)

if __name__ == '__main__':
    main()
