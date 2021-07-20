#!/usr/bin/env python3
import argparse
import librosa
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger
import soundfile as sf
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('input_csv')
parser.add_argument('-o', '--output', type=str, required=True, help='Output data hdf5')
parser.add_argument('-c', type=int, default=4)
parser.add_argument('-sr', type=int, default=16000)
parser.add_argument('-sep', default='\s+', type=str)
args = parser.parse_args()

df = pd.read_csv(args.input_csv, sep=args.sep)
assert 'filename' in df.columns, "Header needs to contain 'filename'"

def read_wav(fname):
    y, sr = sf.read(fname, dtype='float32')
    if y.ndim > 1:
        # Merge channels
        y = y.mean(-1)
    if sr != args.sr:
        y = librosa.resample(y, sr, args.sr)
    return y


with h5py.File(args.output, 'w') as store:
    for fname in tqdm(df['filename'].unique()):
        feat = read_wav(fname)
        if feat is not None:
            try:
                store[fname] = feat
            except OSError:
                logger.warning(f"Warning, {fname} already exists!")
