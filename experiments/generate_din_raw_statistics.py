import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa
from fastdtw import fastdtw
from scipy.signal import resample
from scipy.spatial.distance import euclidean

from neurovoc import scale_to_target_dbfs, bruce, specres, reconstruct

def compute_mcd(y1, y2, sr):
    """
    Compute MCD between two MFCC matrices using DTW alignment.
    """
    # Exclude 0th coefficient (energy)
    mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=13, hop_length=512)
    mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=13, hop_length=512)
    
    mfcc1 = mfcc1[1:].T  # shape: (frames, coefficients)
    mfcc2 = mfcc2[1:].T

    # Use DTW to align sequences
    distance, path = fastdtw(mfcc1, mfcc2, dist=euclidean)

    # Compute MCD over aligned path
    mcd_total = 0.0
    for i, j in path:
        diff = mfcc1[i] - mfcc2[j]
        mcd_total += np.sqrt(np.sum(diff ** 2))

    mcd_const = 10.0 / np.log(10) * np.sqrt(2)
    return mcd_const * mcd_total / len(path)

def main():
    normal_triplets = os.path.join(os.path.dirname(os.path.dirname(__file__)),  "data/din/triplets")
    stats = []
    for fname in tqdm(os.listdir(normal_triplets)):
        normal_trip = os.path.join(normal_triplets, fname)
        norm_y, sr = librosa.load(normal_trip, sr=None)
        norm_y = scale_to_target_dbfs(norm_y, -20)
        record = [fname]
        for model in (bruce, specres):
            neurogram = model(normal_trip)
            fy = reconstruct(neurogram, target_sr=sr)
            fy = resample(fy, norm_y.size)

            distance, path = fastdtw(norm_y, fy)
            p1, p2 = np.array(path).T
            ny_aligned = norm_y[p1]
            fy_aligned = fy[p2]
            rmse = np.mean((ny_aligned - fy_aligned)**2)
            mcd = compute_mcd(ny_aligned, fy_aligned, sr)
            record.extend([distance, rmse, mcd])
        stats.append(record)
        
    stats = pd.DataFrame(stats, columns=[
        "file", 
        "bruce_abs_dist", "bruce_rmse", "bruce_mcd",
        "specres_abs_dist", "specres_rmse", "specres_mcd"
    ])

    stats.to_csv("din_stats.csv")


if __name__ == "__main__":
    main()