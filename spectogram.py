import librosa
import librosa.display
import os
import shutil
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from config import config 

spect_config = config["spectogram"]
sr_ = spect_config["sr"]
n_fft_ = spect_config["n_fft"]
hop_length_ = spect_config["hop_length"]
n_mels_ = spect_config["n_mels"]
fmin_ = spect_config["fmin"]
fmax_ = spect_config["fmax"]
top_db_ = spect_config["top_db"]

def get_melspectrogram_db(file_path, sr, n_fft, hop_length, n_mels, fmin, fmax, top_db):
    wav,sr = librosa.load(file_path,sr=sr)
    if wav.shape[0]<5*sr:
        wav=np.pad(wav,int(np.ceil((5*sr-wav.shape[0])/2)),mode='reflect')
    else:
        wav=wav[:5*sr]
    spec=librosa.feature.melspectrogram(wav, sr=sr, n_fft=n_fft,
              hop_length=hop_length,n_mels=n_mels,fmin=fmin,fmax=fmax)
    spec_db=librosa.power_to_db(spec,top_db=top_db)
    return spec_db

def spec_to_image(path, eps=1e-6):
    spec = get_melspectrogram_db(path, sr_, n_fft_, hop_length_, n_mels_, fmin_, fmax_, top_db_)
    mean = spec.mean()
    std = spec.std()
    spec_norm = (spec - mean) / (std + eps)
    spec_min, spec_max = spec_norm.min(), spec_norm.max()
    spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
    spec_scaled = spec_scaled.astype(np.uint8)
    return np.stack([spec_scaled,spec_scaled,spec_scaled], axis=-1)