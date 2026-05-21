import librosa
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def extract_advanced_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    if len(y) < 2048:
        raise ValueError("Audio is too short")

    features = []

    # 1. ZCR & RMS Energy
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    features.extend([np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)])

    # 2. Pitch (F0)
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        pitch_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0
        pitch_std = np.nanstd(f0) if np.any(~np.isnan(f0)) else 0.0
    except:
        pitch_mean, pitch_std = 0.0, 0.0
    features.extend([pitch_mean, pitch_std])

    # 3. Frequency Domain: Spectral Centroid & Rolloff
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend(
        [np.mean(centroid), np.std(centroid), np.mean(rolloff), np.std(rolloff)]
    )

    # 4. Frequency Domain: Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend([np.mean(chroma_stft), np.std(chroma_stft)])

    # 5. MFCC and MFCC Deltas
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)

    features.extend(np.mean(mfccs, axis=1).tolist())
    features.extend(np.std(mfccs, axis=1).tolist())
    features.extend(np.mean(delta_mfccs, axis=1).tolist())

    return np.array(features)


# --- DIRECTORY PROCESSING ---
# Klasör yolunu kendi sistemine göre güncelle
DATASET_DIR = r"C:\Users\emirc\Desktop\ISTUN\25-26\Bahar\signals-and-systems\midterm-project\midterm-2\Dataset"

search_pattern = os.path.join(DATASET_DIR, "**", "*.wav")
audio_files = glob.glob(search_pattern, recursive=True)

all_data = []
print(f"Found {len(audio_files)} audio files. Starting ADVANCED extraction...")

for file in tqdm(audio_files, desc="Processing"):
    try:
        features = extract_advanced_features(file)
        file_name = os.path.basename(file)
        group_folder = os.path.basename(os.path.dirname(file))

        row = [group_folder, file_name] + features.tolist()
        all_data.append(row)
    except Exception as e:
        pass

# Kolon isimlerini oluşturma
columns = [
    "Group",
    "Filename",
    "ZCR_mean",
    "ZCR_std",
    "RMS_mean",
    "RMS_std",
    "Pitch_mean",
    "Pitch_std",
    "Centroid_mean",
    "Centroid_std",
    "Rolloff_mean",
    "Rolloff_std",
    "Chroma_mean",
    "Chroma_std",
]
columns += [f"MFCC_mean_{i}" for i in range(1, 14)]
columns += [f"MFCC_std_{i}" for i in range(1, 14)]
columns += [f"MFCC_delta_{i}" for i in range(1, 14)]

df = pd.DataFrame(all_data, columns=columns)
df.to_csv("advanced_features.csv", index=False)
print("\nPhase 2 Feature extraction complete. Saved to advanced_features.csv")
