import librosa
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm
import warnings

# Gereksiz uyarı mesajlarını kapatarak terminali temiz tutalım
warnings.filterwarnings('ignore')


def extract_features(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=None)

    # Ses çok kısaysa veya boşsa analiz etmeden hata fırlat (döngüde atlanacak)
    if len(y) < 2048:
        raise ValueError("Audio is too short or completely empty")

    # Extract Mean ZCR, RMS
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))
    rms = np.mean(librosa.feature.rms(y=y))

    # Pitch çıkarımı çöküşe en meyilli yerdir, burayı özel olarak korumaya alıyoruz
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        pitch = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0
    except:
        pitch = 0.0  # Eğer hesaplayamazsa 0 kabul et ve devam et

    # Extract Mean MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)

    # Combine all features
    features = np.hstack(([zcr, rms, pitch], mfcc_mean))
    return features


# --- DIRECTORY PROCESSING ---

DATASET_DIR = r"C:\Users\emirc\Desktop\ISTUN\25-26\Bahar\signals-and-systems\midterm-project\midterm-2\Dataset"

search_pattern = os.path.join(DATASET_DIR, "**", "*.wav")
audio_files = glob.glob(search_pattern, recursive=True)

all_data = []

print(f"Found {len(audio_files)} audio files. Starting extraction...")

# tqdm ile süreci görselleştiriyoruz
for file in tqdm(audio_files, desc="Processing Audio Files"):
    try:
        features = extract_features(file)

        file_name = os.path.basename(file)
        group_folder = os.path.basename(os.path.dirname(file))

        row = [group_folder, file_name] + features.tolist()
        all_data.append(row)

    except Exception as e:
        # Hatalı dosyayı ekrana bas ama işlemi durdurma
        pass  # Terminal kirlenmesin diye pass geçiyoruz, istersen print() yazabilirsin

columns = ["Group", "Filename", "ZCR", "RMS", "Pitch"] + [
    f"MFCC_{i}" for i in range(1, 14)
]

df = pd.DataFrame(all_data, columns=columns)
df.to_csv("extracted_features.csv", index=False)
print("\nFeature extraction complete. Saved to extracted_features.csv")
