import pandas as pd
import librosa
import numpy as np
from scipy.signal import correlate

# 1. Veri setini yukle
master_df = pd.read_excel("Master_Metadata.xlsx")
valid_files = master_df[master_df['File_Exists'] == True]

feature_results = []

for index, row in valid_files.iterrows():
    try:
        y, sr = librosa.load(row['File_Path'], sr=None)

        # 25ms pencere, 10ms kayma (Windowing)
        f_len = int(sr * 0.025)
        h_len = int(sr * 0.010)

        # Sinyali pencerelere bol
        frames = librosa.util.frame(y, frame_length=f_len, hop_length=h_len)

        f0_list = []
        zcr_list = []
        energy_list = []

        for i in range(frames.shape[1]):
            frame = frames[:, i]

            # Enerji ve ZCR Hesaplama
            energy = np.sum(frame**2)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=frame))

            energy_list.append(energy)
            zcr_list.append(zcr)

            # Voiced Region (Sesli Bölge) Tespiti ve F0 Hesabı
            if energy > 0.01:  # Eşik değeri
                corr = correlate(frame, frame, mode='full')[len(frame) - 1 :]
                low, high = int(sr / 500), int(sr / 50)
                if len(corr) > high:
                    peak = np.argmax(corr[low:high]) + low
                    f0_list.append(sr / peak)

        # Ortalama değerleri hesapla
        feature_results.append(
            {
                'File_Path': row['File_Path'],
                'Gender': row['Gender'],
                'Avg_F0': np.mean(f0_list) if f0_list else 0,
                'Avg_ZCR': np.mean(zcr_list),
                'Avg_Energy': np.mean(energy_list),
            }
        )
        print(f"Islendi: {index}")

    except Exception as e:
        print(f"Hata {row['File_Path']}: {e}")

# 2. Sonucları yeni bir Excel olarak kaydet
features_df = pd.DataFrame(feature_results)
features_df.to_excel("Feature_Extracted_Data.xlsx", index=False)
print(
    "\nAdim 3 Tamamlandi: Ozellikler 'Feature_Extracted_Data.xlsx' dosyasina kaydedildi."
)
