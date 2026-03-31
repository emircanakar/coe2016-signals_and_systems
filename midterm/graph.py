import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.signal import correlate

# 1. Veri setini ve gecerli dosyaları yükle
# Hata almamak için valid_files degiskenini burada tekrar tanımlıyoruz
master_df = pd.read_excel("Master_Metadata.xlsx")
valid_files = master_df[master_df['File_Exists'] == True]

# 2. Analiz icin bir ses dosyası seç
sample_path = valid_files.iloc[0]['File_Path']
y, sr = librosa.load(sample_path, sr=None)

# 3. Pencereleme (Windowing) - 25ms'lik bir kesit alalim (Talimat: 20-30 ms)
# Sinyalin duragan kabul edildigi sesli bir bolge seçiyoruz
start_sample = int(len(y) // 2)
end_sample = start_sample + int(sr * 0.025)
frame = y[start_sample:end_sample]

# 4. FFT Spektrumu (Frekans Düzlemi)
N = len(frame)
yf = scipy.fftpack.fft(frame)
xf = np.linspace(0.0, sr / 2.0, N // 2)
fft_magnitude = 2.0 / N * np.abs(yf[: N // 2])

# 5. Otokorelasyon (Zaman Düzlemi) - Rτ = x[n]x[n-τ]
corr = correlate(frame, frame, mode='full')
corr = corr[len(frame) - 1 :]  # Sadece pozitif kaymaları (τ) al

# 6. Gorsellestirme (Yan Yana)
plt.figure(figsize=(14, 5))

# Sol Grafik: FFT Spektrumu
plt.subplot(1, 2, 1)
plt.plot(xf, fft_magnitude, color='blue')
plt.title("FFT Magnitude Spectrum (Frequency Domain)")
plt.xlabel("Frekans (Hz)")
plt.ylabel("Genlik")
plt.xlim(0, 1000)  # Insan sesi F0 aralığına odaklan (0-1000 Hz)
plt.grid(True)

# Sag Grafik: Otokorelasyon Fonksiyonu
plt.subplot(1, 2, 2)
plt.plot(corr, color='green')
plt.title("Autocorrelation Function Rτ (Time Domain)")
plt.xlabel("Gecikme (Lag - Samples)")
plt.ylabel("Benzerlik")
plt.xlim(0, int(sr / 50))  # Ilk periyotlara odaklanmak için limiti daralt
plt.grid(True)

plt.tight_layout()
plt.show()
