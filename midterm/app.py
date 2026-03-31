import streamlit as st
import librosa
import numpy as np
from scipy.signal import correlate
import io


# 1. Analiz Fonksiyonu
def analyze_audio(y, sr):
    f_len = int(sr * 0.025)
    h_len = int(sr * 0.010)
    frames = librosa.util.frame(y, frame_length=f_len, hop_length=h_len)

    f0_list = []
    for i in range(frames.shape[1]):
        frame = frames[:, i]
        energy = np.sum(frame**2)
        if energy > 0.01:
            corr = correlate(frame, frame, mode='full')[len(frame) - 1 :]
            low, high = int(sr / 500), int(sr / 50)
            if len(corr) > high:
                peak = np.argmax(corr[low:high]) + low
                f0_list.append(sr / peak)

    avg_f0 = np.mean(f0_list) if f0_list else 0

    # Siniflandirma Kurallari
    if avg_f0 == 0:
        result = "Tespit Edilemedi"
    elif avg_f0 < 170:
        result = "Male (Erkek)"
    elif 170 <= avg_f0 < 270:
        result = "Female (Kadin)"
    else:
        result = "Child (Cocuk)"

    return avg_f0, result


# 2. UI Tasarimi
st.set_page_config(page_title="Ses Siniflandirma Sistemi", layout="centered")
st.title("🎤 Cinsiyet ve Yas Grubu Siniflandirma")
st.write(
    "Bir ses dosyasi yukleyin, temel frekans (F0) analizi ile sinifini belirleyelim."
)

uploaded_file = st.file_uploader("Ses Dosyasi Sec (.wav)", type=["wav"])

if uploaded_file is not None:
    # Sesi yukle
    y, sr = librosa.load(io.BytesIO(uploaded_file.read()), sr=None)
    st.audio(uploaded_file, format='audio/wav')

    if st.button("Analiz Et"):
        with st.spinner('Analiz ediliyor...'):
            avg_f0, prediction = analyze_audio(y, sr)

            # Sonuclari göster
            st.divider()
            st.subheader(f"Tahmin: {prediction}")
            st.write(f"**Ortalama Temel Frekans (F0):** {avg_f0:.2f} Hz")

            # Teknik Metrikler
            col1, col2 = st.columns(2)
            col1.metric("Örnekleme Hizi", f"{sr} Hz")
            col2.metric("Sinyal Uzunluğu", f"{len(y)/sr:.2f} sn")
