import streamlit as st
import librosa
import numpy as np
import joblib
import warnings
import tempfile
import os

warnings.filterwarnings('ignore')

# 1. Sayfa Yapılandırması
st.set_page_config(page_title="Emo Challenge Live Demo", layout="centered")


# 2. Modelleri Ön Belleğe Al (Her dosya yüklendiğinde baştan okumasın)
@st.cache_resource
def load_system():
    try:
        model = joblib.load('svm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(
            f"Model yüklenemedi. 'svm_model.pkl' ve 'scaler.pkl' dosyalarının varlığını kontrol et. Detay: {e}"
        )
        return None, None


# 3. Faz 2 Birebir Özellik Çıkarımı
def extract_advanced_features(file_path):
    y, sr = librosa.load(file_path, sr=None)

    if len(y) < 2048:
        raise ValueError("Audio is too short")

    features = []

    # ZCR & RMS
    zcr = librosa.feature.zero_crossing_rate(y=y)
    rms = librosa.feature.rms(y=y)
    features.extend([np.mean(zcr), np.std(zcr), np.mean(rms), np.std(rms)])

    # Pitch (F0)
    try:
        f0, _, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')
        )
        pitch_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0.0
        pitch_std = np.nanstd(f0) if np.any(~np.isnan(f0)) else 0.0
    except:
        pitch_mean, pitch_std = 0.0, 0.0
    features.extend([pitch_mean, pitch_std])

    # Spectral Centroid & Rolloff
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend(
        [np.mean(centroid), np.std(centroid), np.mean(rolloff), np.std(rolloff)]
    )

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend([np.mean(chroma_stft), np.std(chroma_stft)])

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta_mfccs = librosa.feature.delta(mfccs)
    features.extend(np.mean(mfccs, axis=1).tolist())
    features.extend(np.std(mfccs, axis=1).tolist())
    features.extend(np.mean(delta_mfccs, axis=1).tolist())

    return np.array(features).reshape(1, -1)


# 4. Arayüz Tasarımı
st.title("🎙️ Sinyal Analizi ve Duygu Sınıflandırma")
st.markdown("---")

model, scaler = load_system()

if model and scaler:
    # Dosya Yükleyici
    uploaded_file = st.file_uploader(
        "Analiz edilecek .wav dosyasını seçin veya sürükleyin", type=['wav']
    )

    if uploaded_file is not None:
        # Sesi Dinleme Aracı
        st.audio(uploaded_file, format='audio/wav')

        if st.button("Sinyali Analiz Et", type="primary"):
            with st.spinner(
                "Öznitelikler çıkarılıyor ve SVM hiperdüzlemi hesaplanıyor..."
            ):
                # Streamlit bellek içi dosyayı Librosa'ya okutmak için geçici diske yazar
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    features = extract_advanced_features(tmp_path)
                    features_scaled = scaler.transform(features)

                    prediction = model.predict(features_scaled)[0]

                    # Sonucu Vurgula
                    st.success(f"### Kesinleşen Tahmin: **{prediction.upper()}**")

                    st.markdown("#### Sınıf Olasılık Dağılımı")
                    try:
                        probabilities = model.predict_proba(features_scaled)[0]
                        classes = model.classes_

                        # Olasılıkları Bar Grafiği Olarak Bas
                        for cls, prob in zip(classes, probabilities):
                            st.progress(float(prob), text=f"{cls}: %{prob*100:.1f}")
                    except Exception:
                        st.warning(
                            "Olasılık dağılımı gösterilemiyor (GridSearchCV'de probability=True eksik)."
                        )

                except Exception as e:
                    st.error(f"Hata oluştu: {e}")
                finally:
                    # Çöp dosyayı sil
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
