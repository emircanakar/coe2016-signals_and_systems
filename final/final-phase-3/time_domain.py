import librosa
import librosa.display
import matplotlib.pyplot as plt


def plot_time_domain_waveform(audio_path):
    print(f"Ses dosyası okunuyor: {audio_path}")
    try:
        # Ses dosyasını yükle
        y, sr = librosa.load(audio_path, sr=None)

        # Dalga formunu çizdir
        plt.figure(figsize=(10, 5))
        librosa.display.waveshow(y, sr=sr, alpha=0.8, color="#005088")

        plt.title('Time Domain Waveform (Amplitude vs. Time)', fontsize=14, pad=15)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel('Amplitude', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        output_filename = 'time_domain_waveform.png'
        plt.savefig(output_filename)
        plt.close()
        print(
            f"-> Başarılı. '{output_filename}' kaydedildi. Doğrudan sunuma ekleyebilirsin."
        )
    except Exception as e:
        print(f"Hata oluştu: {e}")


if __name__ == "__main__":
    # Veri setindeki HERHANGİ BİR ses dosyasının yolunu buraya yaz
    SAMPLE_AUDIO_PATH = r"C:\Users\emirc\Desktop\ISTUN\25-26\Bahar\signals-and-systems\midterm-project\midterm-2\Dataset\GROUP_01\G01_D01_C_11_Angry_C3.wav"

    plot_time_domain_waveform(SAMPLE_AUDIO_PATH)
