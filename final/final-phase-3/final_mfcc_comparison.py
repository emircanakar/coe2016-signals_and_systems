import os
import librosa
import librosa.display
import matplotlib.pyplot as plt


def generate_single_comparison(dataset_root):
    angry_file = None
    neutral_file = None

    print(f"Scanning the dataset directory: {dataset_root}")

    # Bütün veri setini tara ve ilk bulduğun eşleşmeleri al
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".wav"):
                if "Angry" in file and not angry_file:
                    angry_file = os.path.join(root, file)
                elif "Neutral" in file and not neutral_file:
                    neutral_file = os.path.join(root, file)

            # İki dosya da bulunduysa aramayı tamamen durdur
            if angry_file and neutral_file:
                break
        if angry_file and neutral_file:
            break

    if not angry_file or not neutral_file:
        print(
            "Error: Could not find both an 'Angry' and a 'Neutral' file in the dataset."
        )
        return

    print(f"Selected Angry File: {angry_file}")
    print(f"Selected Neutral File: {neutral_file}")
    print("Generating the single comparison heatmap...")

    try:
        # Ses dosyalarını yükle
        y_a, sr_a = librosa.load(angry_file, sr=None)
        y_n, sr_n = librosa.load(neutral_file, sr=None)

        # 13 MFCC bileşenini çıkar
        mfcc_a = librosa.feature.mfcc(y=y_a, sr=sr_a, n_mfcc=13)
        mfcc_n = librosa.feature.mfcc(y=y_n, sr=sr_n, n_mfcc=13)

        # Görselleştirme
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        img1 = librosa.display.specshow(mfcc_a, x_axis='time', ax=ax[0])
        ax[0].set_title('MFCC - Angry Speech')
        fig.colorbar(img1, ax=ax[0], format="%+2.0f dB")

        img2 = librosa.display.specshow(mfcc_n, x_axis='time', ax=ax[1])
        ax[1].set_title('MFCC - Neutral Speech')
        fig.colorbar(img2, ax=ax[1], format="%+2.0f dB")

        plt.tight_layout()
        plt.savefig('final_mfcc_comparison.png')
        plt.close()
        print("-> Success. 'final_mfcc_comparison.png' has been saved.")
    except Exception as e:
        print(f"An error occurred while generating the plot: {e}")


if __name__ == "__main__":
    # Veri seti klasörünün yolunu buraya gir
    DATASET_DIRECTORY = r"C:\Users\emirc\Desktop\ISTUN\25-26\Bahar\signals-and-systems\midterm-project\midterm-2\Dataset"
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt


def generate_focused_mfcc_comparison(dataset_root):
    angry_file = None
    neutral_file = None

    print(f"Scanning the dataset directory: {dataset_root}")

    # Bütün veri setini tara ve ilk bulduğun eşleşmeleri al
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file.endswith(".wav"):
                if "Angry" in file and not angry_file:
                    angry_file = os.path.join(root, file)
                elif "Neutral" in file and not neutral_file:
                    neutral_file = os.path.join(root, file)

            # İki dosya da bulunduysa aramayı tamamen durdur
            if angry_file and neutral_file:
                break
        if angry_file and neutral_file:
            break

    if not angry_file or not neutral_file:
        print(
            "Error: Could not find both an 'Angry' and a 'Neutral' file in the dataset."
        )
        return

    print("Generating the focused comparison heatmap (0th coefficient removed)...")

    try:
        # Ses dosyalarını yükle
        y_a, sr_a = librosa.load(angry_file, sr=None)
        y_n, sr_n = librosa.load(neutral_file, sr=None)

        # 13 MFCC bileşenini çıkar
        mfcc_a_full = librosa.feature.mfcc(y=y_a, sr=sr_a, n_mfcc=13)
        mfcc_n_full = librosa.feature.mfcc(y=y_n, sr=sr_n, n_mfcc=13)

        # 0. KATSAYIYI KIRP (Slicing) - Sadece 1'den 12'ye kadar olanları çizdir
        mfcc_a = mfcc_a_full[1:, :]
        mfcc_n = mfcc_n_full[1:, :]

        # Görselleştirme
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        img1 = librosa.display.specshow(mfcc_a, x_axis='time', ax=ax[0])
        ax[0].set_title('Focused MFCC - Angry Speech')
        fig.colorbar(img1, ax=ax[0], format="%+2.0f dB")

        img2 = librosa.display.specshow(mfcc_n, x_axis='time', ax=ax[1])
        ax[1].set_title('Focused MFCC - Neutral Speech')
        fig.colorbar(img2, ax=ax[1], format="%+2.0f dB")

        plt.tight_layout()
        plt.savefig('final_mfcc_comparison_focused.png')
        plt.close()
        print("-> Success. 'final_mfcc_comparison_focused.png' has been saved.")
    except Exception as e:
        print(f"An error occurred while generating the plot: {e}")


if __name__ == "__main__":
    # Veri seti klasörünün yolunu buraya gir
    DATASET_DIRECTORY = r"C:\Users\emirc\Desktop\ISTUN\25-26\Bahar\signals-and-systems\midterm-project\midterm-2\Dataset"

    generate_focused_mfcc_comparison(DATASET_DIRECTORY)
    generate_single_comparison(DATASET_DIRECTORY)
