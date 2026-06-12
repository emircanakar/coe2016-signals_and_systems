import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def generate_fixed_plots(csv_path):
    print(f"Reading data from: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)

        # 1. DUYGU ETİKETİNİ ZORLA ÇIKARTMA
        # İsimlendirme formatından ana duyguları yakalayacağız.
        emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']

        # Metin içeren sütunu bul (büyük ihtimalle Filename sütunu)
        object_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if not object_cols:
            print(
                "Hata: CSV'de dosya isimlerini içeren metin tabanlı bir sütun bulunamadı."
            )
            return

        filename_col = object_cols[-1]

        # Dosya isminin içinden duyguyu tespit eden fonksiyon
        def extract_emotion(filename):
            filename_str = str(filename).lower()
            for e in emotions:
                if e.lower() in filename_str:
                    return e
            return 'Unknown'

        # Temiz 'Emotion' sütununu oluştur
        df['Emotion'] = df[filename_col].apply(extract_emotion)

        # 2. EKSENLERİ BELİRLEME (Görsellerindeki sütunlara göre)
        col_x = (
            'ZCR_mean'
            if 'ZCR_mean' in df.columns
            else df.select_dtypes(include=['float64']).columns[0]
        )
        col_y = (
            'ZCR_std'
            if 'ZCR_std' in df.columns
            else df.select_dtypes(include=['float64']).columns[1]
        )
        col_delta = (
            'RMS_mean'
            if 'RMS_mean' in df.columns
            else df.select_dtypes(include=['float64']).columns[2]
        )

        # --- 2. Özellik Uzayı Dağılım Grafiği ---
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, x=col_x, y=col_y, hue='Emotion', palette='tab10', s=80, alpha=0.8
        )
        plt.title('Feature Space Distribution (Class Separability)')
        plt.xlabel(col_x)
        plt.ylabel(col_y)
        plt.grid(True, linestyle='--', alpha=0.6)
        # Lejantı grafiğin dışına taşıyıp düzenle
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('2_scatter_plot_fixed.png')
        plt.close()
        print("-> '2_scatter_plot_fixed.png' temiz bir şekilde oluşturuldu.")

        # --- 3. İstatistiksel Kutu Grafiği ---
        plt.figure(figsize=(10, 6))
        # Yalnızca geçerli duygu sınıflarını filtrele ('Unknown' olanları at)
        df_filtered = df[df['Emotion'].isin(emotions)]
        sns.boxplot(data=df_filtered, x='Emotion', y=col_delta, palette='Set2')
        plt.title(f'{col_delta} Distribution and Variance by Emotion Class')
        plt.xlabel('Emotion Classes')
        plt.ylabel(col_delta)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('3_boxplot_fixed.png')
        plt.close()
        print("-> '3_boxplot_fixed.png' temiz bir şekilde oluşturuldu.")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")


if __name__ == "__main__":
    FEATURES_CSV_PATH = "advanced_features.csv"  # Kendi dosya adını yaz
    generate_fixed_plots(FEATURES_CSV_PATH)
