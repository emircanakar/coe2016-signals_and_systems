import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Veriyi yükle
df = pd.read_excel("Feature_Extracted_Data.xlsx")


# 2. Etiketleri tekrar standartlastir (Hata almamak icin)
def clean_labels(val):
    val = str(val).strip().lower()
    if val in ['f', 'female', 'kadın', 'kadin', 'woman']:
        return "Female"
    elif val in ['child', 'c', 'cocuk', 'çocuk']:
        return "Child"
    elif val in ['erkek', 'e', 'm', 'male', 'man']:
        return "Male"
    return "Unknown"


df['Actual'] = df['Gender'].apply(clean_labels)


# 3. Ayni esik değerleriyle tahminleri yap
def predict_gender(f0):
    if f0 == 0:
        return "Unknown"
    if f0 < 170:
        return "Male"
    elif 170 <= f0 < 270:
        return "Female"
    else:
        return "Child"


df['Predicted'] = df['Avg_F0'].apply(predict_gender)

# 4. Sadece gecerli sinifları filtrele (Unknown olanları analize dahil etme)
valid_df = df[(df['Actual'] != "Unknown") & (df['Predicted'] != "Unknown")]

# 5. Matrisi Hesapla ve Ciz
classes = ['Male', 'Female', 'Child']
cm = confusion_matrix(valid_df['Actual'], valid_df['Predicted'], labels=classes)

plt.figure(figsize=(10, 7))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes
)
plt.title('Cinsiyet Siniflandirma Karmasiklik Matrisi')
plt.xlabel('Tahmin Edilen (Predicted)')
plt.ylabel('Gercek Sinif (Actual)')
plt.show()

# 6. Detayli raporu konsola yazdir
print("\n--- SINIFLANDIRMA RAPORU ---")
print(
    classification_report(
        valid_df['Actual'], valid_df['Predicted'], target_names=classes
    )
)
