import pandas as pd
import numpy as np

# 1. Ozellik tablosunu yükle
df = pd.read_excel("Feature_Extracted_Data.xlsx")


# 2. ETIKET STANDARTLASTIRMA
def clean_labels(val):
    val = str(val).strip().lower()

    # Haritalama Grupları
    female_list = ['f', 'female', 'kadın', 'kadin', 'woman']
    child_list = ['child', 'c', 'cocuk', 'çocuk']
    male_list = ['erkek', 'e', 'm', 'male', 'man']

    if val in female_list:
        return "Female"
    elif val in child_list:
        return "Child"
    elif val in male_list:
        return "Male"
    else:
        return "Unknown"


df['Gender_Clean'] = df['Gender'].apply(clean_labels)


# 3. KURAL TABANLI SINIFLANDIRICI (Thresholds)
def predict_gender(f0):
    if f0 == 0:
        return "Unknown"
    # Istatistiksel verilere dayanan esik degerleri
    if f0 < 170:
        return "Male"
    elif 170 <= f0 < 270:
        return "Female"
    else:
        return "Child"


df['Prediction'] = df['Avg_F0'].apply(predict_gender)
df['Is_Correct'] = df['Gender_Clean'] == df['Prediction']

# 4. ISTATISTIKSEL TABLO
stats = df.groupby('Gender_Clean')['Avg_F0'].agg(['count', 'mean', 'std']).reset_index()
success = df.groupby('Gender_Clean')['Is_Correct'].mean() * 100
success = success.reset_index().rename(columns={'Is_Correct': 'Success (%)'})

# Unknown olanları rapordan temizleyelim
final_table = pd.merge(stats, success, on='Gender_Clean')
final_table = final_table[final_table['Gender_Clean'] != 'Unknown']

final_table.columns = [
    'Class',
    'Number of Samples',
    'Average F0 (Hz)',
    'Standard Deviation',
    'Success (%)',
]

print("--- RAPORA EKLENECEK FINAL TABLO ---")
print(final_table.to_string(index=False))

# Excel olarak kaydet
final_table.to_excel("Final_Performance_Table.xlsx", index=False)
