import pandas as pd
import glob
import os

# 1. Dosyalari bul
excel_files = glob.glob(os.path.join("Dataset", "**", "Group_*.xlsx"), recursive=True)

if not excel_files:
    print("Hata: 'Dataset' klasoru veya Excel dosyalari bulunamadi!")
else:
    all_dataframes = []

    # Olasi sutun varyasyonlari
    dosya_varyasyon = ['File name', 'File Name', 'File_Name', 'Dosya_Adi']
    cinsiyet_varyasyon = ['Gender', 'gender', 'Cinsiyet', 'cinsiyet']
    yas_varyasyon = ['Age', 'age', 'Yas', 'yas', 'Yaş']

    for f in excel_files:
        if "GROUP_17" in f.upper():
            continue

        df = pd.read_excel(f)
        mevcut_sutunlar = df.columns.tolist()

        # Klasor adini al (Orn: GROUP_01)
        folder_name = os.path.basename(os.path.dirname(f))

        col_file = next((s for s in dosya_varyasyon if s in mevcut_sutunlar), None)
        col_gender = next((s for s in cinsiyet_varyasyon if s in mevcut_sutunlar), None)
        col_age = next((s for s in yas_varyasyon if s in mevcut_sutunlar), None)

        if col_file and col_gender and col_age:

            def build_relative_path(name):
                name = str(name).strip()  # Bosluklari temizle
                # SADELESTIRME: Eger zaten .wav ile bitiyorsa ekleme yapma
                if not name.lower().endswith('.wav'):
                    name += '.wav'
                # Sadece 'Dataset/Klasor/Dosya.wav' formatinda tut
                return os.path.join("Dataset", folder_name, name)

            temp_df = pd.DataFrame()
            temp_df['File_Path'] = df[col_file].apply(build_relative_path)
            temp_df['Gender'] = df[col_gender]
            temp_df['Age'] = df[col_age]

            all_dataframes.append(temp_df)
        else:
            print(f"Uyari: {f} dosyasinda sutunlar eksik.")

    # 2. Birlestir ve Kontrol Et
    master_df = pd.concat(all_dataframes, ignore_index=True)
    master_df['File_Exists'] = master_df['File_Path'].apply(lambda x: os.path.exists(x))

    # 3. Kaydet
    master_df.to_excel("Master_Metadata.xlsx", index=False)

    # --- TESHIS: Neden FALSE? ---
    failed = master_df[master_df['File_Exists'] == False]
    if not failed.empty:
        print(f"\n--- {len(failed)} DOSYA HALA BULUNAMADI ---")
        print("Kodun aradigi ilk 3 hatali yol sunlar:")
        for p in failed['File_Path'].head(3):
            print(f"-> '{p}'")
        print("\nLutfen bu yollari bilgisayarindaki klasor yapisiyla birebir kiyasla.")
    else:
        print("\nBasarili! Tum dosyalar bulundu (TRUE).")
