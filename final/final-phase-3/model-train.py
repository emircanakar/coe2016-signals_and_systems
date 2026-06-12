import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# 1. Genişletilmiş veri setini yükle
df = pd.read_csv("advanced_features.csv")


# 2. Duyguları etiketle (Faz 1 ile aynı mantık)
def get_emotion(filename):
    filename = filename.lower()
    if "happy" in filename:
        return "Happy"
    elif "angry" in filename or "furious" in filename:
        return "Angry"
    elif "sad" in filename:
        return "Sad"
    elif "surprised" in filename:
        return "Surprised"
    elif "neutral" in filename:
        return "Neutral"
    else:
        return "Unknown"


df['Emotion'] = df['Filename'].apply(get_emotion)
df = df[df['Emotion'] != 'Unknown']

# 3. Özellikleri (X) ve Hedefi (y) ayır
# Group ve Filename harici tüm sütunlar özellik (feature)
X = df.drop(columns=['Group', 'Filename', 'Emotion'])
y = df['Emotion']

# 4. Veriyi ölçeklendir (Scaling) - SVM için hayati önem taşır!
# SVM mesafeler üzerinden çalıştığı için büyük sayılar küçükleri ezmesin diye veriyi standartlaştırıyoruz.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Veriyi Eğitim (%80) ve Test (%20) olarak böl
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 6. SVM Modelini kur ve Hiperparametre Optimizasyonu (Grid Search) yap
print("Model eğitiliyor ve en iyi parametreler aranıyor (Bu biraz sürebilir)...")

# Denediğimiz parametre aralıkları
param_grid = {
    'C': [0.1, 1, 10, 100],  # Hata toleransı
    'gamma': ['scale', 'auto', 0.01, 0.1],  # Etki alanı
    'kernel': ['rbf'],  # Radyal Tabanlı Fonksiyon (En sık kullanılanı)
}

# GridSearchCV: Verilen parametreleri deneyip en iyi kombinasyonu bulur
grid_search = GridSearchCV(
    SVC(random_state=42, probability=True),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
)
grid_search.fit(X_train, y_train)

# En iyi modeli al
best_model = grid_search.best_estimator_
print(f"En iyi parametreler: {grid_search.best_params_}")

# 7. Modeli Test Et
predictions = best_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("-" * 40)
print(f"PHASE 2 - ADVANCED MODEL ACCURACY: {accuracy * 100:.2f}%")
print("-" * 40)

print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions))

cm = confusion_matrix(y_test, predictions, labels=best_model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=best_model.classes_,
    yticklabels=best_model.classes_,
)
plt.title('Confusion Matrix - SVM Model')
plt.ylabel('True Emotion')
plt.xlabel('Predicted Emotion')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion Matrix 'confusion_matrix.png' olarak kaydedildi.")

# Eğitilmiş en iyi modeli ve scaler'ı kaydet
joblib.dump(best_model, 'svm_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("-> SVM Modeli ve Scaler canlı demo için başarıyla diske kaydedildi.")
