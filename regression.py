import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1. VERİ YÜKLEME VE ÖN İŞLEME 
df = pd.read_csv("Summary of Weather.csv", low_memory=False)

# Sütun isimlerinin başındaki/sonundaki gereksiz boşlukları temizliyoruz.
df.columns = df.columns.str.strip()

# Tamamı boş (NaN) olan sütunları siliyoruz.
df.dropna(axis=1, inplace=True, how="all") 

# Analiz için gereksiz görülen veya çok fazla eksik veriye sahip sütunların listesi.
kolonlar_to_drop = [
    "WindGustSpd", "DR", "SPD", "SND", "FT", "FB", "FTI", "ITH", 
    "PGT", "SD3", "RHX", "RHN", "RVG", "WTE", "PoorWeather", "TSHDSBRSGF"
]
# Belirlenen bu sütunları veri setinden atıyoruz.
df.drop(kolonlar_to_drop, axis=1, inplace=True, errors="ignore") 

# Date sütununu "datetime" formatına çeviriyoruz .
df['Date'] = pd.to_datetime(df['Date'])

# 2. EKSİK VERİLERİN DOLDURULMASI

# 'Snowfall' sütununu sayısal değere çeviriyoruz.
df['Snowfall'] = pd.to_numeric(df['Snowfall'], errors='coerce')
# Eksik verileri (NaN), sütunun ortalaması ile dolduruyoruz.
ortalama_deger = df['Snowfall'].mean()
df['Snowfall'] = df['Snowfall'].fillna(ortalama_deger)

# 'Precip' sütununu sayısal tipe çevirip eksikleri ortalama ile dolduruyoruz.
df['Precip'] = pd.to_numeric(df['Precip'], errors='coerce')
precip_mean = df['Precip'].mean()
df['Precip'] = df['Precip'].fillna(precip_mean)

# Yukarıdaki işlemlerden sonra hala eksik veri içeren satır varsa onları siliyoruz.
df.dropna(inplace=True)

# 'PRCP' sütunu için temizlik ve dönüşüm .
df['PRCP'] = pd.to_numeric(df['PRCP'], errors='coerce')
precip_mean = df['PRCP'].mean()
df['PRCP'] = df['PRCP'].fillna(precip_mean)
df['PRCP'] = df['PRCP'].astype(float)

# Modelde "MeanTemp" tahmin edileceği için;
# MAX , MIN gibi doğrudan cevabı içeren sütunları ve
# işlenmiş tarih sütunlarını (YR, MO, DA) siliyoruz.
df.drop(['MAX','MIN','MEA','YR','MO','DA','PRCP','SNF'], axis=1, inplace=True)

# 3. ÖZELLİK MÜHENDİSLİĞİ

# Tarih sütunundan "Ay" ve "Yıl" bilgisini çıkarıp yeni öznitelik olarak ekliyoruz.
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# 4. EĞİTİM VE TEST AYRIMI

# Hedef değişken (y): MeanTemp
# Bağımsız değişkenler (X): MeanTemp ve Date hariç hepsi.
X = df.drop(["MeanTemp", "Date"], axis=1)
y = df["MeanTemp"]

# Veriyi %75 Eğitim, %25 Test olacak şekilde bölüyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)

# 5. ÖLÇEKLENDİRME (SCALING)

# Verileri standartlaştırıyoruz
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Eğitim setine göre fit et
X_test_scaled = scaler.transform(X_test)       # Test setini aynı parametrelerle dönüştür

# 6. MODEL 1: LINEAR REGRESSION

linear = LinearRegression() 
linear.fit(X_train_scaled, y_train) # Modeli eğit
y_pred_linear = linear.predict(X_test_scaled) # Test seti ile tahmin yap

# Başarı Metriklerinin Hesaplanması
mae_linear = mean_absolute_error(y_test, y_pred_linear)
mse_linear = mean_squared_error(y_test, y_pred_linear)
score_linear = r2_score(y_test, y_pred_linear)

print("--- LINEAR REGRESSION METRICS ---")
print(f"MAE: {mae_linear:.4f}")
print(f"MSE: {mse_linear:.4f}")
print(f"R2 Score: {score_linear:.4f}")

# Linear Regression Sonuçlarını Görselleştirme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, alpha=0.5)
# Kırmızı kesikli çizgi ideal tahmin doğrusunu (y=x) gösterir.
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Linear Regression: Actual vs. Predicted')
plt.xlabel('Actual MeanTemp')
plt.ylabel('Predicted MeanTemp')

# 7. MODEL 2: LASSO REGRESSION (LassoCV)

# LassoCV: En iyi alpha (ceza) parametresini bulmak için Cross-Validation yapar.
lasso_cv = LassoCV(cv=5, random_state=15, max_iter=10000) 
lasso_cv.fit(X_train_scaled, y_train)
y_pred_lasso_cv = lasso_cv.predict(X_test_scaled)

# Lasso Metriklerinin Hesaplanması
mae_lasso_cv = mean_absolute_error(y_test, y_pred_lasso_cv)
mse_lasso_cv = mean_squared_error(y_test, y_pred_lasso_cv)
score_lasso_cv = r2_score(y_test, y_pred_lasso_cv)

print("\n--- OPTIMIZED LASSO (LassoCV) METRICS ---")
print(f"Optimal Alpha (Regularization Strength): {lasso_cv.alpha_:.4f}")
print(f"MAE: {mae_lasso_cv:.4f}")
print(f"MSE: {mse_lasso_cv:.4f}")
print(f"R2 Score: {score_lasso_cv:.4f}")

# Lasso Sonuçlarını Görselleştirme
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso_cv, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title(f'Optimized Lasso (Alpha={lasso_cv.alpha_:.2f}): Actual vs. Predicted')
plt.xlabel('Actual MeanTemp')
plt.ylabel('Predicted MeanTemp')
plt.tight_layout()
plt.show()

# 8. ÖZNİTELİK ÖNEM DÜZEYLERİ

# Lasso katsayılarını (coefficients) alıyoruz.
lasso_coefs = pd.Series(lasso_cv.coef_, index=X_train.columns)

# Sadece katsayısı 0'dan anlamlı derecede farklı olanları seçip sıralıyoruz.
non_zero_coefs = lasso_coefs[lasso_coefs.abs() > 1e-4].sort_values(ascending=False)

# Hangi özelliğin sıcaklık üzerinde ne kadar etkisi olduğunu çubuk grafiği ile gösteriyoruz.
plt.figure(figsize=(10, 6))
sns.barplot(x=non_zero_coefs.values, y=non_zero_coefs.index, hue=non_zero_coefs.index, palette='viridis', legend=False)
plt.title('LassoCV Coefficient Weights (Feature Importance)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()