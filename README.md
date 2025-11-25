# ☀️ Hava Durumu Regresyon Analizi (Mean Temperature Prediction)

Bu depo, **2. Dünya Savaşı dönemi** hava durumu kayıtlarını kullanarak günlük **Ortalama Sıcaklığı (MeanTemp)** tahmin eden bir Makine Öğrenimi Regresyon projesini barındırmaktadır.

Proje, hem temel **Lineer Regresyon** modelini hem de daha sağlam ve özellik seçimi yapabilen **Lasso Regresyonu (LassoCV)** modelini uygulayarak performanslarını karşılaştırmaktadır. Odak noktası, yüksek tahmin doğruluğu ve modelin yorumlanabilirliğidir.

---

## ✨ Proje Özellikleri ve Veri Seti

### 💾 Veri Seti

* **Dosya Adı:** Summary of Weather.csv
* **İçerik:** Çeşitli meteorolojik istasyonlardan toplanmış, **Maksimum/Minimum Sıcaklık**, **Yağış**, **Rüzgar Hızı** gibi özellikleri içeren tarihsel veriler.

### ⚙️ Veri Ön İşleme (Data Preprocessing)

`regression.py` dosyası, modeli eğitmeden önce aşağıdaki önemli adımları gerçekleştirir:

1.  **Temizleme:** Eksik verinin çok fazla olduğu veya analiz için gereksiz görülen sütunlar (WindGustSpd, DR, SPD, SND, FT, vb.) veri setinden çıkarılır.
2.  **Öznitelik Mühendisliği:** Veri setindeki bazı kategorik ve tarihsel sütunlar (yıl, ay, gün) makine öğrenimine uygun hale getirilir.
3.  **Ölçekleme:** Aşırı değerlerin model performansını etkilemesini engellemek için tüm sayısal özellikler **StandardScaler** kullanılarak ölçeklenir.

## 🏗️ Modelleme Yaklaşımı

Bu projede iki ana regresyon modeli kullanılmıştır:

### 1. Lineer Regresyon
Temel bir karşılaştırma tabanı (baseline) oluşturmak için kullanılır. Tüm özniteliklerin hedef değişken (MeanTemp) üzerindeki etkisini doğrusal olarak modellemeye çalışır.

### 2. LassoCV (Lasso Regresyonu ile Çapraz Doğrulama)
Lasso, aşırı öğrenmeyi (overfitting) önlemek ve en etkili özellikleri otomatik olarak seçmek için $L_1$ düzenlileştirmesi (regularization) kullanır. Model, **LassoCV** ile en uygun düzenlileştirme gücünü ($\alpha$) çapraz doğrulama (Cross-Validation) yoluyla otomatik olarak belirler.

---

## 📊 Öznitelik Önem Düzeyleri (Feature Importance)

Lasso Regresyonu'nun katsayı analizi (coefficient analysis) ile MeanTemp tahmininde en kritik rolü oynayan öznitelikler şunlardır:

* **MaxTemp**
* **MinTemp**
* **DewPoint** (Çiy Noktası)
* **SeaLevelPress** (Deniz Seviyesi Basıncı)

---

## ⚙️ Nasıl Çalıştırılır (How to Run)

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin.

### 1. Kurulum (Installation)

Projenin gerektirdiği tüm Python kütüphanelerini (Pandas, Scikit-learn, Matplotlib vb.) tek seferde kurmak için:

```bash
pip install -r requirements.txt
