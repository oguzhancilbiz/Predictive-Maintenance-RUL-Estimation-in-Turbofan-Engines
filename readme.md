# ✈️ Kestirimci Bakım (Predictive Maintenance) & RAG Karar Destek Sistemi

Bu proje, NASA C-MAPSS turbofan uçak motoru zaman serisi verilerini kullanarak motorların **Kalan Faydalı Ömrünü (RUL)** tahmin eden, anomali tespiti yapan ve bakım kararlarını **finansal maliyet simülasyonları ve Pekiştirmeli Öğrenme (RL)** ile optimize eden uçtan uca bir makine öğrenmesi hattıdır (pipeline). 

Sistem sadece analitik bir tahmin üretmekle kalmaz; aynı zamanda **FAA (Federal Havacılık İdaresi)** geçmiş arıza kayıtlarını kullanarak **RAG (Retrieval-Augmented Generation)** mimarisiyle teknisyene eyleme dönüştürülebilir özetler sunar. Tüm süreç **FastAPI**, **Streamlit** ve **MLflow** ile uçtan uca bir ürüne dönüştürülmüştür.

---

## 🏗️ Sistem Mimarisi (4 Katmanlı Yapı)

1. **Sensör Katmanı:** RUL tahmini (LSTM, XGBoost, LightGBM) ve Anomali Tespiti (Isolation Forest).
2. **Karar Katmanı:** Finansal optimizasyon, Sabit Takvim vs. Eşik Politikası ve Q-Learning.
3. **Bilgi Katmanı (RAG):** TF-IDF ve Embeddings ile FAA SDR metinlerinden kaynaklı özet üretimi.
4. **Ürün Katmanı:** Canlı API (FastAPI), Filo Yönetim Paneli (Streamlit) ve Deney Takibi (MLflow).

---

## 📂 Kod Mimarisi ve Dosya İşlevleri

Projedeki her bir betik (script), modüler bir veri bilimi boru hattının (pipeline) bir parçasını oluşturacak şekilde tasarlanmıştır.

### 1. Veri Hazırlığı ve Öznitelik Mühendisliği (Feature Engineering)
* **`01_veri_hazirla.py`**: Ham C-MAPSS `.txt` dosyalarını okur. Motor ömürlerini hesaplar, gereksiz/sabit sensörleri düşürür ve model hedefi olan RUL değerini hesaplar. RUL değeri, modelin yeni motorlarda yanılgıya düşmemesi için maksimum `125` döngü ile sınırlandırılır (`RUL_kirpik`).
* **`02_eda.py`**: Keşifçi Veri Analizi (EDA). Sensörlerin RUL ile korelasyonlarını, motor bozulma eğrilerini ve **Welch t-testi** ile sağlıklı/bozuk dönemler arasındaki istatistiksel farklılıkları analiz eder.
* **`03_ozellikler.py`**: Zaman serilerindeki gürültüyü azaltmak ve trendleri yakalamak için motor bazında (`groupby("unit")`) kayan pencere ortalaması (`rolling mean`), standart sapma ve gecikme farkları (`lags`) türetir. Model girdi boyutunu 16'dan 80'e çıkarır.

### 2. Modelleme, Sızıntı Önleme (Data Leakage) ve Derin Öğrenme
* **`04_modeller.py`**: Ridge, KNN, RF, XGBoost ve LightGBM algoritmalarını yarıştıran ana modelleme betiğidir. En kritik noktası; veri sızıntısını engellemek için **`GroupKFold`** kullanarak 5-katlı motor bazlı çapraz doğrulama yapmasıdır.
* **`06_test_degerlendirme.py`**: Geliştirilen modelleri, literatürdeki standart araştırma protokolüne uygun şekilde, 100 test motorunun **sadece son döngüsü** üzerinden test eder.
* **`07_sizinti_deneyi.py`**: Veri sızıntısının etkisini deneysel olarak kanıtlar. Rastgele `KFold` bölmenin hatayı (RMSE) 11.50 gösterdiğini, doğru `GroupKFold` bölmenin ise 15.69 olduğunu ispatlayarak sızıntının %27'lik sahte bir iyileşme yarattığını raporlar.
* **`08_hiperparametre.py`**: XGBoost modeli için `RandomizedSearchCV` ve `GroupKFold` entegrasyonu ile parametre optimizasyonu yapar ve Feature Importance (Özellik Önemi) çıkarır.
* **`09_pencere.py` & `10_lstm.py`**: Zaman serisi verilerini Derin Öğrenme için 3 boyutlu `(örnek, zaman_adımı, özellik)` tensörlere çevirir. 2 katmanlı **LSTM** ağını eğitir. Geç tahmin riskini (motorun arızalanması) minimize etmek için standart MSE yerine **Custom Asymmetric NASA Loss** fonksiyonu kullanır.

### 3. Anomali Tespiti ve Finansal Bakım Optimizasyonu
* **`11_anomali.py`**: Denetimsiz Öğrenme (`Isolation Forest`) kullanılarak sadece sağlıklı motor verisiyle "normal" durum modellenir. Sensör sapmaları analiz edilerek arızadan ortalama **34-39 döngü önce** erken uyarı (anomali) üretilir.
* **`05_politika.py`**: Klasik Sabit Takvimli (N) bakım ile Tahmine Dayalı Eşik (T) politikalarını finansal olarak yarıştırır. Arıza maliyetini planlı bakımın 10 katı alarak optimize edilmiş karar eşiklerini bulur.
* **`12_rl_karar.py`**: Karar sürecini bir adım ileri taşıyarak **Q-Learning (Pekiştirmeli Öğrenme)** ajanı kurgular. Durum (RUL & Yaş) ve Eylem (Bekle / Bakıma Al) uzayında eğitilen ajan ile eşik bazlı politikalar dürüst (hiç görülmemiş motorlarda) test edilir.
* **`16_duyarlilik.py` & `17_anomali_duyarlilik.py`**: Arıza maliyeti varsayımlarının (10x yerine 2x, 5x, 50x) ve anomali ardışık aşım kurallarının model kararlılığına olan duyarlılığını test eder.
* **`18_alt_kumeler.py` & `19_rul_siniri.py`**: Mimariyi çok koşullu veri setlerine (FD002-FD004) genellemek için KMeans ile koşul tabanlı normalizasyon uygular ve RUL kırpma üst sınırının (125) optimal olduğunu doğrular.

### 4. RAG Mimari, NLP Katmanı ve Ürünleştirme
* **`13a_sdr_hazirla.py` & `13_metin_indeksi.py`**: FAA SDR (Service Difficulty Report) kamuya açık 5.548 uçak motoru arıza metnini temizler, Kişisel Verileri (PII - N-Number, Email, Telefon) Regex ile maskeler. Verileri `TF-IDF` ve `all-MiniLM-L6-v2` embedding modelleriyle vektör uzayına alır.
* **`14_rag.py` & `14b_etiket_hazirla.py`**: Modelin tahmin ettiği RUL seviyesine göre dinamik sorgular (örn: "compressor imminent failure") üreterek en benzer geçmiş bakım raporlarını getiren bir RAG sistemi kurar.
* **`15_mlflow_kayit.py`**: (MLOps) Tüm deneyleri, RMSE skorlarını ve hiperparametreleri SQLite tabanlı yerel `mlflow.db` üzerine kaydeder.
* **`20_sunum_grafikleri.py` & `21_mimari_sekli.py`**: Çıktı metriklerinden dinamik sunum grafikleri ve mimari diyagramlar üretir.
* **`api.py` & `api_dene.py`**: Eğitilen en iyi modeli `FastAPI` ile canlı bir RESTful servise dönüştürür. Uçuş kayıtlarını JSON formatında alıp RUL ve bakım statüsü döner.
* **`panel.py`**: `Streamlit` kullanılarak filodaki 100 motorun sağlık durumunu, uyarı listesini, finansal kazançları ve RAG tabanlı geçmiş arıza raporlarını gösteren interaktif dashboard.

---

## 🚀 Öne Çıkan Mühendislik Geliştirmeleri

1. **Sıfır Veri Sızıntısı (Zero Data Leakage):** Zaman serilerinde yapılan yaygın K-Fold hatası önlenmiş, tüm veri `GroupKFold` ile motor bazlı izole edilmiştir.
2. **Asimetrik Kayıp (Asymmetric Loss):** Havacılıkta "geç tahminin" ölümcül/maliyetli olması prensibiyle, geç tahminlere üstel (exponential) ceza kesen TensorFlow tabanlı özel bir loss fonksiyonu kurgulanmıştır.
3. **Maliyet-Duyarlı Optimizasyon:** Salt "düşük RMSE" hedeflenmemiş, gerçek dünyadaki planlı bakım (1x) ile arıza (10x) maliyetleri arasında denge kuran algoritmalar iş odaklı olarak kurgulanmıştır.

---

## 📊 Özet Metrikler

| Model / Yöntem | CV RMSE (GroupKFold) | Test RMSE (Protokol) | NASA Skoru |
| :--- | :---: | :---: | :---: |
| **LightGBM (Ürün Modeli)** | **15.63** | 16.02 | - |
| **XGBoost** | **15.63** | 17.34 | 482 |
| **LSTM (Asimetrik Kayıp)** | 16.48 | 16.02 | **393** (En İyi) |
| **Sabit Takvim Maliyeti** | - | - | 1.49 Birim |
| **Tahmine Dayalı Maliyet** | - | - | **1.09 Birim (%26 Tasarruf)** |

---

## 💻 Kurulum ve Kullanım

### Bağımlılıkların Yüklenmesi
```bash
pip install -r requirements.txt
```

### 1. Eğitim ve Pipeline Süreci
Aşağıdaki betikleri sırasıyla çalıştırarak modelleri eğitebilir, özellikleri türetebilir ve anomalileri tespit edebilirsiniz:
```bash
python kod/01_veri_hazirla.py
python kod/03_ozellikler.py
python kod/04_modeller.py
python kod/05_politika.py
python kod/11_anomali.py
```

### 2. Filo Yönetim Paneli (Streamlit)
```bash
streamlit run kod/panel.py
```
*(Tarayıcınızda `http://localhost:8501` adresinde açılacaktır.)*

### 3. API Servisi (FastAPI)
```bash
uvicorn kod.api:app --port 8000
```
*(Servis durumunu ve RUL tahmin isteklerini `http://localhost:8000/docs` arayüzünden test edebilirsiniz.)*

### 4. MLflow Deney Takibi
```bash
python kod/15_mlflow_kayit.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
*(Sonuçlar `http://localhost:5000` adresinden görüntülenebilir.)*