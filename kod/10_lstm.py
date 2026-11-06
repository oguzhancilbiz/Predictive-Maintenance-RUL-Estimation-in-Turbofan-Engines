"""
10_lstm.py — LSTM ile RUL tahmini, iki farkli kayip fonksiyonuyla.

LSTM NEDIR: Long Short-Term Memory. Sirali veriyi isleyen bir sinir agi turu.
Normal bir ag her girdiyi bagimsiz gorur; LSTM ise onceki adimlardan bir
"hafiza" tasir. Bu yuzden "son 30 dongude yukseliyor" gibi oruntuleri
yakalayabilir.


OLCEKLEME SIZINTISI DUZELTMESI: Olcekleyici SADECE egitim motorlarinin
pencerelerine uydurulur (fit), dogrulama ve test motorlarina sadece uygulanir.

IKI DEGERLENDIRME:
  1) Dogrulama motorlari (egitimde hic gorulmeyen 20 motor, tum pencereler)
  2) STANDART PROTOKOL: 100 test motoru, her birinin son dongusu -> 100 tahmin.
     Literaturle karsilastirilabilecek sayi budur (06_test_degerlendirme.py ile ayni).

ONCE CALISTIR: 09_pencere.py
CIKTI: ciktilar/lstm_sonuclari.csv, ciktilar/lstm_test_tahminleri.csv, modeller/lstm_*.keras

"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # TensorFlow'un gereksiz uyarilarini sustur

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import ortak

# Tohum (seed): rastgelelik sabitlensin diye. Farkli tohumla denemek icin: TOHUM=1 python kod/10_lstm.py
SEED = int(os.environ.get("TOHUM", 42))
EK = "" if SEED == 42 else f"_tohum{SEED}"
np.random.seed(SEED)
tf.random.set_seed(SEED)

X = np.load(ortak.VERI / "pencere_X.npy")
y = np.load(ortak.VERI / "pencere_y.npy")
gruplar = np.load(ortak.VERI / "pencere_grup.npy")
X_test = np.load(ortak.VERI / "test_pencere_X.npy")
y_test = np.load(ortak.VERI / "test_pencere_y.npy")      # ham (kirpilmamis) gercek RUL


def asimetrik_kayip(y_true, y_pred):
    """
    NASA skor fonksiyonunun Keras surumu.

    Normal kayip (MSE) erken ve gec tahmini esit cezalandirir.
    Bu fonksiyon gec tahmini (motor beklenenden once bozulur) daha agir
    cezalandirir, cunku o durum tehlikelidir.

    tf.where(kosul, dogruysa, yanlissa): numpy'daki np.where ile ayni mantik,
    ama TensorFlow tensorlerinde calisir ve turevi alinabilir.

    clip_by_value: exp() cok buyuk farklarda patlar (exp(125/10) ~ 270 bin) ve
    egitimi bozar. Farki -/+60 ile sinirliyoruz. Bu kucuk bir muhendislik
    karari; raporda belirtilmeli.
    """
    d = tf.clip_by_value(tf.reshape(y_pred, [-1]) - tf.reshape(y_true, [-1]), -60.0, 60.0)
    return tf.reduce_mean(tf.where(d < 0,
                                   tf.exp(-d / 13.0) - 1.0,   # erken: hafif ceza
                                   tf.exp(d / 10.0) - 1.0))   # gec: agir ceza


def olcekle(egitim_X, *digerleri):
    """
    Olcekleyiciyi SADECE egitim penceresine uydurur, hepsine uygular.
    Pencereleri (ornek, zaman, ozellik) -> (ornek*zaman, ozellik) yapip olcekler,
    sonra eski sekle dondurur.
    """
    n_oz = egitim_X.shape[2]
    sc = StandardScaler().fit(egitim_X.reshape(-1, n_oz))

    def uygula(a):
        return sc.transform(a.reshape(-1, n_oz)).reshape(a.shape).astype(np.float32)

    return [uygula(egitim_X)] + [uygula(d) for d in digerleri]


def lstm_kur(zaman_adimi, ozellik_sayisi):
    """2 katmanli LSTM agi kurar."""
    return keras.Sequential([
        keras.layers.Input(shape=(zaman_adimi, ozellik_sayisi)),
        # return_sequences=True: her zaman adimi icin cikti ver.
        # Ust uste LSTM koyacaksak alttakinin bunu yapmasi gerekir.
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),          # asiri ogrenmeyi (overfitting) engeller
        keras.layers.LSTM(32),              # son katman sadece son ciktiyi verir
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),              # tek sayi: RUL tahmini
    ])


if __name__ == "__main__":
    # Motor bazli tek bolme: 80 motor egitim, 20 motor dogrulama
    egitim_idx, dogr_idx = next(GroupKFold(n_splits=5).split(X, y, groups=gruplar))
    Xe, Xd, Xt = olcekle(X[egitim_idx], X[dogr_idx], X_test)
    ye, yd = y[egitim_idx], y[dogr_idx]

    print(f"Egitim: {len(egitim_idx)} pencere ({len(np.unique(gruplar[egitim_idx]))} motor) | "
          f"Dogrulama: {len(dogr_idx)} pencere ({len(np.unique(gruplar[dogr_idx]))} motor) | "
          f"Test: {len(Xt)} motor\n")

    sonuclar, test_tahminleri = [], {}
    for kayip_adi, kayip, dosya in [("MSE (standart)", "mse", "mse"),
                                    ("Asimetrik (NASA)", asimetrik_kayip, "asimetrik")]:
        print(f"--- {kayip_adi} ile egitiliyor ---")
        keras.utils.set_random_seed(SEED)
        model = lstm_kur(Xe.shape[1], Xe.shape[2])
        # clipnorm: gradyan cok buyurse kes (asimetrik kayipta egitimi sabitler)
        model.compile(optimizer=keras.optimizers.Adam(0.001, clipnorm=1.0), loss=kayip)

        # EarlyStopping: dogrulama hatasi 6 tur iyilesmezse dur, en iyi haline don.
        durdur = keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
        model.fit(Xe, ye, validation_data=(Xd, yd), epochs=40, batch_size=256,
                  callbacks=[durdur], verbose=2)

        # 1) Dogrulama motorlari (tum pencereler, kirpilmis hedef)
        tahmin_d = model.predict(Xd, verbose=0).flatten()
        # 2) Standart protokol: test motorlarinin son donguleri.
        #    Model en fazla ~125 tahmin edebilir; gercek RUL ham degerle kiyaslanir.
        tahmin_t = model.predict(Xt, verbose=0).flatten()
        y_t_kirpik = np.minimum(y_test, ortak.RUL_UST_SINIR)

        satir = {
            "Model": f"LSTM - {kayip_adi}",
            "Dogrulama_RMSE": round(ortak.rmse(tahmin_d, yd), 2),
            "Dogrulama_NASA": round(ortak.nasa_skor(tahmin_d, yd), 0),
            "Test_RMSE_ham": round(ortak.rmse(tahmin_t, y_test), 2),
            "Test_RMSE_kirpik": round(ortak.rmse(tahmin_t, y_t_kirpik), 2),
            "Test_NASA": round(ortak.nasa_skor(tahmin_t, y_t_kirpik), 0),
            "Test_gec_tahmin_orani": round(float(np.mean(tahmin_t > y_t_kirpik)), 2),
        }
        sonuclar.append(satir)
        test_tahminleri[dosya] = tahmin_t
        for k, v in satir.items():
            print(f"  {k:<24}: {v}")
        print()
        model.save(ortak.MODELLER / f"lstm_{dosya}{EK}.keras")

    pd.DataFrame(sonuclar).to_csv(ortak.CIKTILAR / f"lstm_sonuclari{EK}.csv", index=False)
    pd.DataFrame({"gercek_RUL": y_test, **test_tahminleri}).to_csv(
        ortak.CIKTILAR / "lstm_test_tahminleri.csv", index=False)
    print("Kaydedildi -> ciktilar/lstm_sonuclari.csv ve ciktilar/lstm_test_tahminleri.csv")
