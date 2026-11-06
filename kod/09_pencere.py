"""
09_pencere.py — Veriyi derin ogrenme icin 3 boyutlu pencerelere boler.

NEDEN 3 BOYUT: Klasik modeller (satir, sutun) seklinde 2 boyutlu veri ister.
LSTM ve CNN ise (ornek, zaman, sutun) seklinde 3 boyut ister. Yani her ornek
tek bir satir degil, ardisik 30 dongunun tamami olur.

OLCEKLEME BURADA YAPILMAZ: Olcekleyiciyi tum veriye uydurmak sizinti olur.
Olcekleme 10_lstm.py icinde, sadece egitim motorlarina uydurularak yapilir.

IKI CIKTI SETI:
  - egitim pencereleri: her motorun tum kayan pencereleri
  - test pencereleri  : her TEST motorunun SON 30 dongusu (standart protokol:
                        her motor icin tek tahmin, son donguden)

ONCE CALISTIR: 03_ozellikler.py
CIKTI: veri/pencere_X.npy, pencere_y.npy, pencere_grup.npy,
       veri/test_pencere_X.npy, test_pencere_y.npy
"""

import pandas as pd
import numpy as np
import ortak

PENCERE = ortak.PENCERE   # kac donguluk gecmise bakilacak (ortak.py'de 30)


def pencere_uret(df, sutunlar, pencere=PENCERE):
    """
    Her motor icin kayan pencereler uretir.

    Ornek: motorun 200 dongusu varsa ve pencere 30 ise,
    171 tane ornek cikar (1-30, 2-31, ..., 171-200).
    """
    X_listesi, y_listesi, grup_listesi = [], [], []

    for motor, d in df.groupby("unit"):
        d = d.sort_values("cycle")
        degerler = d[sutunlar].to_numpy()     # (dongu_sayisi, sutun_sayisi)
        hedefler = d["RUL_kirpik"].to_numpy()

        # Motor pencereden kisaysa atla
        if len(d) < pencere:
            continue

        # Kayan pencereler
        for i in range(len(d) - pencere + 1):
            X_listesi.append(degerler[i:i + pencere])      # 30 satirlik blok
            y_listesi.append(hedefler[i + pencere - 1])    # blogun SON dongusunun RUL'u
            grup_listesi.append(motor)

    return (np.array(X_listesi, dtype=np.float32),
            np.array(y_listesi, dtype=np.float32),
            np.array(grup_listesi))


def test_penceresi(df, sutunlar, pencere=PENCERE):
    """Her test motorunun SON `pencere` dongusunu ve o andaki gercek RUL'u dondurur."""
    X_listesi, y_listesi = [], []
    for motor, d in df.groupby("unit"):
        d = d.sort_values("cycle")
        X_listesi.append(d[sutunlar].to_numpy()[-pencere:])
        y_listesi.append(d["RUL"].iloc[-1])    # ham (kirpilmamis) gercek RUL
    return (np.array(X_listesi, dtype=np.float32),
            np.array(y_listesi, dtype=np.float32))


if __name__ == "__main__":
    egitim = pd.read_csv(ortak.VERI / "FD001_egitim_ozellikli.csv")
    test = pd.read_csv(ortak.VERI / "FD001_test_ozellikli.csv")
    sut = (ortak.VERI / "model_sutunlari.txt").read_text().split("\n")

    X, y, gruplar = pencere_uret(egitim, sut)
    Xt, yt = test_penceresi(test, sut)

    print(f"Pencere boyutu     : {PENCERE} dongu")
    print(f"Egitim X sekli     : {X.shape}   (ornek, zaman, ozellik)")
    print(f"Egitim y sekli     : {y.shape}")
    print(f"Egitim motor sayisi: {len(np.unique(gruplar))}")
    print(f"Test X sekli       : {Xt.shape}   (her test motoru icin 1 pencere)")

    np.save(ortak.VERI / "pencere_X.npy", X)
    np.save(ortak.VERI / "pencere_y.npy", y)
    np.save(ortak.VERI / "pencere_grup.npy", gruplar)
    np.save(ortak.VERI / "test_pencere_X.npy", Xt)
    np.save(ortak.VERI / "test_pencere_y.npy", yt)
    print("\nKaydedildi -> veri/pencere_*.npy ve veri/test_pencere_*.npy")
