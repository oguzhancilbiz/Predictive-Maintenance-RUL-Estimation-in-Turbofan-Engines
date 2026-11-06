"""
03_ozellikler.py — Oznitelik muhendisligi (feature engineering).

FIKIR: Modele tek bir anlik olcum vermek yerine, "son N dongude ne oldu"
bilgisini de veriyoruz. Tek olcum gurultuludur; egilim bilgi tasir.

ONCE CALISTIR: 01_veri_hazirla.py
CIKTI: veri/FD001_egitim_ozellikli.csv, veri/FD001_test_ozellikli.csv
"""

import pandas as pd
import ortak

# Kayan pencere ve gecikme ozelliklerini hangi sensorler icin uretecegiz?
# Hepsi icin uretsek sutun sayisi patlar. Korelasyonu en guclu olanlari sectik
# (02_eda.py ciktisina bakarak). Istersen bu listeyi degistirebilirsin.
ONEMLI = ["s4", "s7", "s11", "s12", "s15", "s20", "s21"]

# Kac donguluk pencerelere bakacagiz
PENCERELER = [5, 10, 20]

# Kac dongu oncesine gore fark alacagiz
GECIKMELER = [1, 5, 10]


def ozellik_uret(df):
    """
    Her motor icin kayan pencere ve gecikme ozellikleri turetir.

    EN KRITIK NOKTA: butun hesaplar groupby("unit") ile MOTOR BAZINDA yapilir.
    Yapmazsan bir motorun son dongusu, siradaki motorun ilk dongusune karisir
    ve veri bozulur. Bu hata gozle fark edilmez ama sonuclari bozar.
    """
    df = df.sort_values(["unit", "cycle"]).copy()   # once motora, sonra donguye gore sirala
    gruplar = df.groupby("unit")

    yeni_sutunlar = []

    for s in ONEMLI:
        for p in PENCERELER:
            # KAYAN PENCERE ORTALAMASI: son p dongunun ortalamasi.
            # Gurultuyu yumusatir, egilimi gosterir.
            # min_periods=1 : ilk dongulerde p kadar veri yoksa eldekiyle hesapla
            ad = f"{s}_ort{p}"
            df[ad] = gruplar[s].transform(lambda x: x.rolling(p, min_periods=1).mean())
            yeni_sutunlar.append(ad)

            # KAYAN PENCERE STANDART SAPMASI: son p dongude ne kadar oynadi.
            # Bozulma basladiginda genellikle oynaklik artar.
            ad = f"{s}_sap{p}"
            df[ad] = gruplar[s].transform(lambda x: x.rolling(p, min_periods=1).std())
            yeni_sutunlar.append(ad)

        for g in GECIKMELER:
            # GECIKME FARKI: su anki deger eksi g dongu onceki deger.
            # Modele "bu sensor yukseliyor mu dusuyor mu" bilgisini verir.
            ad = f"{s}_fark{g}"
            df[ad] = gruplar[s].transform(lambda x: x - x.shift(g))
            yeni_sutunlar.append(ad)

    # DONGU NUMARASI da bir ozelliktir: motor ne kadar sure calismis?
    # Zaten "cycle" sutununda var, modele verecegiz.

    # Rolling std ilk satirda NaN (bos) uretir cunku tek degerin sapmasi yoktur.
    # Shift de ilk g satirda NaN uretir. Bunlari 0 ile dolduruyoruz.
    df[yeni_sutunlar] = df[yeni_sutunlar].fillna(0)

    print(f"  {len(yeni_sutunlar)} yeni ozellik uretildi")
    return df, yeni_sutunlar


def model_sutunlarini_sec(df, yeni_sutunlar):
    """Modele verilecek sutunlarin nihai listesi."""
    temel = ortak.kullanilabilir_sutunlar(df)   # sabit olmayan ham sensorler
    return temel + ["cycle"] + yeni_sutunlar


if __name__ == "__main__":
    for ad in ["egitim", "test"]:
        print(f"{ad} verisi isleniyor...")
        df = pd.read_csv(ortak.VERI / f"FD001_{ad}.csv")
        df, yeni = ozellik_uret(df)
        df.to_csv(ortak.VERI / f"FD001_{ad}_ozellikli.csv", index=False)
        print(f"  Kaydedildi: {df.shape[0]} satir, {df.shape[1]} sutun\n")

    # Model sutun listesini bir dosyaya yaz ki 04 numarali betik okusun
    sutunlar = model_sutunlarini_sec(df, yeni)
    (ortak.VERI / "model_sutunlari.txt").write_text("\n".join(sutunlar))
    print(f"Modelde kullanilacak {len(sutunlar)} sutun -> veri/model_sutunlari.txt")
