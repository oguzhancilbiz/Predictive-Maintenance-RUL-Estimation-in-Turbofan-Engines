"""
02_eda.py — Kesifsel Veri Analizi (EDA). Veriyi taniyoruz ve grafikleri uretiyoruz.


ONCE CALISTIR: 01_veri_hazirla.py
CIKTI: ciktilar/ klasorune 4 adet PNG grafik + ekrana istatistik ozeti
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Ekran acmadan dosyaya cizim yap (daha hizli, hatasiz)
import matplotlib.pyplot as plt
from scipy import stats        # Istatistiksel test icin 
import ortak

# Grafiklerin ortak gorunumu — her grafikte tekrar yazmamak icin 
plt.rcParams.update({"font.size": 11,
                     "axes.spines.top": False,     # ust cerceve cizgisini kaldir
                     "axes.spines.right": False})
LACIVERT, TURUNCU, GRI = "#1F3A5F", "#D9822B", "#55606E"


def veriyi_yukle():
    """01 numarali betigin urettigi CSV'yi okur."""
    return pd.read_csv(ortak.VERI / "FD001_egitim.csv")


def ozet_istatistik(df, sutunlar):
    """Her sensorun ortalama, sapma, en kucuk ve en buyuk degeri."""
    print("\n=== OZET ISTATISTIKLER ===")
    print(df[sutunlar].describe().T[["mean", "std", "min", "max"]].round(2))


def korelasyon_analizi(df, sutunlar):
    """
    Hangi sensor RUL ile en cok birlikte hareket ediyor?

    Korelasyon -1 ile +1 arasinda bir sayidir:
      +1'e yakin : sensor artarken RUL de artiyor
      -1'e yakin : sensor artarken RUL azaliyor (bozulma gostergesi!)
       0'a yakin : iliski yok
    """
    kor = df[sutunlar + ["RUL"]].corr()["RUL"].drop("RUL")
    # key=abs : isarete bakma, buyuklugune gore sirala (-0.7, +0.7 kadar guclu)
    kor = kor.sort_values(key=abs, ascending=False)
    print("\n=== RUL ILE KORELASYON (en gucluden) ===")
    print(kor.round(3).to_string())

    en_guclu = kor.head(8)[::-1]        # ilk 8'i al, ters cevir (grafikte alttan ustte)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(en_guclu.index, en_guclu.values,
            color=[TURUNCU if v > 0 else LACIVERT for v in en_guclu.values])
    ax.set_xlabel("RUL ile korelasyon")
    ax.set_title("Kalan ömürle en ilişkili 8 sensör")
    plt.tight_layout()                   # kenarlardaki yazilar kesilmesin
    plt.savefig(ortak.CIKTILAR / "01_korelasyon.png", dpi=150)
    plt.close()                          # bellegi bosalt
    return kor


# Grafiklerde gosterilecek sensor aciklamalari (NASA veri dokumanindan)
SENSOR_ADI = {
    "s2":  "s2 — LPC cikis sicakligi",
    "s4":  "s4 — HPC cikis sicakligi",
    "s11": "s11 — HPC cikis statik basinci",
    "s12": "s12 — yakit akisi / basinc orani",
    "s15": "s15 — bypass orani",
}


def bozulma_egrileri(df, sensorler=("s4", "s11", "s12")):
    """
    Birkac motorun sensor izini, arizaya kalan sureye gore cizer.

    X eksenini TERS ceviriyoruz: solda motor yeni, sagda ariza ani.
    Boylece grafik soldan saga "zaman akisi" gibi okunur.
    """
    fig, eksenler = plt.subplots(1, len(sensorler), figsize=(13, 3.6))
    for eksen, sensor in zip(eksenler, sensorler):
        for motor in [1, 2, 3, 4, 5]:             # ilk 5 motoru ciz
            d = df[df["unit"] == motor]
            eksen.plot(d["RUL"], d[sensor], color=LACIVERT, alpha=0.6, lw=1)
        eksen.invert_xaxis()                      # ekseni ters cevir
        eksen.set_title(SENSOR_ADI.get(sensor, sensor), fontsize=11)
        eksen.set_xlabel("Kalan ömür (döngü)")
    plt.tight_layout()
    plt.savefig(ortak.CIKTILAR / "02_bozulma_egrileri.png", dpi=150)
    plt.close()


def omur_dagilimi(df):
    """100 motorun kac dongu yasadigini histogram olarak gosterir."""
    omurler = df.groupby("unit")["omur"].first()   # her motorun omru, bir kez
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(omurler, bins=15, color=LACIVERT, edgecolor="white")
    ax.axvline(omurler.mean(), color=TURUNCU, lw=2)   # ortalamayi dikey cizgi olarak isaretle
    ax.set_xlabel("Motor ömrü (döngü)")
    ax.set_ylabel("Motor sayısı")
    ax.set_title(f"Motor ömürleri (ortalama {omurler.mean():.0f} döngü)")
    plt.tight_layout()
    plt.savefig(ortak.CIKTILAR / "03_omur_dagilimi.png", dpi=150)
    plt.close()


def hipotez_testi(df, sutunlar):
    """
    SPRINT 512 / GUN 4 — Istatistiksel Cikarim dersinin projedeki karsiligi.

    Soru: saglikli donemdeki sensor ortalamasi ile bozulma donemindeki ortalama
    GERCEKTEN farkli mi, yoksa fark rastlantisal mi?

    Yontem: bagimsiz iki orneklem t-testi.
    p-degeri: "aslinda fark yokken bu kadar buyuk bir fark gorme olasiligi".
    p < 0.05 ise farki rastlantiyla aciklayamayiz -> anlamli kabul ederiz.
    """
    saglikli = df[df["RUL"] > 100]      # motor henuz saglam
    bozulan  = df[df["RUL"] < 30]       # ariza yakin

    print("\n=== HIPOTEZ TESTI: saglikli donem vs bozulma donemi ===")
    print(f"{'Sensor':<8}{'Saglikli':>12}{'Bozulan':>12}{'p-degeri':>14}  Sonuc")
    sonuclar = []
    for s in sutunlar:
        # equal_var=False : iki grubun varyansi esit olmayabilir (Welch t-testi)
        t_ist, p = stats.ttest_ind(saglikli[s], bozulan[s], equal_var=False)
        anlamli = "ANLAMLI" if p < 0.05 else "anlamsiz"
        print(f"{s:<8}{saglikli[s].mean():>12.2f}{bozulan[s].mean():>12.2f}"
              f"{p:>14.2e}  {anlamli}")
        sonuclar.append((s, p))
    return sonuclar


if __name__ == "__main__":
    df = veriyi_yukle()
    sutunlar = ortak.kullanilabilir_sutunlar(df)

    ozet_istatistik(df, sutunlar)
    korelasyon_analizi(df, sutunlar)
    bozulma_egrileri(df)
    omur_dagilimi(df)
    hipotez_testi(df, sutunlar)

    print(f"\nGrafikler kaydedildi -> {ortak.CIKTILAR}")
