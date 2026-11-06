"""
11_anomali.py — Isolation Forest ile bozulma baslangicini tespit eder.

FIKIR: Modele SADECE saglikli donemi gosteriyoruz. Model "normal nedir"i
ogreniyor. Sonra tum dongulere bakiyor ve normalden sapmaya basladigi ani
isaretliyor. Bu bir DENETIMSIZ ogrenme uygulamasidir (Sprint 513): modele
"su an arizali" etiketi vermiyoruz, sadece "normal budur" diyoruz.

ISOLATION FOREST NASIL CALISIR: Rastgele bolmelerle veriyi ayirir. Sira disi
noktalar az bolmeyle ayrilir (kenarda dururlar), normal noktalar cok bolme
ister. Bu "ayrilma kolayligi" anomali skoru olur.

DURUST DEGERLENDIRME: Model her motor icin O MOTORU HIC GORMEMIS olarak calisir.
Motorlar 5 gruba bolunur (GroupKFold); her grup icin model digerlerinin
saglikli doneminde egitilir ve o grubun motorlarini puanlar.

NOT: 'cycle' (dongu numarasi) ozelligini KULLANMAYIZ. Yoksa model sadece
"motor yaslandi" der, sensorlerde bozulma gormeden alarm verir.

ONCE CALISTIR: 03_ozellikler.py
CIKTI: ciktilar/anomali_baslangic.csv, ciktilar/05_anomali.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import ortak

SAGLIKLI_SINIR = 100   # RUL bunun uzerindeyse "saglikli" sayiyoruz (sadece egitim icin)
ARDISIK_SAYI   = 5     # kac dongu ust uste asarsa "bozulma basladi" diyecegiz
ERKEN_ALARM    = 125   # RUL bunun ustundeyken gelen alarm "cok erken / sahte alarm" sayilir

df = pd.read_csv(ortak.VERI / "FD001_egitim_ozellikli.csv")
sut = [s for s in (ortak.VERI / "model_sutunlari.txt").read_text().split("\n") if s != "cycle"]


def skorla(egitim_motorlari, test_motorlari):
    """
    Egitim motorlarinin SAGLIKLI donemiyle modeli kurar, test motorlarini puanlar.
    Doner: (test satirlari icin anomali skoru, esik)
    """
    e = df[df["unit"].isin(egitim_motorlari)]
    saglikli = e[e["RUL"] > SAGLIKLI_SINIR]

    olcek = StandardScaler().fit(saglikli[sut])          # olcekleyici sadece saglikli egitim verisine uydurulur
    model = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
    model.fit(olcek.transform(saglikli[sut]))

    # score_samples: yuksek = normal, dusuk = anormal.
    # Isareti ters ceviriyoruz ki yuksek = anormal olsun, okumasi kolay olsun.
    esik = np.percentile(-model.score_samples(olcek.transform(saglikli[sut])), 99)

    t = df[df["unit"].isin(test_motorlari)]
    return -model.score_samples(olcek.transform(t[sut])), esik, t.index


def bozulma_baslangici(d, esik):
    """Esigi ARDISIK_SAYI kez ust uste asan ilk donguyu bulur."""
    d = d.sort_values("cycle")
    asim = (d["anomali_skoru"] > esik).to_numpy()
    for i in range(len(asim) - ARDISIK_SAYI + 1):
        if asim[i:i + ARDISIK_SAYI].all():     # hepsi True mu
            return d["cycle"].iloc[i], d["RUL"].iloc[i]
    return None, None


if __name__ == "__main__":
    df["anomali_skoru"] = np.nan
    esikler = {}
    motorlar = df["unit"].unique()
    kat_motor = df["unit"]

    # Her motor, onu gormemis bir modelle puanlanir
    for e_idx, t_idx in GroupKFold(n_splits=5).split(df, groups=kat_motor):
        egitim_m, test_m = df.iloc[e_idx]["unit"].unique(), df.iloc[t_idx]["unit"].unique()
        skor, esik, idx = skorla(egitim_m, test_m)
        df.loc[idx, "anomali_skoru"] = skor
        for m in test_m:
            esikler[m] = esik            # her motorun kendi katinin esigi kullanilir

    sonuclar = []
    for motor, d in df.groupby("unit"):
        dongu, kalan = bozulma_baslangici(d, esikler[motor])
        sonuclar.append({"motor": motor, "esik": round(esikler[motor], 4),
                         "baslangic_dongusu": dongu, "o_anda_kalan_omur": kalan})
    s = pd.DataFrame(sonuclar)
    tespit = s.dropna(subset=["o_anda_kalan_omur"])

    # Saglikli donemde (RUL > 100) esigi asan satirlarin orani: yanlis alarm egilimi
    yanlis = np.mean([(r.anomali_skoru > esikler[r.unit])
                      for r in df[df["RUL"] > SAGLIKLI_SINIR][["unit", "anomali_skoru"]].itertuples()])

    kalan = tespit["o_anda_kalan_omur"]
    print(f"Bozulma baslangici tespit edilen motor : {len(tespit)} / {len(s)}")
    print(f"Ortalama erken uyari suresi            : {kalan.mean():.0f} dongu")
    print(f"Ortanca (medyan) erken uyari suresi    : {kalan.median():.0f} dongu")
    print(f"En erken / en gec uyari                : {kalan.max():.0f} / {kalan.min():.0f} dongu once")
    print(f"Cok erken alarm (RUL > {ERKEN_ALARM})           : {(kalan > ERKEN_ALARM).sum()} motor")
    print(f"Arizadan 30 dongu once veya daha gec    : {(kalan <= 30).sum()} motor (uyari neredeyse hic zaman birakmiyor)")
    print(f"Saglikli donemde esik asan satir orani : %{100 * yanlis:.1f}  (yanlis alarm egilimi)")

    s.to_csv(ortak.CIKTILAR / "anomali_baslangic.csv", index=False)

    # Grafik: bir motorun anomali skorunun zamanla yukselisi
    ornek = int(tespit.iloc[0]["motor"])
    d = df[df["unit"] == ornek].sort_values("cycle")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(d["cycle"], d["anomali_skoru"], color="#1F3A5F", lw=1.2)
    ax.axhline(esikler[ornek], color="#B3261E", ls="--", lw=1, label="Anomali eşiği")
    ax.axvline(tespit.iloc[0]["baslangic_dongusu"], color="#D9822B", lw=2,
               label="Tespit edilen bozulma başlangıcı")
    ax.set_xlabel("Döngü"); ax.set_ylabel("Anomali skoru")
    ax.set_title(f"Motor {ornek} — anomali skorunun zamanla değişimi")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(ortak.CIKTILAR / "05_anomali.png", dpi=150)
    print("\nKaydedildi -> ciktilar/anomali_baslangic.csv ve ciktilar/05_anomali.png")
