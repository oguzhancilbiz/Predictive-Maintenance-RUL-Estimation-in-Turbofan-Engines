"""
18_alt_kumeler.py — FD001 disindaki C-MAPSS alt kumelerine (FD002, FD003, FD004) genelleme testi.

ALT KUMELER (NASA belgesi):
  FD001: 1 calisma kosulu, 1 ariza modu      (projenin ana verisi)
  FD002: 6 calisma kosulu, 1 ariza modu
  FD003: 1 calisma kosulu, 2 ariza modu
  FD004: 6 calisma kosulu, 2 ariza modu      (en zor)

ONEMLI FARK: 6 kosullu kumelerde (FD002, FD004) sensor degerleri motorun hangi kosulda
calistigina gore buyuk oynar; bozulma bu gurultunun icinde kaybolur. Cozum: calisma
ayarlarini (op1-3) KMeans ile 6 kumeye ayirip her sensoru KENDI KOSULUNDA z-skora ceviririz
(ortalama/std sadece EGITIM motorlarindan ogrenilir). Ayni islem 4 kumeye de uygulanir ki
karsilastirma adil olsun (FD001/FD003'te tek kosul oldugu icin etkisi kucuktur).

MODEL: LightGBM (hizli). Ozellikler: normalize sensorler + 5 ve 10 donguluk kayan ortalama + cycle.
DEGERLENDIRME: (1) motor bazli GroupKFold(5) RMSE, (2) standart protokol: her test motorunun
SON dongusu, gercek RUL = RUL_FDxxx.txt (ham, kirpilmamis) ile kiyas, NASA skoru.

ONEMLI UYARI: FD001 sonucu bu betikte, ana hattaki (03/04/06) ozellik setinden FARKLI ve daha
basit bir ozellik setiyle yeniden hesaplanir. Alt kumeler birbiriyle bu betik icinde kiyaslanir;
ana hattaki sayilarla birebir kiyaslama.

CIKTI: ciktilar/alt_kumeler.csv
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMRegressor
import ortak

SENSORLER = [f"s{i}" for i in range(1, 22)]


def nasa_skoru(tahmin, gercek):
    d = np.asarray(tahmin) - np.asarray(gercek)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def hazirla(ad):
    egitim = ortak.ham_veri_oku(f"train_{ad}")
    test = ortak.ham_veri_oku(f"test_{ad}")
    ust = egitim.groupby("unit")["cycle"].transform("max")
    egitim["RUL"] = (ust - egitim["cycle"]).clip(upper=ortak.RUL_UST_SINIR)
    egitim["omur"] = ust

    # calisma kosulu kumeleri (egitimden ogren)
    ayar = ["op1", "op2", "op3"]
    k = 6 if ad in ("FD002", "FD004") else 1
    if k > 1:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(egitim[ayar])
        egitim["kosul"] = km.predict(egitim[ayar]); test["kosul"] = km.predict(test[ayar])
    else:
        egitim["kosul"] = 0; test["kosul"] = 0

    # kosula gore z-skor; egitimde neredeyse sabit sensorleri at
    ist = egitim.groupby("kosul")[SENSORLER].agg(["mean", "std"])
    tut = [s for s in SENSORLER if (egitim.groupby("kosul")[s].std().fillna(0) > 1e-6).all()]
    for d in (egitim, test):
        for s in tut:
            d[s + "_z"] = (d[s] - d["kosul"].map(ist[(s, "mean")])) / d["kosul"].map(ist[(s, "std")])
    z = [s + "_z" for s in tut]

    def ozellik(d):
        d = d.sort_values(["unit", "cycle"]).copy()
        g = d.groupby("unit")
        yeni = {}
        for s in z:
            for p in (5, 10):
                yeni[f"{s}_ort{p}"] = g[s].transform(lambda x: x.rolling(p, min_periods=1).mean())
        d = pd.concat([d, pd.DataFrame(yeni, index=d.index)], axis=1)
        return d, z + list(yeni) + ["cycle"]

    egitim, sut = ozellik(egitim)
    test, _ = ozellik(test)
    return egitim, test, sut, len(tut)


def model():
    return LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                         colsample_bytree=0.8, random_state=0, n_jobs=2, verbose=-1)


satirlar = []
for ad in ("FD001", "FD002", "FD003", "FD004"):
    egitim, test, sut, n_sensor = hazirla(ad)
    # 1) motor bazli GroupKFold
    hatalar = []
    for e, t in GroupKFold(5).split(egitim, groups=egitim["unit"]):
        m = model().fit(egitim.iloc[e][sut], egitim.iloc[e]["RUL"])
        hatalar.append(np.sqrt(np.mean((m.predict(egitim.iloc[t][sut]) - egitim.iloc[t]["RUL"]) ** 2)))
    cv = float(np.mean(hatalar))
    # 2) standart protokol
    m = model().fit(egitim[sut], egitim["RUL"])
    son = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).sort_values("unit")
    gercek = np.loadtxt(ortak.VERI / f"RUL_{ad}.txt")
    tahmin = np.maximum(m.predict(son[sut]), 0)
    rmse = float(np.sqrt(np.mean((tahmin - gercek) ** 2)))
    rmse_k = float(np.sqrt(np.mean((tahmin - np.minimum(gercek, ortak.RUL_UST_SINIR)) ** 2)))
    satirlar.append({"alt_kume": ad, "egitim_motor": egitim["unit"].nunique(), "test_motor": len(son),
                     "kosul_sayisi": 6 if ad in ("FD002", "FD004") else 1, "kullanilan_sensor": n_sensor,
                     "CV_RMSE": round(cv, 2), "Test_RMSE_ham": round(rmse, 2),
                     "Test_RMSE_kirpik": round(rmse_k, 2), "NASA_skor": round(nasa_skoru(tahmin, gercek)),
                     "gec_tahmin_orani": round(float(np.mean(tahmin > gercek)), 2)})
    print(satirlar[-1], flush=True)

t = pd.DataFrame(satirlar)
t.to_csv(ortak.CIKTILAR / "alt_kumeler.csv", index=False)
print(t.to_string(index=False))
