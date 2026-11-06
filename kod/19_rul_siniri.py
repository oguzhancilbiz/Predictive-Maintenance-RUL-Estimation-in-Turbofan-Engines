"""
19_rul_siniri.py — RUL ust sinirini (kirpma degeri) degistirip sonucun buna ne kadar bagli oldugunu olcer.

SORU: Ben 125 sectim; literatur 125-130 kullaniyor. 100, 125, 130, 150 sonucu degistirir mi?
YONTEM: FD001, 03_ozellikler.py'nin urettigi 80 sutun, LightGBM. Hedef (RUL) her seferinde farkli
degerde kirpilir. Degerlendirme standart protokol: 100 test motorunun SON dongusu.
ADIL KIYAS ICIN: hata iki sekilde olculur
  - ham RMSE      : gercek (kirpilmamis) RUL ile
  - ortak RMSE    : gercek RUL'u HER ZAMAN 125 ile kirpip (tum denemeler ayni olcude kiyaslansin diye)
ONCE CALISTIR: 03_ozellikler.py
CIKTI: ciktilar/rul_siniri.csv
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import ortak

egitim = pd.read_csv(ortak.VERI / "FD001_egitim_ozellikli.csv")
test = pd.read_csv(ortak.VERI / "FD001_test_ozellikli.csv")
sut = [s for s in (ortak.VERI / "model_sutunlari.txt").read_text().split("\n") if s]
omur = egitim.groupby("unit")["cycle"].transform("max")
ham_rul = omur - egitim["cycle"]                    # kirpilmamis kalan omur
son = test.sort_values(["unit", "cycle"]).groupby("unit").tail(1).sort_values("unit")
gercek = np.loadtxt(ortak.VERI / "RUL_FD001.txt")

satirlar = []
for sinir in (100, 125, 130, 150):
    tahminler = []
    for tohum in range(3):                          # 3 tohum ortalamasi: tek sonuca guvenme
        m = LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, random_state=tohum, n_jobs=2, verbose=-1)
        m.fit(egitim[sut], ham_rul.clip(upper=sinir))
        tahminler.append(np.maximum(m.predict(son[sut]), 0))
    t = np.mean(tahminler, axis=0)
    satirlar.append({"rul_siniri": sinir,
                     "RMSE_ham": round(float(np.sqrt(np.mean((t - gercek) ** 2))), 2),
                     "RMSE_ortak_125": round(float(np.sqrt(np.mean((np.minimum(t, 125) - np.minimum(gercek, 125)) ** 2))), 2),
                     "gec_tahmin_orani": round(float(np.mean(t > gercek)), 2)})
t = pd.DataFrame(satirlar)
t.to_csv(ortak.CIKTILAR / "rul_siniri.csv", index=False)
print(t.to_string(index=False))
