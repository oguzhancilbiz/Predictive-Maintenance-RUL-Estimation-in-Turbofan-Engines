"""
17_anomali_duyarlilik.py — "Kac ardisik asim?" secimimin (5) sonucu ne kadar etkiledigini olcer.
3, 5, 8, 12 dene. Ayni anomali skorlari kullanilir; sadece kural degisir.
ONCE CALISTIR: 03_ozellikler.py. CIKTI: ciktilar/anomali_duyarlilik.csv
"""
import importlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import ortak

an = importlib.import_module("11_anomali")
df = an.df
df["anomali_skoru"] = np.nan
esikler = {}
for e_idx, t_idx in GroupKFold(n_splits=5).split(df, groups=df["unit"]):
    egitim_m, test_m = df.iloc[e_idx]["unit"].unique(), df.iloc[t_idx]["unit"].unique()
    skor, esik, idx = an.skorla(egitim_m, test_m)
    df.loc[idx, "anomali_skoru"] = skor
    for m in test_m:
        esikler[m] = esik

satirlar = []
for k in (1, 3, 5, 8, 12):
    an.ARDISIK_SAYI = k
    kalan = []
    for motor, d in df.groupby("unit"):
        _, r = an.bozulma_baslangici(d, esikler[motor])
        if r is not None:
            kalan.append(r)
    kalan = pd.Series(kalan)
    satirlar.append({"ardisik_asim": k, "tespit_edilen_motor": len(kalan),
                     "ort_uyari_payi": round(kalan.mean(), 1), "medyan": kalan.median(),
                     "en_az": kalan.min(), "cok_erken_RUL_ustu_125": int((kalan > 125).sum()),
                     "uyari_payi_30_alti": int((kalan <= 30).sum())})
t = pd.DataFrame(satirlar)
t.to_csv(ortak.CIKTILAR / "anomali_duyarlilik.csv", index=False)
print(t.to_string(index=False))
