"""
07_sizinti_deneyi.py — Veri sizintisinin sonuca etkisini olcer.

AMAC: Veriyi rastgele bolmek ile motor bazinda bolmek arasindaki farki
gostermek. Rastgele bolmede ayni motorun komsu satirlari hem egitime hem
sinava girer; model kopya ceker ve sonuc oldugundan iyi gorunur.

ONCE CALISTIR: 03_ozellikler.py
CIKTI: ciktilar/sizinti_deneyi.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from xgboost import XGBRegressor
import ortak

# Veriyi ve model sutunlarini yukle
df = pd.read_csv(ortak.VERI / "FD001_egitim_ozellikli.csv")
sut = (ortak.VERI / "model_sutunlari.txt").read_text().split("\n")
X, y, gruplar = df[sut], df["RUL_kirpik"], df["unit"]


def capraz_dogrula(bolucu, gruplar=None):
    """Verilen bolucu ile capraz dogrulama yapip RMSE dondurur."""
    tahminler = np.zeros(len(X))

    # GroupKFold groups parametresi ister, KFold istemez.
    # Bu yuzden iki durumu ayri ele aliyoruz.
    if gruplar is not None:
        bolumler = bolucu.split(X, y, groups=gruplar)
    else:
        bolumler = bolucu.split(X)

    for egitim_idx, test_idx in bolumler:
        model = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                             n_jobs=-1, random_state=42, verbosity=0)
        model.fit(X.iloc[egitim_idx], y.iloc[egitim_idx])
        tahminler[test_idx] = model.predict(X.iloc[test_idx])

    return ortak.rmse(tahminler, y)


if __name__ == "__main__":
    print("Iki bolme yontemi karsilastiriliyor (her biri ~30 saniye)...\n")

    # YANLIS YONTEM: satirlari rastgele karistir ve bol.
    # shuffle=True satirlari karistirir -> ayni motorun satirlari dagilir.
    yanlis = capraz_dogrula(KFold(n_splits=5, shuffle=True, random_state=42))

    # DOGRU YONTEM: motorlari butun halinde ayir.
    dogru = capraz_dogrula(GroupKFold(n_splits=5), gruplar)

    fark = 100 * (dogru - yanlis) / dogru

    print("=== VERI SIZINTISI DENEYI ===")
    print(f"  Rastgele bolme    (YANLIS): RMSE {yanlis:.2f}")
    print(f"  Motor bazli bolme (DOGRU) : RMSE {dogru:.2f}")
    print(f"\n  Yanlis yontem modeli %{fark:.0f} daha iyi gosteriyor.")
    print("  Yani sizintiyi fark etmeseydim, raporuma gercekte olmayan")
    print("  bir basariyi yazmis olacaktim.")

    # Sonucu dosyaya kaydet ki rapora koyabileyim
    pd.DataFrame([{"yontem": "Rastgele bolme (yanlis)", "RMSE": round(yanlis, 2)},
                  {"yontem": "Motor bazli bolme (dogru)", "RMSE": round(dogru, 2)}]
                 ).to_csv(ortak.CIKTILAR / "sizinti_deneyi.csv", index=False)
    print("\nKaydedildi -> ciktilar/sizinti_deneyi.csv")
