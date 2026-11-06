"""
06_test_degerlendirme.py — Literaturle KARSILASTIRILABILIR sonuc uretir.

NEDEN AYRI BIR BETIK:
04_modeller.py capraz dogrulama yapar ve 20.631 satirin HEPSI icin tahmin uretir.
Bu, modelin genel kalitesini gosterir ama literaturdeki sayilarla kiyaslanamaz.

C-MAPSS literaturunde STANDART PROTOKOL sudur:
  1. Modeli train_FD001 ile egit
  2. test_FD001'deki her motorun SADECE SON dongusu icin tahmin yap
  3. Bu 100 tahmini RUL_FD001.txt ile karsilastir
Yani 100 tahmin, 20.631 degil. Makalelerdeki RMSE degerleri budur.

Bu betik o protokolu uygular. Raporda iki sayiyi da ver ve farki acikla.

ONCE CALISTIR: 03_ozellikler.py
CIKTI: ciktilar/test_sonuclari.csv
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import ortak


def modelleri_tanimla():
    modeller = {
        "Ridge":        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                              n_jobs=-1, random_state=42),
    }
    try:
        from xgboost import XGBRegressor
        modeller["XGBoost"] = XGBRegressor(n_estimators=400, max_depth=6,
                                           learning_rate=0.05, n_jobs=-1,
                                           random_state=42, verbosity=0)
    except ImportError:
        print("[uyari] xgboost kurulu degil, atlaniyor")
    return modeller


if __name__ == "__main__":
    egitim = pd.read_csv(ortak.VERI / "FD001_egitim_ozellikli.csv")
    test   = pd.read_csv(ortak.VERI / "FD001_test_ozellikli.csv")
    sutunlar = (ortak.VERI / "model_sutunlari.txt").read_text().split("\n")

    # Her test motorunun SADECE son dongusunu al
    # sort_values + groupby + tail(1) = her motorun en buyuk dongu numarali satiri
    son_dongular = test.sort_values("cycle").groupby("unit").tail(1)
    print(f"Egitim: {len(egitim)} satir | Test: {len(son_dongular)} motor (her birinin son dongusu)\n")

    sonuclar = []
    for ad, model in modelleri_tanimla().items():
        model.fit(egitim[sutunlar], egitim["RUL_kirpik"])
        tahmin = model.predict(son_dongular[sutunlar])

        # Iki farkli referansa gore RMSE:
        #  - kirpilmis hedef: modelin ogrendigi hedefle tutarli
        #  - ham RUL: literaturun kullandigi referans
        r_kirpik = ortak.rmse(tahmin, son_dongular["RUL_kirpik"])
        r_ham    = ortak.rmse(tahmin, son_dongular["RUL"])
        skor     = ortak.nasa_skor(tahmin, son_dongular["RUL"])
        mae      = float(np.mean(np.abs(tahmin - son_dongular["RUL"])))

        sonuclar.append({"Model": ad, "RMSE_ham": round(r_ham, 2),
                         "RMSE_kirpik": round(r_kirpik, 2),
                         "NASA_skor": round(skor, 0), "MAE": round(mae, 2)})
        print(f"{ad:<14} RMSE(ham)={r_ham:6.2f}  RMSE(kirpik)={r_kirpik:6.2f}  "
              f"NASA={skor:8.0f}  MAE={mae:6.2f}")

    tablo = pd.DataFrame(sonuclar).sort_values("RMSE_ham")
    tablo.to_csv(ortak.CIKTILAR / "test_sonuclari.csv", index=False)

    print("\n--- LITERATURLE KARSILASTIRMA (FD001, ayni protokol) ---")
    print("  Ridge (literatur)   : 16.63")
    print("  1D CNN (literatur)  : 16.97")
    print("  LSTM  (literatur)   : 14.93")
    print("  Kaynak: arXiv 2604.27234, 'RUL Estimation for Turbofan Engines:")
    print("          A Comparative Study of Classical, CNN, and LSTM Approaches'")
    print("\nNOT: Bu sayilar ayni protokolle uretildigi icin kiyaslanabilir.")
    print("     04_modeller.py'nin urettigi RMSE ise TUM dongulerde capraz")
    print("     dogrulama sonucudur; literaturle DOGRUDAN kiyaslanamaz.")
    print(f"\nKaydedildi -> ciktilar/test_sonuclari.csv")
