"""
08_hiperparametre.py — XGBoost ayarlarini otomatik arar.

HIPERPARAMETRE NEDIR: Modelin kendi ogrenmedigi, bizim disaridan verdigimiz
ayarlar. Agac sayisi, agac derinligi, ogrenme hizi gibi. Dogru ayar modeli
belirgin olcude iyilestirebilir.

NEDEN RASTGELE ARAMA: Butun kombinasyonlari denemek (GridSearch) 3^6 = 729
deneme eder, saatler surer. RandomizedSearchCV rastgele n_iter tane secer;
pratikte neredeyse ayni sonucu cok daha hizli verir.

SURE: n_iter=12 ile birkac dakika. Artirirsan daha iyi sonuc, daha uzun sure.
ONCE CALISTIR: 03_ozellikler.py
CIKTI: modeller/xgb_ayarli.joblib, ciktilar/ozellik_onemi.csv
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from xgboost import XGBRegressor
import ortak

df = pd.read_csv(ortak.VERI / "FD001_egitim_ozellikli.csv")
sut = (ortak.VERI / "model_sutunlari.txt").read_text().split("\n")
X, y, gruplar = df[sut], df["RUL_kirpik"], df["unit"]

# Aranacak ayar alani. Her anahtar bir hiperparametre, her liste denenecek degerler.
ARAMA_ALANI = {
    "n_estimators":     [200, 400, 600],    # kac agac kurulacak
    "max_depth":        [4, 6, 8],          # her agac ne kadar derin
    "learning_rate":    [0.03, 0.05, 0.1],  # her agacin katkisi ne kadar
    "subsample":        [0.7, 0.85, 1.0],   # her agac satirlarin yuzde kacini gorsun
    "colsample_bytree": [0.7, 0.85, 1.0],   # her agac sutunlarin yuzde kacini gorsun
    "min_child_weight": [1, 5, 10],         # bir yaprakta en az kac ornek olsun
}

if __name__ == "__main__":
    print("Hiperparametre aramasi basliyor...\n")

    arama = RandomizedSearchCV(
        XGBRegressor(n_jobs=-1, random_state=42, verbosity=0),
        ARAMA_ALANI,
        n_iter=12,                  # kac farkli kombinasyon denensin
        cv=GroupKFold(n_splits=3),  # motor bazli bolme - sizinti olmasin
        scoring="neg_root_mean_squared_error",  # sklearn buyuk=iyi ister, o yuzden negatif
        random_state=42,
        n_jobs=1,                   # XGBoost zaten cok cekirdek kullaniyor
    )
    # groups parametresi GroupKFold'a gider: hangi satir hangi motora ait
    arama.fit(X, y, groups=gruplar)

    print("=== SONUC ===")
    print(f"  En iyi RMSE : {-arama.best_score_:.2f}")
    print("  En iyi ayarlar:")
    for anahtar, deger in arama.best_params_.items():
        print(f"    {anahtar:<20} = {deger}")

    # --- Ozellik onemi: model hangi sutunlara bakiyor? ---
    en_iyi = arama.best_estimator_
    onem = pd.Series(en_iyi.feature_importances_, index=sut).sort_values(ascending=False)

    print("\n=== EN ONEMLI 10 OZELLIK ===")
    print(onem.head(10).round(4).to_string())

    # Turettigim ozelliklerin payi ne kadar?
    # Turetilmis ozelliklerin adinda "_ort", "_sap" veya "_fark" geciyor.
    turetilmis = [s for s in onem.head(10).index
                  if any(ek in s for ek in ("_ort", "_sap", "_fark"))]
    print(f"\n  Ilk 10 ozelligin {len(turetilmis)} tanesi benim turettigim ozellik.")

    # Kaydet
    joblib.dump({"model": en_iyi, "sutunlar": sut},
                ortak.MODELLER / "xgb_ayarli.joblib")
    onem.to_csv(ortak.CIKTILAR / "ozellik_onemi.csv", header=["onem"])
    print("\nKaydedildi -> modeller/xgb_ayarli.joblib ve ciktilar/ozellik_onemi.csv")
