"""
04_modeller.py — Model karsilastirmasi. Basitten karmasiga.

EN ONEMLI KAVRAM: MOTOR BAZLI CAPRAZ DOGRULAMA.
Ayni motorun komsu satirlari neredeyse aynidir. Veriyi rastgele bolersen
ayni motor hem egitime hem sinava girer; model kopya ceker ve sonuc
oldugundan iyi gorunur. Buna VERI SIZINTISI denir.
Cozum: GroupKFold ile motorlari butun halinde ayir.

ONCE CALISTIR: 03_ozellikler.py
CIKTI: ciktilar/model_karsilastirma.csv, veri/cv_tahminleri.csv, modeller/*.joblib
"""

import pandas as pd
import numpy as np
import joblib                                    # Egitilmis modeli diske kaydetmek icin
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline       # Olcekleme + model tek paket
from sklearn.metrics import f1_score, recall_score, precision_score
import ortak

KAT_SAYISI = 5     # Veriyi kac parcaya bolecegiz (5-kat capraz dogrulama)


def veriyi_yukle():
    df = pd.read_csv(ortak.VERI / "FD001_egitim_ozellikli.csv")
    sutunlar = (ortak.VERI / "model_sutunlari.txt").read_text().split("\n")
    return df, sutunlar


def modelleri_tanimla():
    """
    Denenecek modeller. Basitten karmasiga siralanmis.

    make_pipeline(StandardScaler(), Model()) : once olcekle sonra modele ver.
    Neden pipeline: olcekleyiciyi SADECE egitim katina uydurmamiz (fit) gerekir.
    Pipeline bunu capraz dogrulamada otomatik dogru yapar. Elle yaparsan
    kolayca sizinti uretirsin.
    Agac modelleri (RandomForest) olceklemeye ihtiyac duymaz, o yuzden onlarda yok.
    """
    modeller = {
        "Ridge":        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "KNN":          make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "RandomForest": RandomForestRegressor(n_estimators=150, min_samples_leaf=8,
                                              n_jobs=-1, random_state=42),
    }
    # XGBoost ve LightGBM kurulu degilse atla — proje yine calissin
    try:
        from xgboost import XGBRegressor
        modeller["XGBoost"] = XGBRegressor(n_estimators=300, max_depth=6,
                                           learning_rate=0.05, n_jobs=-1,
                                           random_state=42, verbosity=0)
    except ImportError:
        print("[uyari] xgboost kurulu degil, atlaniyor. Kurmak icin: pip install xgboost")
    try:
        from lightgbm import LGBMRegressor
        modeller["LightGBM"] = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                             n_jobs=-1, random_state=42, verbose=-1)
    except ImportError:
        print("[uyari] lightgbm kurulu degil, atlaniyor. Kurmak icin: pip install lightgbm")
    return modeller


def capraz_dogrula(model, X, y, gruplar):
    """
    Motor bazli 5-kat capraz dogrulama yapar ve her satir icin
    "kat disi" (out-of-fold) tahmin uretir.

    Kat disi tahmin ne demek: her satirin tahmini, O SATIRIN MOTORUNU hic
    gormemis bir model tarafindan yapilir. Yani hepsi durust tahmindir.
    """
    tahminler = np.zeros(len(X))
    gk = GroupKFold(n_splits=KAT_SAYISI)

    # gk.split(...) her turda (egitim_indeksleri, test_indeksleri) ciftini verir
    for egitim_idx, test_idx in gk.split(X, y, groups=gruplar):
        model.fit(X.iloc[egitim_idx], y.iloc[egitim_idx])      # egit
        tahminler[test_idx] = model.predict(X.iloc[test_idx])  # gormedigi kismi tahmin et
    return tahminler


def siniflandirma_degerlendir(df, sutunlar):
    """
    Soru: motor onumuzdeki 30 donguye ariza yapacak mi? (evet/hayir)

    DIKKAT: veri dengesiz (%15 'evet'). Model "hep hayir" dese %85 dogruluk
    alir ama hicbir arizayi yakalamaz. Bu yuzden DOGRULUGA BAKMA.
    Bak: recall (gercek arizalarin yuzde kacini yakaladik)
         precision (alarm verdiklerimizin yuzde kaci gercekti)
         F1 (ikisinin dengesi)
    """
    X, y, g = df[sutunlar], df["ariza_yakin"], df["unit"]
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=1000, class_weight="balanced"))
    # class_weight="balanced" : azinlik sinifina daha fazla agirlik ver

    tahmin = np.zeros(len(X), dtype=int)
    for e_idx, t_idx in GroupKFold(n_splits=KAT_SAYISI).split(X, y, groups=g):
        model.fit(X.iloc[e_idx], y.iloc[e_idx])
        tahmin[t_idx] = model.predict(X.iloc[t_idx])

    print("\n=== SINIFLANDIRMA (ariza 30 dongu icinde mi?) ===")
    print(f"  Recall    : {recall_score(y, tahmin):.3f}  (arizalarin ne kadarini yakaladik)")
    print(f"  Precision : {precision_score(y, tahmin):.3f}  (alarmlarin ne kadari gercekti)")
    print(f"  F1        : {f1_score(y, tahmin):.3f}  (ikisinin dengesi)")


if __name__ == "__main__":
    df, sutunlar = veriyi_yukle()
    X = df[sutunlar]
    y = df["RUL_kirpik"]        # hedef: ust siniri uygulanmis kalan omur
    gruplar = df["unit"]        # capraz dogrulamada motorlari ayirmak icin

    print(f"Veri: {len(X)} satir, {len(sutunlar)} ozellik, "
          f"{gruplar.nunique()} motor\n")

    sonuclar = []
    en_iyi_ad, en_iyi_rmse, en_iyi_model = None, 1e9, None

    for ad, model in modelleri_tanimla().items():
        print(f"{ad} egitiliyor...", end=" ", flush=True)
        tahmin = capraz_dogrula(model, X, y, gruplar)

        r = ortak.rmse(tahmin, y)
        # NASA skoru KIRPILMIS hedefe gore hesaplanir (y), kirpilmamis RUL'a gore DEGIL.
        # Neden: model en fazla 125 tahmin edebiliyor. Motor yeniyken gercek RUL 300 ise
        # fark -175 olur ve exp(175/13) astronomik bir sayi uretir; skor anlamsizlasir.
        # Literaturde skor genellikle motorun SON dongusunde hesaplanir; burada
        # tum dongulerde kirpilmis hedefe gore hesapliyoruz ve raporda boyle belirtiyoruz.
        skor = ortak.nasa_skor(tahmin, y)
        sonuclar.append({"Model": ad, "RMSE": round(r, 2), "NASA_skor": round(skor, 0)})
        print(f"RMSE {r:.2f}")

        if r < en_iyi_rmse:
            en_iyi_ad, en_iyi_rmse, en_iyi_model = ad, r, model
            df["tahmin"] = tahmin      # en iyi modelin tahminlerini sakla

    tablo = pd.DataFrame(sonuclar).sort_values("RMSE")
    print("\n=== MODEL KARSILASTIRMASI ===")
    print(tablo.to_string(index=False))
    tablo.to_csv(ortak.CIKTILAR / "model_karsilastirma.csv", index=False)

    siniflandirma_degerlendir(df, sutunlar)

    # En iyi modeli TUM veriyle yeniden egit ve kaydet (panel bunu kullanacak)
    print(f"\nEn iyi model: {en_iyi_ad} (RMSE {en_iyi_rmse:.2f}) — tum veriyle egitiliyor...")
    en_iyi_model.fit(X, y)
    joblib.dump({"model": en_iyi_model, "sutunlar": sutunlar},
                ortak.MODELLER / "en_iyi_model.joblib")

    # Kat disi tahminleri kaydet — 05_politika.py bunlari kullanacak
    df.to_csv(ortak.VERI / "cv_tahminleri.csv", index=False)
    print(f"Kaydedildi -> modeller/en_iyi_model.joblib ve veri/cv_tahminleri.csv")
