"""
15_mlflow_kayit.py — Deney sonuclarini MLflow ile kaydeder (MLOps, Sprint 518).

MLFLOW NEDIR: Her model denemesinin (hangi model, hangi ayar, hangi sonuc) kaydini tutan
defter. "Gecen hafta XGBoost'un RMSE'si kactı, hangi ayarlarla?" sorusunu bir hafta sonra
da cevaplayabilmek icin. Elle Excel'e yazmak yerine program tutar.

BU BETIK: Onceki betiklerin urettigi sonuc dosyalarini (ciktilar/*.csv) okuyup her model
icin bir MLflow "run" olarak kaydeder. Modelleri yeniden egitmez; hizlidir.

KURULUM: pip install mlflow
NOT: Betigi her calistirdiginda ayni deneyler TEKRAR kaydedilir (cift kayit olur). Temiz baslamak icin mlflow.db dosyasini ve mlruns klasorunu sil.
ARAYUZU ACMAK: kayittan sonra, PROJE klasorunde terminalde:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
Tarayicida http://localhost:5000 acilir. Durdurmak icin Ctrl+C.

ONCE CALISTIR: 04, 06, 07, 10, 12 numarali betikler (ne kadarini calistirdiysan o kadari kaydedilir).
CIKTI: PROJE/mlflow.db (kayit defteri) ve PROJE/mlruns/ (eklenen dosyalar)
"""

import pandas as pd
import mlflow
import ortak

# Kayit defteri bir SQLite dosyasi (mlflow.db). Eski "klasor" tabanli yontem MLflow'da artik onerilmiyor.
mlflow.set_tracking_uri(f"sqlite:///{(ortak.ANA_KLASOR / 'mlflow.db').as_posix()}")
DENEY = "kestirimci-bakim-FD001"
if mlflow.get_experiment_by_name(DENEY) is None:
    # Eklenen dosyalar (artifact) proje icindeki mlruns/ klasorune gitsin
    mlflow.create_experiment(DENEY, artifact_location=(ortak.ANA_KLASOR / "mlruns").as_uri())
mlflow.set_experiment(DENEY)

ETIKETLER = {"veri": "NASA C-MAPSS FD001 (simulasyon)", "rul_ust_sinir": str(ortak.RUL_UST_SINIR)}
kayit = 0


def oku(ad):
    p = ortak.CIKTILAR / ad
    return pd.read_csv(p) if p.exists() else None


def kaydet(run_adi, parametreler, metrikler, dosya=None):
    """Tek bir deneyi (run) MLflow'a yazar."""
    global kayit
    with mlflow.start_run(run_name=run_adi):
        mlflow.set_tags(ETIKETLER)
        mlflow.log_params(parametreler)
        mlflow.log_metrics({k: float(v) for k, v in metrikler.items()})
        if dosya is not None and dosya.exists():
            mlflow.log_artifact(str(dosya))      # sonuc dosyasini da ekle
    kayit += 1


# 1) 04: motor bazli capraz dogrulama (her model icin bir run)
t = oku("model_karsilastirma.csv")
if t is not None:
    for _, r in t.iterrows():
        kaydet(f"CV-{r['Model']}", {"asama": "04 capraz dogrulama (GroupKFold 5)", "model": r["Model"]},
               {"RMSE": r["RMSE"], "NASA_skor": r["NASA_skor"]}, ortak.CIKTILAR / "model_karsilastirma.csv")

# 2) 06: standart protokol (100 test motoru, son dongu)
t = oku("test_sonuclari.csv")
if t is not None:
    for _, r in t.iterrows():
        kaydet(f"TEST-{r['Model']}", {"asama": "06 standart protokol", "model": r["Model"]},
               {"RMSE_ham": r["RMSE_ham"], "RMSE_kirpik": r["RMSE_kirpik"], "NASA_skor": r["NASA_skor"],
                "MAE": r["MAE"]}, ortak.CIKTILAR / "test_sonuclari.csv")

# 3) 07: sizinti deneyi
t = oku("sizinti_deneyi.csv")
if t is not None:
    for _, r in t.iterrows():
        kaydet(f"SIZINTI-{r['yontem']}", {"asama": "07 sizinti deneyi", "yontem": r["yontem"]},
               {"RMSE": r["RMSE"]}, ortak.CIKTILAR / "sizinti_deneyi.csv")

# 4) 10: LSTM (her tohum ayri run; bulunan tum tohum dosyalari)
for ek, tohum in [("", 42)] + [(f"_tohum{i}", i) for i in range(1, 20)]:
    t = oku(f"lstm_sonuclari{ek}.csv")
    if t is not None:
        for _, r in t.iterrows():
            kaydet(f"{r['Model']} (tohum {tohum})", {"asama": "10 lstm", "model": r["Model"], "tohum": tohum},
                   {"Dogrulama_RMSE": r["Dogrulama_RMSE"], "Test_RMSE_ham": r["Test_RMSE_ham"],
                    "Test_NASA": r["Test_NASA"], "Test_gec_tahmin_orani": r["Test_gec_tahmin_orani"]},
                   ortak.CIKTILAR / f"lstm_sonuclari{ek}.csv")

# 4b) 18: alt kumeler (FD001-FD004)
t = oku("alt_kumeler.csv")
if t is not None:
    for _, r in t.iterrows():
        kaydet(f"ALTKUME-{r['alt_kume']}", {"asama": "18 alt kume genelleme", "alt_kume": r["alt_kume"]},
               {"CV_RMSE": r["CV_RMSE"], "Test_RMSE_ham": r["Test_RMSE_ham"], "NASA_skor": r["NASA_skor"]},
               ortak.CIKTILAR / "alt_kumeler.csv")

# 5) 12: bakim politikalari (ortalama)
t = oku("rl_karsilastirma.csv")
if t is not None:
    for ad, g in t.groupby("politika"):
        kaydet(f"POLITIKA-{ad}", {"asama": "12 politika (20 deneme ortalamasi)", "politika": ad},
               {"maliyet_motor_basi": g["maliyet_motor_basi"].mean(), "ariza_toplam": g["ariza"].sum(),
                "israf_motor_basi": g["israf_motor_basi"].mean()}, ortak.CIKTILAR / "rl_karsilastirma.csv")

print(f"{kayit} deney MLflow'a kaydedildi -> mlflow.db")
print("Arayuz icin: mlflow ui --backend-store-uri sqlite:///mlflow.db")
