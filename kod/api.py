"""
api.py — Tahmin servisi (FastAPI). Panelin ve baska sistemlerin modeli cagirabilmesi icin.

NEDEN API: Model bir dosya olarak durursa sadece bu bilgisayarda kullanilir. API ile
"bir motorun gecmis olcumlerini gonder, tahmini kalan omrunu ve bakim durumunu geri al"
diyebiliriz. Boylece baska bir program (ornegin bir bakim yazilimi) modeli kullanabilir.

NASIL CALISTIRILIR (VS Code terminali, ortam etkinken, PROJE klasorundeyken):
    uvicorn kod.api:app --port 8000
Sonra tarayicida http://localhost:8000/docs ac: orada butonla deneyebilirsin.
Durdurmak icin terminalde Ctrl+C.

ONCE CALISTIR: 03_ozellikler.py ve 04_modeller.py (modeller/en_iyi_model.joblib gerekiyor).
Benzer rapor ozelligi icin ayrica 13a/13 (yoksa o kisim bos doner).

DIKKAT (insan dikkati):
  - Model SADECE C-MAPSS FD001 simulasyon verisiyle egitildi. Baska bir motor tipine uygulama.
  - En az ~20 donguluk gecmis gonder; kisa gecmiste kayan pencere ozellikleri eksik veriyle
    hesaplanir ve tahmin guvenilirligi duser (cevapta uyari doner).
  - Kimlik dogrulama YOK. Bu bir gelistirme surumudur; internete acma.
"""

import contextlib
import importlib
import io
import sys
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
import ortak

# 03_ozellikler.py'deki AYNI fonksiyonu kullaniyoruz: egitimdeki ozellik hesabi ile
# canli hesap birebir ayni olmali, yoksa model yanlis girdi gorur ("egitim-servis kaymasi").
_ozellikler = importlib.import_module("03_ozellikler")

SENSORLER = [f"s{i}" for i in range(1, 22)]
AYARLAR = ["op1", "op2", "op3"]
GEREKLI = ["cycle"] + AYARLAR + SENSORLER

_paket = joblib.load(ortak.MODELLER / "en_iyi_model.joblib")
MODEL, SUTUNLAR = _paket["model"], _paket["sutunlar"]

app = FastAPI(title="Kestirimci Bakim API", version="1.0",
              description="C-MAPSS FD001 simulasyon verisiyle egitilmis RUL tahmin servisi. "
                          "Karar destek amaclidir; bakim karari yetkili teknisyene aittir.")


class Olcum(BaseModel):
    """Bir ucus dongusunun olcumu: dongu numarasi, 3 calisma ayari, 21 sensor."""
    cycle: int = Field(ge=1)
    op1: float
    op2: float
    op3: float
    s1: float; s2: float; s3: float; s4: float; s5: float; s6: float; s7: float
    s8: float; s9: float; s10: float; s11: float; s12: float; s13: float; s14: float
    s15: float; s16: float; s17: float; s18: float; s19: float; s20: float; s21: float


class TahminIstegi(BaseModel):
    motor_id: str = Field(description="Motorun adi/numarasi (sadece cevapta geri doner)")
    olcumler: list[Olcum] = Field(min_length=1, description="Motorun gecmis donguleri (en az ~20)")
    karar_esigi: int = Field(default=7, ge=0, description="Tahmini RUL bunun altindaysa BAKIMA AL")
    uyari_esigi: int = Field(default=30, ge=0, description="Tahmini RUL bunun altindaysa PLANLA")
    rapor_sayisi: int = Field(default=0, ge=0, le=5, description="Kac benzer gecmis rapor getirilsin")


def _benzer_raporlar(tahmini_rul, kac):
    """14_rag.py varsa benzer raporlari getirir; yoksa bos liste."""
    if kac == 0:
        return [], None
    try:
        rag = importlib.import_module("14_rag")
        sorgu = rag.sorgu_uret(tahmini_rul, "compressor")
        sonuc = rag.benzer_bul(sorgu, kac, "tfidf")
        return [{"rapor_id": r["id"], "benzerlik": round(float(r["benzerlik"]), 3),
                 "kaynak": r.get("kaynak", ""), "metin": r["temiz"]} for _, r in sonuc.iterrows()], sorgu
    except Exception as hata:
        return [], f"rapor arama hazir degil ({type(hata).__name__})"


@app.get("/saglik")
def saglik():
    """Servis ayakta mi? Kutu izleme sistemleri bunu duzenli sorar."""
    return {"durum": "ok", "model": type(MODEL).__name__, "ozellik_sayisi": len(SUTUNLAR)}


@app.post("/tahmin")
def tahmin(istek: TahminIstegi):
    """Bir motorun gecmis olcumlerinden SON dongu icin tahmini kalan omru ve durumu dondurur."""
    df = pd.DataFrame([o.model_dump() for o in istek.olcumler])
    if not df["cycle"].is_monotonic_increasing or df["cycle"].duplicated().any():
        raise HTTPException(422, "Dongu numaralari artan sirada ve tekrarsiz olmali.")
    if istek.uyari_esigi < istek.karar_esigi:
        raise HTTPException(422, "uyari_esigi, karar_esigi'nden kucuk olamaz.")

    df["unit"] = 1                                   # tek motor: ozellik fonksiyonu 'unit' bekler
    with contextlib.redirect_stdout(io.StringIO()):  # fonksiyonun print'ini sustur
        df, _ = _ozellikler.ozellik_uret(df)
    rul = max(0.0, float(MODEL.predict(df[SUTUNLAR].iloc[[-1]])[0]))   # RUL negatif olamaz

    durum = ("BAKIMA AL" if rul <= istek.karar_esigi
             else "PLANLA" if rul <= istek.uyari_esigi else "IZLE")
    uyarilar = ["Bu bir karar destek ciktisidir; bakim karari yetkili teknisyene aittir.",
                "Model C-MAPSS FD001 simulasyon verisiyle egitildi; baska motor tiplerine uygulanamaz."]
    if len(df) < 20:
        uyarilar.append(f"Gecmis kisa ({len(df)} dongu): kayan pencere ozellikleri eksik hesaplandi, "
                        "tahmin guvenilirligi dusuk.")
    raporlar, sorgu = _benzer_raporlar(rul, istek.rapor_sayisi)
    return {"motor_id": istek.motor_id, "son_dongu": int(df["cycle"].iloc[-1]),
            "tahmini_rul": round(rul, 1), "durum": durum,
            "esikler": {"karar": istek.karar_esigi, "uyari": istek.uyari_esigi},
            "benzer_raporlar": raporlar, "rapor_sorgusu": sorgu, "uyarilar": uyarilar}
