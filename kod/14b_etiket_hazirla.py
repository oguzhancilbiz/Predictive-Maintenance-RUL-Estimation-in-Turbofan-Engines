"""
14b_etiket_hazirla.py — RAG arama kalitesini olcmek icin ELLE etiketleme dosyasi hazirlar.

NEDEN: "Arama sonucu ilgili mi?" sorusunu sadece INSAN cevaplayabilir. Program bunu bilemez.
Ben 15 ornek sorgu icin gelen sonuclara 1 (ilgili) ya da 0 (ilgisiz) yazacagim, sonra
14_rag.py bunu kullanip precision@3 hesaplayacak (ilk 3 sonucun kaci ilgili).

ADIM 1 (hazirla):   python kod/14b_etiket_hazirla.py
   -> veri/etiketle_BUNU_DOLDUR.csv olusur. Excel'de ac.
ADIM 2 (elle):      'ilgili' sutununa her satir icin 1 veya 0 yaz. 'metin' sutununu OKU.
   KURAL (onceden sabitle, sonuca bakarak degistirme):
     1 = metin, sorgudaki ariza turuyle (ornegin kompresor hasari / yag kacagi / titresim)
         ayni turde bir olayi anlatiyor.
     0 = baska bir konu.
   Excel'de kaydederken "CSV (virgulle ayrilmis)" ya da "CSV UTF-8" sec; ayni adla kaydet.
ADIM 3 (birlestir): python kod/14b_etiket_hazirla.py --birlestir
   -> veri/etiketler.csv olusur (bos birakilan satirlar uyari verir ve atilir).
ADIM 4 (olc):       python kod/14_rag.py --degerlendir

ONCE CALISTIR: 13_metin_indeksi.py
DIKKAT: Havuz (pooling): her sorgu icin TF-IDF'in VE (varsa) gomme yonteminin ilk 5 sonucu
birlestirilir. Boylece iki yontem de adil olculur (etiketsiz rapor ilgisiz sayildigi icin).
"""

import sys
import pandas as pd
import ortak

SORGULAR = [
    "compressor blade damage imminent failure",
    "compressor stall surge during climb",
    "high exhaust gas temperature advanced deterioration",
    "engine vibration high advanced deterioration",
    "oil leak engine low oil quantity",
    "oil pressure fluctuation chip detector",
    "metal particles found on magnetic chip detector",
    "turbine blade damage borescope inspection",
    "bleed air leak duct overheat",
    "fuel flow fluctuation fuel control unit",
    "engine shutdown in flight",
    "foreign object damage bird strike engine inspection",
    "ignition exciter igniter failure no start",
    "exhaust gas temperature indication erratic sensor",
    "early performance deterioration gradual trend engine",
]
HAVUZ_K = 5
DOSYA = ortak.VERI / "etiketle_BUNU_DOLDUR.csv"


def hazirla():
    rag = __import__("importlib").import_module("14_rag")
    yontemler = ["tfidf"] + (["gomme"] if rag.VEKTORLER is not None else [])
    satirlar = []
    for q in SORGULAR:
        goruldu = set()
        for y in yontemler:
            for _, r in rag.benzer_bul(q, HAVUZ_K, y).iterrows():
                if r["id"] not in goruldu:
                    goruldu.add(r["id"])
                    satirlar.append({"sorgu": q, "rapor_id": r["id"], "ilgili": "", "metin": r["temiz"][:400]})
    if bool(rag.raporlar["sentetik"].iloc[0]):
        print("UYARI: raporlar SENTETIK. Gercek olcum icin once 13a ve 13 calistirilmali.")
    pd.DataFrame(satirlar).to_csv(DOSYA, index=False, encoding="utf-8-sig")
    print(f"{len(satirlar)} satir yazildi -> {DOSYA}")
    print("Yontemler:", yontemler, "| Simdi Excel'de ac, 'ilgili' sutununa 1/0 yaz.")


def birlestir():
    d = pd.read_csv(DOSYA, sep=None, engine="python", encoding="utf-8-sig")   # ; veya , otomatik
    d["ilgili"] = pd.to_numeric(d["ilgili"], errors="coerce")
    bos = d["ilgili"].isna().sum()
    if bos:
        print(f"UYARI: {bos} satir bos birakilmis; bunlar atildi. (Hepsini doldurman daha dogru.)")
    d = d.dropna(subset=["ilgili"])
    if not set(d["ilgili"]).issubset({0, 1}):
        raise SystemExit("HATA: 'ilgili' sutununda 0 ve 1 disinda deger var.")
    d["ilgili"] = d["ilgili"].astype(int)
    d[["sorgu", "rapor_id", "ilgili"]].to_csv(ortak.VERI / "etiketler.csv", index=False)
    print(f"{len(d)} etiket -> veri/etiketler.csv (ilgili orani {d['ilgili'].mean():.0%}, sorgu {d['sorgu'].nunique()})")


if __name__ == "__main__":
    birlestir() if "--birlestir" in sys.argv else hazirla()
