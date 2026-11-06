"""
13_metin_indeksi.py — Ariza raporlarini aranabilir hale getirir.

IKI YONTEM KURULUR (Sprint 516-517 NLP bolumu):
  1) TF-IDF  : kelime sayimina dayali klasik yontem. Internet/indirme gerektirmez, her zaman calisir.
  2) GOMME VEKTORU (embedding) : metnin ANLAMINI sayi dizisine cevirir (all-MiniLM-L6-v2, 384 sayi).
     Benzer anlamli iki metin, birbirine yakin vektorler alir. "Compressor erosion" ile
     "blade wear" ayni kelimeleri icermese de yakin cikar. Model ilk seferde internetten
     (~90 MB) indirilir. Indirilemezse bu adim atlanir, TF-IDF ile devam edilir.

Neden ikisi: raporda "gomme vektoru TF-IDF'ten gercekten daha iyi mi?" sorusunu
olcerek cevaplayabilirim (14_rag.py icinde precision@k).

VERI: veri/raporlar.csv ('id' ve 'metin' sutunlari). Bu dosya YOKSA kod testi icin
veri/raporlar_ORNEK_SENTETIK.csv kullanilir. O dosyadaki metinleri KOD TESTI ICIN
yazdim, GERCEK RAPOR DEGILDIR. Raporda ve sunumda asla gercek veri gibi gosterme.

ONCE: veri/raporlar.csv hazir olmali (bkz. PROJE_REHBERI, Adim 3.1).
CIKTI: veri/raporlar_temiz.csv, veri/tfidf_model.joblib, veri/rapor_vektorleri.npy (varsa)
"""

import re
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import ortak

EN_AZ_KELIME = 20   # bundan kisa raporlari atacagiz (bilgi tasimiyorlar)

# Havacilik kisaltmalari: metinde genis halleri de bulunsun ki iki yazim da eslessin.
KISALTMALAR = {
    "ENG": "engine", "HPC": "high pressure compressor", "LPC": "low pressure compressor",
    "HPT": "high pressure turbine", "EGT": "exhaust gas temperature", "FOD": "foreign object damage",
    "APU": "auxiliary power unit", "MX": "maintenance", "INSP": "inspection",
    # SDR metinlerinde cok gecen kisaltmalar
    "IAW": "in accordance with", "AMM": "aircraft maintenance manual", "EEC": "electronic engine control",
    "EMCD": "electronic magnetic chip detector", "ECAM": "electronic centralized aircraft monitor",
    "SVC": "service", "R&R": "removed and replaced",
}


def temizle(metin):
    """Metni arama icin hazirlar: kisaltmalari acar, fazla bosluklari siler, kucuk harfe cevirir."""
    if not isinstance(metin, str):
        return ""
    m = metin
    for kisa, uzun in KISALTMALAR.items():
        # (?i): buyuk/kucuk harf fark etmesin. \b = kelime siniri (ENGINE icindeki ENG'i yakalamasin).
        m = re.sub(rf"(?i)\b{kisa}\b", f"{kisa} {uzun}", m)
    return re.sub(r"\s+", " ", m).strip().lower()


def raporlari_oku():
    """Gercek raporlari okur; yoksa ORNEK dosyaya duser ve uyarir."""
    gercek = ortak.VERI / "raporlar.csv"
    ornek = ortak.VERI / "raporlar_ORNEK_SENTETIK.csv"
    if gercek.exists():
        return pd.read_csv(gercek), False
    print("*" * 78)
    print("UYARI: veri/raporlar.csv yok. SENTETIK ORNEK dosya kullaniliyor.")
    print("       Bu metinler kod testi icin yazildi, GERCEK RAPOR DEGIL.")
    print("*" * 78 + "\n")
    return pd.read_csv(ornek), True


if __name__ == "__main__":
    df, sentetik = raporlari_oku()
    print(f"Ham rapor sayisi : {len(df)}")
    df["temiz"] = df["metin"].apply(temizle)
    df = df[df["temiz"].str.split().str.len() >= EN_AZ_KELIME].reset_index(drop=True)
    df["sentetik"] = sentetik
    print(f"Temizlik sonrasi : {len(df)}")
    print(f"Ortalama uzunluk : {df['temiz'].str.split().str.len().mean():.0f} kelime")

    # ---- 1) TF-IDF ---------------------------------------------------------
    # ngram_range=(1,2): tek kelimeler ve ikili kelime gruplari ("compressor blade").
    # sublinear_tf: bir kelimenin cok tekrari lineer degil, logaritmik artsin.
    # stop_words="english": "the", "and" gibi anlamsiz kelimeleri at.
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english", min_df=1)
    matris = tfidf.fit_transform(df["temiz"])
    joblib.dump({"vektorleyici": tfidf, "matris": matris}, ortak.VERI / "tfidf_model.joblib")
    print(f"TF-IDF matrisi   : {matris.shape}  (rapor sayisi, kelime/ikili sayisi)")

    # ---- 2) Gomme vektoru (opsiyonel) -------------------------------------
    try:
        from sentence_transformers import SentenceTransformer
        print("\nGomme modeli yukleniyor (ilk seferde ~90 MB indirilir)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vektorler = model.encode(df["temiz"].tolist(), show_progress_bar=True, batch_size=64,
                                 normalize_embeddings=True)   # kosinus benzerligi icin
        np.save(ortak.VERI / "rapor_vektorleri.npy", vektorler)
        print(f"Vektor sekli     : {vektorler.shape}  (rapor sayisi, 384)")
    except Exception as hata:    # paket yok ya da model indirilemedi
        print(f"\n[uyari] Gomme vektoru atlandi ({type(hata).__name__}). TF-IDF ile devam edilecek.")
        print("        Cozum: pip install sentence-transformers  ve internet baglantisi.")
        (ortak.VERI / "rapor_vektorleri.npy").unlink(missing_ok=True)

    df.to_csv(ortak.VERI / "raporlar_temiz.csv", index=False)
    print("\nKaydedildi -> veri/raporlar_temiz.csv, veri/tfidf_model.joblib")
