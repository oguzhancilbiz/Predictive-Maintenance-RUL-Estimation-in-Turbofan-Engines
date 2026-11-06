"""
14_rag.py — Bilgi katmani: benzer gecmis raporlari bul, kaynakli bakim ozeti uret.

RAG = Retrieval-Augmented Generation (Getirmeyle Desteklenmis Uretim).
Mantik: once ilgili belgeleri bul (retrieval), SONRA onlara dayanarak metin uret.
Boylece her cumlenin bir kaynagi olur, sistem uydurma yapmaz.

BU PROJEDE: "uretim" kismi SABLON tabanlidir (dil modeli YOK). Neden: uydurma riski sifir,
her satirin kaynagi (rapor no, benzerlik) gorunur. Dil modeli baglamak istersen her
cumlenin kaynagini gostermek ZORUNLU olmali. Rapor, bunu "sinirlilik" olarak yazar.

GERCEKCI SINIR: Raporlar gercek ucaklardan, sensor verisi NASA simulasyonundan gelir.
Sistem "bu rapor bu motoru anlatiyor" DEMEZ; "benzer ariza turunden gecmis deneyim" der.

DEGERLENDIRME: python kod/14_rag.py --degerlendir
  veri/etiketler.csv dosyasini (sorgu, rapor_id, ilgili) okuyup her yontem icin
  precision@3 hesaplar. Etiketleri SEN elle verirsin (bkz. rehber).

ONCE CALISTIR: 13_metin_indeksi.py
"""

import sys
import joblib
import numpy as np
import pandas as pd
import ortak

raporlar = pd.read_csv(ortak.VERI / "raporlar_temiz.csv")
_tf = joblib.load(ortak.VERI / "tfidf_model.joblib")
VEKTORLEYICI, MATRIS = _tf["vektorleyici"], _tf["matris"]

# Gomme vektorleri varsa yukle (yoksa sadece TF-IDF calisir)
_vp = ortak.VERI / "rapor_vektorleri.npy"
VEKTORLER = np.load(_vp) if _vp.exists() else None
_model = None


def _gomme_modeli():
    """Gomme modelini ilk kullanimda yukler (her cagrida yeniden yuklememek icin)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def benzer_bul(sorgu, kac_tane=3, yontem="tfidf"):
    """
    Sorguya en benzer raporlari dondurur.
    yontem: "tfidf" ya da "gomme". Her ikisinde de skor = kosinus benzerligi
    (1'e yakin cok benzer, 0'a yakin ilgisiz).
    """
    sorgu = sorgu.lower()
    if yontem == "gomme":
        if VEKTORLER is None:
            raise RuntimeError("Gomme vektorleri yok. 13_metin_indeksi.py'yi internetli ortamda calistir.")
        v = _gomme_modeli().encode([sorgu], normalize_embeddings=True)[0]
        benzerlik = VEKTORLER @ v                          # @ = matris carpimi
    else:
        # TF-IDF vektorleri varsayilan olarak L2-normalize; nokta carpimi = kosinus
        benzerlik = (MATRIS @ VEKTORLEYICI.transform([sorgu]).T).toarray().ravel()
    en_iyi = np.argsort(benzerlik)[::-1][:kac_tane]        # buyukten kucuge ilk kac_tane
    sonuc = raporlar.iloc[en_iyi].copy()
    sonuc["benzerlik"] = benzerlik[en_iyi]
    return sonuc


def sorgu_uret(tahmini_rul, bozulan_sistem="compressor"):
    """
    Sensor katmaninin ciktisini arama cumlesine cevirir.
    VARSAYIM: RUL'a gore bozulma evresini metne ceviriyorum (benim kuralim, esikler
    ayarlanabilir). Gercek sistemde sensor deseni de sorguya girebilir.
    """
    if tahmini_rul <= 15:
        evre = "imminent failure, severe damage, engine removed"
    elif tahmini_rul <= 40:
        evre = "advanced deterioration, rising exhaust gas temperature, vibration"
    else:
        evre = "early performance deterioration, gradual trend"
    return f"{bozulan_sistem} {evre}"


def bakim_ozeti(motor_no, tahmini_rul, bozulan_sistem="compressor", yontem="tfidf"):
    """SABLON tabanli, kaynakli ozet. Her benzer rapor numarasi ve skoruyla gosterilir."""
    sorgu = sorgu_uret(tahmini_rul, bozulan_sistem)
    benzerler = benzer_bul(sorgu, kac_tane=3, yontem=yontem)
    satirlar = [
        "OTOMATIK URETILMIS BAKIM OZETI",
        "Bu ozet otomatik uretilmistir; bakim karari yetkili teknisyen tarafindan verilir.",
        "",
        f"Motor        : {motor_no}",
        f"Tahmini RUL  : {tahmini_rul:.0f} dongu",
        f"Ilgili sistem: {bozulan_sistem}",
        f"Arama sorgusu: {sorgu}  (yontem: {yontem})",
        "",
        "Benzer gecmis raporlar:",
    ]
    for i, (_, r) in enumerate(benzerler.iterrows(), 1):
        ozet = r["temiz"][:170] + ("..." if len(r["temiz"]) > 170 else "")
        satirlar.append(f"  {i}. [{r['id']} | benzerlik {r['benzerlik']:.2f}] {ozet}")
    satirlar += ["",
                 "NOT: Bu raporlar gercek ucaklardan gelmektedir ve bu motoru anlatmamaktadir.",
                 "     Benzer ariza turunden gecmis deneyimdir."]
    if bool(raporlar["sentetik"].iloc[0]):
        satirlar += ["", "!!! DIKKAT: Raporlar SENTETIK ORNEKTIR (kod testi). Gercek veri degildir. !!!"]
    return "\n".join(satirlar)


def precision_at_k(k=3):
    """
    veri/etiketler.csv: sorgu, rapor_id, ilgili (1 = ilgili, 0 = degil).
    Her sorgu icin her yontemin ilk k sonucundan kaci 'ilgili' diye bakar.
    Dikkat: sadece etiketledigin raporlar sayilir; etiketsiz rapor 'ilgisiz' kabul edilir.
    """
    et = pd.read_csv(ortak.VERI / "etiketler.csv")
    yontemler = ["tfidf"] + (["gomme"] if VEKTORLER is not None else [])
    sonuc = []
    for yontem in yontemler:
        skorlar = []
        for sorgu, g in et.groupby("sorgu"):
            ilgili = set(g.loc[g["ilgili"] == 1, "rapor_id"])
            bulunan = benzer_bul(sorgu, kac_tane=k, yontem=yontem)["id"].tolist()
            skorlar.append(sum(b in ilgili for b in bulunan) / k)
        sonuc.append({"yontem": yontem, f"precision@{k}": round(float(np.mean(skorlar)), 2),
                      "sorgu_sayisi": len(skorlar)})
    return pd.DataFrame(sonuc)


if __name__ == "__main__":
    if "--degerlendir" in sys.argv:
        print(precision_at_k().to_string(index=False))
    else:
        print(bakim_ozeti(motor_no=42, tahmini_rul=25))
        print("\n" + "=" * 70 + "\n")
        print(bakim_ozeti(motor_no=7, tahmini_rul=12, bozulan_sistem="bearing oil"))
