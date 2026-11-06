"""
ortak.py — Butun py  dosyalarının ortak kullandigi ayarlar ve yardimci fonksiyonlar.

Bu dosyayi dogrudan calistirmazsin. Diger dosyalar "import ortak" diyerek
buradaki ayarlari ve fonksiyonlari kullanir. Boylece bir ayari degistirmek
istedigimizde tek bir yerden degistiririz.
"""

# ---- 1. Kutuphane cagirmalari -------------------------------------------
from pathlib import Path      # Dosya yollarini isletim sisteminden bagimsiz yazmak icin
import numpy as np            # Sayisal hesaplar (dizi, matematik)
import pandas as pd           # Tablo halindeki veriyle calismak icin

# ---- 2. Klasor yollari ---------------------------------------------------
# __file__ = bu dosyanin kendi yolu. .parent = icinde bulundugu klasor (kod/).
# .parent tekrar = projenin ana klasoru. Boylece betigi nereden calistirirsan
# calistir, yollar dogru cozulur.
ANA_KLASOR  = Path(__file__).parent.parent
VERI        = ANA_KLASOR / "veri"
CIKTILAR    = ANA_KLASOR / "ciktilar"
MODELLER    = ANA_KLASOR / "modeller"

# Klasorler yoksa olustur. exist_ok=True: zaten varsa hata verme.
for k in (CIKTILAR, MODELLER):
    k.mkdir(exist_ok=True)

# ---- 3. Veri setinin sutun isimleri --------------------------------------
# Ham .txt dosyalarinda basliklar yoktur, sadece sayilar vardir.
# Sutun isimlerini NASA'nin veri dokumanindan biz veriyoruz.
SUTUNLAR = (["unit", "cycle", "op1", "op2", "op3"]
            + [f"s{i}" for i in range(1, 22)])   # s1, s2, ... s21

# ---- 4. Proje genelinde gecerli varsayimlar ------------------------------
# Bunlar benim kararlarım. 
RUL_UST_SINIR = 125   # Motor yeniyken "300 dongu kaldi" ile "200 kaldi" arasinda
                      # sensorlerde fark yok. Bu yuzden buyuk degerleri kirpiyoruz.
ARIZA_ESIGI   = 30    # RUL bu degerin altindaysa "ariza yakin" (1) diyoruz.
PENCERE       = 30    # Derin ogrenme icin kac donguluk pencere kullanilacak.

# Maliyet modeli (Hafta 2'de kullanilacak) — bunlar da varsayim.
MALIYET_PLANLI = 1.0    # Planli bakimin bedeli
MALIYET_ARIZA  = 10.0   # Arizanin bedeli (planlinin 10 kati)


def ham_veri_oku(dosya_adi):
    """
    Ham .txt dosyasini okuyup pandas tablosuna cevirir.

    dosya_adi: "train_FD001" gibi, uzantisiz isim.
    Doner: pandas DataFrame (tablo)
    """
    # sep=r"\s+" : sutunlar arasinda bir veya daha fazla bosluk var demek.
    # header=None: dosyanin ilk satiri baslik degil, veridir.
    # names=SUTUNLAR: basliklari biz veriyoruz.
    return pd.read_csv(VERI / f"{dosya_adi}.txt", sep=r"\s+",
                       header=None, names=SUTUNLAR)


def rul_etiketle(df):
    """
    Egitim verisine hedef sutunlarini ekler.

    Mantik: egitim motorlarini arizalanana kadar izliyoruz. Yani bir motorun
    EN BUYUK dongu numarasi, onun arizalandigi andir.
    RUL = o motorun son dongusu - su anki dongu.
    """
    # groupby("unit"): her motoru ayri ele al.
    # transform("max"): her satira, o motorun en buyuk dongu degerini yaz.
    df["omur"] = df.groupby("unit")["cycle"].transform("max")

    # Kalan omur = toplam omur - su anki dongu
    df["RUL"] = df["omur"] - df["cycle"]

    # Ust sinir uygulanmis hali (modelin ogrenmesi icin)
    df["RUL_kirpik"] = df["RUL"].clip(upper=RUL_UST_SINIR)

    # Siniflandirma hedefi: ariza yakin mi? 1 = evet, 0 = hayir
    df["ariza_yakin"] = (df["RUL"] <= ARIZA_ESIGI).astype(int)
    return df


def kullanilabilir_sutunlar(df):
    """
    Hic degismeyen (sabit) sutunlari eler ve geriye kalanlari dondurur.

    Neden: bir sensor hep ayni degeri veriyorsa modele hicbir bilgi tasimaz.
    Dahasi, bazi modelleri (olceklemede sifira bolme gibi) bozabilir.
    """
    adaylar = [s for s in SUTUNLAR if s not in ("unit", "cycle")]
    # Iki olcut birden:
    #   std() > 1e-6     -> hic degismeyen sutunlari eler (s1, s16, s18, s19, op3)
    #   nunique() > 2    -> sadece 2 farkli deger alan sutunlari da eler (s6, s10)
    # Ikinci olcut lazim cunku s6 gibi bir sensor "degisiyor" gorunur ama
    # aslinda sadece iki kademe arasinda gidip gelir; modele bilgi tasimaz
    # ve istatistik testlerinde sayisal uyari uretir.
    return [s for s in adaylar if df[s].std() > 1e-6 and df[s].nunique() > 2]


def nasa_skor(tahmin, gercek):
    """
    NASA'nin asimetrik ceza fonksiyonu.

    Neden asimetrik: motorun omrunu OLDUGUNDAN FAZLA tahmin etmek (gec uyari)
    tehlikelidir, cunku bakim zamaninda yapilmaz. Erken tahmin ise sadece
    gereksiz bakimdir. Bu yuzden gec tahmin daha agir cezalandirilir.
    """
    d = np.asarray(tahmin) - np.asarray(gercek)   # fark: + ise gec, - ise erken
    # np.where(kosul, kosul_dogruysa, kosul_yanlissa)
    return float(np.sum(np.where(d < 0,
                                 np.exp(-d / 13) - 1,   # erken tahmin: hafif ceza
                                 np.exp( d / 10) - 1))) # gec tahmin: agir ceza


def rmse(tahmin, gercek):
    """
    Root Mean Squared Error = ortalama karesel hatanin karekoku.
    Tahminlerin ortalama olarak kac dongu saptigini soyler. Kucuk olmasi iyidir.
    """
    return float(np.sqrt(np.mean((np.asarray(tahmin) - np.asarray(gercek)) ** 2)))
