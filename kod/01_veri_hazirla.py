"""
01_veri_hazirla.py — Ham veriyi okur, hedef sutunlarini ekler, CSV olarak kaydeder.

"""

import pandas as pd
import ortak   # kendi yazdigimiz ortak.py dosyasi


def egitim_verisini_hazirla():
    """Egitim verisi: motorlar arizalanana kadar izlenmis. RUL'u kendimiz hesaplariz."""
    print("Egitim verisi okunuyor...")
    df = ortak.ham_veri_oku("train_FD001")      # ham .txt -> tablo
    df = ortak.rul_etiketle(df)                 # RUL, RUL_kirpik, ariza_yakin ekle

    print(f"  {df['unit'].nunique()} motor, {len(df)} satir")
    print(f"  Eksik deger sayisi: {df.isna().sum().sum()}")
    # DIKKAT: ortalamayi motor bazinda al. df['omur'].mean() satir bazli olur ve
    # uzun omurlu motorlarin satiri cok oldugu icin yanlis (yuksek) cikar.
    omurler = df.groupby('unit')['omur'].first()
    print(f"  Motor omru: en kisa {omurler.min()}, en uzun {omurler.max()}, "
          f"ortalama {omurler.mean():.0f} dongu")
    print(f"  'Ariza yakin' etiketli satir orani: %{100*df['ariza_yakin'].mean():.1f}")
    return df


def test_verisini_hazirla():
    """
    Test verisi FARKLI: motorlar arizadan ONCE kesilmis.
    Yani son dongu, ariza anı degil. Gercek kalan omur ayri bir dosyada
    (RUL_FD001.txt) veriliyor: her motorun SON dongusundeki RUL degeri.
    """
    print("Test verisi okunuyor...")
    df  = ortak.ham_veri_oku("test_FD001")
    # RUL dosyasi tek sutunlu: 1. satir 1. motorun, 2. satir 2. motorun RUL'u...
    rul = pd.read_csv(ortak.VERI / "RUL_FD001.txt", header=None)[0]

    # Her motorun kayittaki son dongusu
    son_dongu = df.groupby("unit")["cycle"].max()

    # Motorun gercek toplam omru = kayittaki son dongu + dosyadaki kalan omur
    # rul.iloc[u-1] : motor 1 -> satir 0, motor 2 -> satir 1 ... (indeks 0'dan baslar)
    gercek_omur = {u: son_dongu[u] + rul.iloc[u - 1] for u in son_dongu.index}

    df["omur"]        = df["unit"].map(gercek_omur)
    df["RUL"]         = df["omur"] - df["cycle"]
    df["RUL_kirpik"]  = df["RUL"].clip(upper=ortak.RUL_UST_SINIR)
    df["ariza_yakin"] = (df["RUL"] <= ortak.ARIZA_ESIGI).astype(int)

    print(f"  {df['unit'].nunique()} motor, {len(df)} satir")
    return df


def sabit_sutunlari_bildir(df):
    """Hangi sensorler hic degismiyor? Bunlari modelde kullanmayacagiz."""
    tumu  = [s for s in ortak.SUTUNLAR if s not in ("unit", "cycle")]
    ise_yarar = ortak.kullanilabilir_sutunlar(df)
    sabit = [s for s in tumu if s not in ise_yarar]
    print(f"\nSabit (ise yaramaz) sutunlar [{len(sabit)} adet]: {sabit}")
    print(f"Kullanilabilir sutunlar     [{len(ise_yarar)} adet]: {ise_yarar}")
    return ise_yarar


# Bu blok sadece dosya DOGRUDAN calistirildiginda isler.
# Baska bir dosya "import 01_veri_hazirla" derse calismaz.
if __name__ == "__main__":
    egitim = egitim_verisini_hazirla()
    test   = test_verisini_hazirla()
    sabit_sutunlari_bildir(egitim)

    # index=False : pandas'in otomatik satir numarasini dosyaya yazma
    egitim.to_csv(ortak.VERI / "FD001_egitim.csv", index=False)
    test.to_csv(  ortak.VERI / "FD001_test.csv",   index=False)
    print(f"\nKaydedildi -> {ortak.VERI / 'FD001_egitim.csv'}")
    print(f"Kaydedildi -> {ortak.VERI / 'FD001_test.csv'}")
