"""
12_rl_karar.py — Q-ogrenme ile bakim karari ve UC POLITIKANIN durust karsilastirmasi.

PEKISTIRMELI OGRENME NEDIR: Ajan bir durumda bir eylem secer, odul alir ve
tecrubeden ogrenir. Satranc oynamayi ogrenen yapay zekalar boyle calisir.

KURGU:
  Durum : modelin tahmin ettigi RUL (kovalara bolunmus) + motorun yasi (kovali)
  Eylem : 0 = bekle, 1 = bakima al
  Odul  : eksi maliyet (planli bakim -1, ariza -10, bosa giden omur kucuk ceza)

NEDEN TABLO TABANLI (Q-table): Durum uzayi kucuk oldugu icin derin aga gerek yok.
Basit olan once denenir; ise yararsa karmasiklastirmaya gerek kalmaz.

DURUST DEGERLENDIRME (bu betigin en onemli ozelligi):
  Politikayi egittigin motorlarda test edersen sonuc oldugundan iyi cikar.
  Bu yuzden her denemede motorlari ikiye bolerim:
     %70 egitim motoru -> A'nin N'ini, B'nin T'sini ve RL'i BUNLARDA ogren/ayarla
     %30 test motoru   -> uc politikayi HIC GORMEDIKLERI motorlarda degerlendir
  Bunu 20 farkli rastgele bolmeyle tekrarlarim ve ortalama +- sapma raporlarim.
  Kullanilan RUL tahminleri de kat-disi (out-of-fold) tahminlerdir (04_modeller.py).

UC POLITIKA:
  A) Sabit takvim    : her motoru N. dongude bakima al
  B) Tahmine dayali  : model "RUL <= T" dedigi ilk anda bakima al
  C) RL (Q-ogrenme)  : ajan ne zaman bakima alacagini kendi ogrenir
       C1: durum = tahmini RUL + yas   C2: durum = sadece tahmini RUL
  (Referans: kusursuz bilgi = her motor arizadan hemen once alinir, maliyet 1)

ONCE CALISTIR: 04_modeller.py
CIKTI: ciktilar/rl_karsilastirma.csv, ciktilar/06_rl_karsilastirma.png,
       modeller/q_tablosu.npy
SURE: 1-3 dakika
"""

import importlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ortak

# 05_politika.py'deki A ve B politikalarini yeniden kullaniyoruz (kodu tekrar yazmamak icin).
# Dosya adi rakamla basladigi icin "import 05_politika" yazamayiz; importlib gerekir.
pol = importlib.import_module("05_politika")

# ---- RL ayarlari ----------------------------------------------------------
# Durum kovalari. Dar kovalar DUSUK RUL bolgesinde: karar orada veriliyor.
RUL_SINIRLARI = np.array([5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100])  # 12 kova olusur
YAS_KOVA_ADIMI = 50      # motorun yasi her 50 dongude bir yeni kova
YAS_KOVA_SAYISI = 6      # 0-49, 50-99, ..., 250+
BOLUM_SAYISI   = 4000    # egitimde kac motor omru oynanacak
OGRENME_HIZI   = 0.1     # yeni bilgi eski bilgiyi ne kadar guncellesin
INDIRIM        = 1.0     # 1.0 = gelecekteki maliyeti indirimsiz say (omur sonlu, maliyet gercek)
KESIF_BASLANGIC, KESIF_BITIS = 1.0, 0.05

DENEME_SAYISI  = 20      # kac farkli rastgele motor bolmesi denenecek
EGITIM_ORANI   = 0.70
EYLEMLER       = (0, 1)  # 0 = bekle, 1 = bakima al


def durumlar(tahmin, dongu):
    """
    Bir motorun TUM donguleri icin durum indekslerini tek seferde hesaplar.
    np.digitize: her tahmini hangi RUL kovasina dustugunu soyler (0..11).
    Yas kovasi: dongu // 50, en fazla 5.
    """
    r = np.digitize(tahmin, RUL_SINIRLARI)
    y = np.minimum(dongu // YAS_KOVA_ADIMI, YAS_KOVA_SAYISI - 1)
    return r.astype(int), y.astype(int)


def motorlari_yukle():
    """Her motor icin (omur, tahmin_dizisi, gercek_RUL, rul_kova, yas_kova) dondurur."""
    df = pd.read_csv(ortak.VERI / "cv_tahminleri.csv")
    motorlar = []
    for _, d in df.groupby("unit"):
        d = d.sort_values("cycle")
        tahmin, cyc = d["tahmin"].to_numpy(), d["cycle"].to_numpy()
        r, y = durumlar(tahmin, cyc)
        motorlar.append({"omur": int(d["omur"].iloc[0]), "tahmin": tahmin,
                         "rul": d["RUL"].to_numpy(), "r": r, "y": y})
    return motorlar


def karar_ver(q):
    """
    Bir durumdaki iki eylem degerinden birini secer: [bekle, bakima al].
    GUVENLI VARSAYILAN: iki deger esitse (ornegin o durum hic gorulmemis, ikisi de 0)
    BAKIMA AL. Bilmedigi durumda beklemek arizaya yol acabilir; bakima almak ise
    sadece biraz israftir. np.argmax esitlikte ilkini (bekle) secerdi, bunu istemiyoruz.
    """
    return 1 if q[1] >= q[0] else 0


def rl_egit(motorlar, birim, rng):
    """Q-tablosunu egitir. Q[rul_kova, yas_kova, eylem] = o eylemin beklenen (eksi) maliyeti."""
    Q = np.zeros((len(RUL_SINIRLARI) + 1, YAS_KOVA_SAYISI, 2))
    for bolum in range(BOLUM_SAYISI):
        # Kesif orani zamanla azalir: basta dene, sonra ogrendigini kullan
        kesif = max(KESIF_BITIS, KESIF_BASLANGIC * (1 - bolum / BOLUM_SAYISI))
        m = motorlar[rng.integers(len(motorlar))]
        n = len(m["r"])
        for i in range(n):
            r, y = m["r"][i], m["y"][i]
            # Epsilon-greedy: ya rastgele dene ya en iyi bildigini yap
            eylem = int(rng.integers(2)) if rng.random() < kesif else karar_ver(Q[r, y])

            if eylem == 1:                       # bakima al: bolum biter
                odul = -(ortak.MALIYET_PLANLI + birim * m["rul"][i])
                Q[r, y, 1] += OGRENME_HIZI * (odul - Q[r, y, 1])
                break
            if i == n - 1:                       # bekledi ve motor arizalandi: bolum biter
                Q[r, y, 0] += OGRENME_HIZI * (-ortak.MALIYET_ARIZA - Q[r, y, 0])
                break
            # Bekledi, motor saglam: anlik maliyet yok, bir sonraki durumun degerini tasi
            hedef = INDIRIM * np.max(Q[m["r"][i + 1], m["y"][i + 1]])
            Q[r, y, 0] += OGRENME_HIZI * (hedef - Q[r, y, 0])
    return Q


def rl_degerlendir(Q, motorlar, birim):
    """Ogrenilen politikayi (her durumda en iyi eylem) verilen motorlarda calistirir."""
    maliyet = israf = ariza = 0
    for m in motorlar:
        # Q'ya gore "bakima al" diyen ilk dongu
        q = Q[m["r"], m["y"]]                  # her dongu icin [bekle, bakima al] degerleri
        karar = q[:, 1] >= q[:, 0]             # esitlikte bakima al (karar_ver ile ayni kural)
        if karar.any():
            i = int(np.argmax(karar))
            maliyet += ortak.MALIYET_PLANLI
            israf += int(m["rul"][i])
        else:
            maliyet += ortak.MALIYET_ARIZA
            ariza += 1
    maliyet += birim * israf
    n = len(motorlar)
    return {"maliyet_motor_basi": maliyet / n, "ariza": ariza, "israf_motor_basi": israf / n}


def yassiz(motorlar):
    """Ayni motorlar, ama yas bilgisi silinmis (hepsi tek yas kovasi). Durum sadece tahmini RUL olur."""
    return [{**m, "y": np.zeros_like(m["y"])} for m in motorlar]


def a_b_icin_motor_listesi(motorlar):
    """05_politika.py (omur, tahmin, rul) uclusu bekler; ayni bicime cevir."""
    return [(m["omur"], m["tahmin"], m["rul"]) for m in motorlar]


def sonuc_satiri(r, n):
    """05_politika sonucunu (toplam maliyet) motor basina maliyete cevirir."""
    return {"maliyet_motor_basi": r["maliyet"] / n, "ariza": r["ariza"],
            "israf_motor_basi": r["israf_motor_basi"]}


if __name__ == "__main__":
    tum = motorlari_yukle()
    N_motor = len(tum)
    ort_omur = np.mean([m["omur"] for m in tum])
    birim = ortak.MALIYET_PLANLI / ort_omur          # bir donguluk omrun bedeli
    print(f"{N_motor} motor, ortalama omur {ort_omur:.0f} dongu, {DENEME_SAYISI} deneme x "
          f"{BOLUM_SAYISI} bolum RL egitimi...\n")

    satirlar = []
    for deneme in range(DENEME_SAYISI):
        rng = np.random.default_rng(1000 + deneme)
        sira = rng.permutation(N_motor)
        n_egitim = int(EGITIM_ORANI * N_motor)
        egitim = [tum[i] for i in sira[:n_egitim]]
        test = [tum[i] for i in sira[n_egitim:]]

        # A ve B: parametreyi EGITIM motorlarinda sec, TEST motorlarinda degerlendir
        e_ab, t_ab = a_b_icin_motor_listesi(egitim), a_b_icin_motor_listesi(test)
        A = [pol.politika_a(e_ab, N, birim) for N in range(60, 230, 2)]
        B = [pol.politika_b(e_ab, T, birim) for T in range(2, 100, 1)]
        N_sec = pol.en_iyiyi_sec(A)["parametre"]
        T_sec = pol.en_iyiyi_sec(B)["parametre"]
        sonuc = {
            "A_sabit_takvim": sonuc_satiri(pol.politika_a(t_ab, N_sec, birim), len(test)),
            "B_tahmine_dayali": sonuc_satiri(pol.politika_b(t_ab, T_sec, birim), len(test)),
        }
        # C1: RL, durum = tahmini RUL + yas.  C2: RL, durum = sadece tahmini RUL.
        # Ikisini de EGITIM motorlarinda egit, TEST motorlarinda degerlendir.
        Q1 = rl_egit(egitim, birim, rng)
        sonuc["C1_RL_RUL_ve_yas"] = rl_degerlendir(Q1, test, birim)
        Q2 = rl_egit(yassiz(egitim), birim, rng)
        sonuc["C2_RL_sadece_RUL"] = rl_degerlendir(Q2, yassiz(test), birim)

        for ad, r in sonuc.items():
            satirlar.append({"deneme": deneme, "politika": ad, "parametre": {
                "A_sabit_takvim": N_sec, "B_tahmine_dayali": T_sec}.get(ad, ""),
                "test_motor": len(test), **r})
        print(f"  deneme {deneme + 1:>2}/{DENEME_SAYISI}: "
              f"A={sonuc['A_sabit_takvim']['maliyet_motor_basi']:.2f}  "
              f"B={sonuc['B_tahmine_dayali']['maliyet_motor_basi']:.2f}  "
              f"C1={sonuc['C1_RL_RUL_ve_yas']['maliyet_motor_basi']:.2f}  "
              f"C2={sonuc['C2_RL_sadece_RUL']['maliyet_motor_basi']:.2f}  "
              f"(ariza A/B/C1/C2: {sonuc['A_sabit_takvim']['ariza']}/"
              f"{sonuc['B_tahmine_dayali']['ariza']}/{sonuc['C1_RL_RUL_ve_yas']['ariza']}/"
              f"{sonuc['C2_RL_sadece_RUL']['ariza']})")

    tablo = pd.DataFrame(satirlar)
    tablo.to_csv(ortak.CIKTILAR / "rl_karsilastirma.csv", index=False)

    ozet = tablo.groupby("politika").agg(
        maliyet_ort=("maliyet_motor_basi", "mean"), maliyet_sapma=("maliyet_motor_basi", "std"),
        ariza_ort=("ariza", "mean"), ariza_toplam=("ariza", "sum"),
        israf_ort=("israf_motor_basi", "mean")).round(2)
    print("\n=== UC POLITIKA (TEST motorlari, 20 denemenin ortalamasi) ===")
    print("  maliyet = motor basina; ariza_ort = denemede ortalama arizalanan motor (30 test motorundan)")
    print(ozet.to_string())
    a = ozet.loc["A_sabit_takvim", "maliyet_ort"]
    for ad in ("B_tahmine_dayali", "C1_RL_RUL_ve_yas", "C2_RL_sadece_RUL"):
        k = ozet.loc[ad, "maliyet_ort"]
        print(f"  {ad}: sabit takvime gore %{100 * (a - k) / a:.0f} daha ucuz")
    print("  Referans (kusursuz bilgi): motor basi maliyet 1.00")

    # Grafik: ortalama maliyet, hata cubugu = denemeler arasi standart sapma
    sira_ad = ["A_sabit_takvim", "B_tahmine_dayali", "C1_RL_RUL_ve_yas", "C2_RL_sadece_RUL"]
    etiket = ["Sabit takvim", "Tahmine dayali\n(esik)", "RL\n(RUL + yas)", "RL\n(sadece RUL)"]
    renk = [pol.GRI, pol.TURUNCU, "#7A93B5", pol.LACIVERT]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(etiket, [ozet.loc[k, "maliyet_ort"] for k in sira_ad],
           yerr=[ozet.loc[k, "maliyet_sapma"] for k in sira_ad], color=renk, capsize=4)
    ax.axhline(1.0, color="#B3261E", ls="--", lw=1, label="Kusursuz bilgi (1.0)")
    ax.set_ylabel("Motor basina maliyet (birim)")
    ax.set_title("Hic gorulmemis motorlarda bakim politikalari")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(ortak.CIKTILAR / "06_rl_karsilastirma.png", dpi=150)
    plt.close()

    # Gosterim icin: TUM motorlarla egitilmis bir Q tablosu kaydet (panelde kullanilabilir)
    Q_tum = rl_egit(yassiz(tum), birim, np.random.default_rng(42))
    np.save(ortak.MODELLER / "q_tablosu.npy", Q_tum)
    print("\nKaydedildi -> ciktilar/rl_karsilastirma.csv, 06_rl_karsilastirma.png, modeller/q_tablosu.npy")
