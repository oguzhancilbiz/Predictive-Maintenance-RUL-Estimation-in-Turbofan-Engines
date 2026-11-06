"""
05_politika.py — Bakim karar politikalarinin karsilastirilmasi.

BU BETIK PROJENIN AYIRT EDICI PARCASI.
Diger betikler "kac dongu kaldi" sorusunu cevapliyor. Bu betik
"ne zaman bakima almaliyim" sorusunu cevapliyor ve sonucu PARAYA ceviriyor.

IKI POLITIKA KARSILASTIRILIR:
  A) Sabit takvim    : her N donguye bir bakim (klasik yontem)
  B) Tahmine dayali  : model "RUL <= T" dediginde bakim

MALIYET MODELI (varsayim, raporda belirt):
  planli bakim      = 1 birim
  ariza             = 10 birim
  bosa giden omur   = her dongu icin 1/ortalama_omur birim
                      (motoru erken degistirince kalan omru cope atmis olursun)


ONCE CALISTIR: 04_modeller.py
CIKTI: ciktilar/politika_karsilastirma.png ve politika_sonuclari.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ortak

LACIVERT, TURUNCU, GRI = "#1F3A5F", "#D9822B", "#55606E"


def motorlari_hazirla():
    """
    Her motoru (omur, tahmin_dizisi, gercek_RUL_dizisi) ucluzu haline getirir.
    Donguler icinde pandas kullanmak yavastir; once numpy dizilerine ceviriyoruz.
    """
    df = pd.read_csv(ortak.VERI / "cv_tahminleri.csv")
    motorlar = []
    for _, d in df.groupby("unit"):
        d = d.sort_values("cycle")
        motorlar.append((int(d["omur"].iloc[0]),      # motorun toplam omru
                         d["tahmin"].to_numpy(),      # modelin her dongudeki tahmini
                         d["RUL"].to_numpy()))        # gercek kalan omur
    return motorlar


def politika_a(motorlar, N, birim_dongu_bedeli):
    """
    SABIT TAKVIM: motoru N. donguye geldiginde bakima al.

    Iki sonuc mumkun:
      - Motor N'den once arizalandi  -> ariza maliyeti (gec kaldik)
      - Motor N'e ulasti             -> planli bakim + bosa giden omur (erken aldik)
    """
    maliyet = arizalar = israf = 0
    for omur, _, _ in motorlar:
        if omur <= N:
            maliyet += ortak.MALIYET_ARIZA      # yetisemedik
            arizalar += 1
        else:
            maliyet += ortak.MALIYET_PLANLI
            israf += omur - N                   # motorun kullanilmayan omru
    maliyet += birim_dongu_bedeli * israf
    return {"parametre": N, "maliyet": maliyet, "ariza": arizalar,
            "israf_motor_basi": israf / len(motorlar)}


def politika_b(motorlar, T, birim_dongu_bedeli):
    """
    TAHMINE DAYALI: model "kalan omur <= T" dedigi ilk anda bakima al.

    Not: tahmin hatali olabilir. Model gec uyarirsa motor arizalanir.
    """
    maliyet = arizalar = israf = 0
    for omur, tahminler, gercek_rul in motorlar:
        # Esigi asan ilk dongu. argmax, True olan ilk indeksi verir.
        i = int(np.argmax(tahminler <= T))
        if tahminler[i] > T:
            # Hic esigin altina inmedi -> uyari hic gelmedi -> motor arizalandi
            maliyet += ortak.MALIYET_ARIZA
            arizalar += 1
        else:
            maliyet += ortak.MALIYET_PLANLI
            israf += int(gercek_rul[i])        # o anda motorun GERCEK kalan omru
    maliyet += birim_dongu_bedeli * israf
    return {"parametre": T, "maliyet": maliyet, "ariza": arizalar,
            "israf_motor_basi": israf / len(motorlar)}


def en_iyiyi_sec(sonuclar, sifir_ariza_sarti=True):
    """
    En dusuk maliyetli politikayi secer.
    sifir_ariza_sarti=True ise once hic ariza olmayanlara bakar (emniyet onceligi).
    """
    aday = [r for r in sonuclar if r["ariza"] == 0] if sifir_ariza_sarti else sonuclar
    if not aday:                 # hicbiri sifir ariza vermediyse hepsine bak
        aday = sonuclar
    return min(aday, key=lambda r: r["maliyet"])


def grafik_ciz(A, B, en_iyi_a, en_iyi_b):
    """Iki panel: solda odunlesim egrisi, sagda secilen politikalarin karsilastirmasi."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    # SOL: her parametre degeri icin (israf, ariza) noktasi.
    # Sol-alta yakin olmak iyidir: az israf, az ariza.
    ax[0].plot([r["israf_motor_basi"] for r in A], [r["ariza"] for r in A],
               "-o", ms=3, color=GRI, label="Sabit takvim")
    ax[0].plot([r["israf_motor_basi"] for r in B], [r["ariza"] for r in B],
               "-o", ms=3, color=TURUNCU, label="Tahmine dayali")
    ax[0].set_xlabel("Bosa giden omur (dongu / motor)")
    ax[0].set_ylabel("Arizalanan motor sayisi")
    ax[0].set_title("Odunlesim: israf mi, risk mi?")
    ax[0].legend(frameon=False)

    # SAG: secilen iki politikanin israf ve maliyeti yan yana
    etiket = [f"Sabit takvim\n(N={en_iyi_a['parametre']})",
              f"Tahmine dayali\n(T={en_iyi_b['parametre']})"]
    x = np.arange(2)
    ax[1].bar(x - 0.19, [en_iyi_a["israf_motor_basi"], en_iyi_b["israf_motor_basi"]],
              0.38, color=GRI, label="Bosa giden omur")
    ikinci = ax[1].twinx()           # ayni grafige ikinci bir y ekseni ekle
    ikinci.bar(x + 0.19, [en_iyi_a["maliyet"], en_iyi_b["maliyet"]],
               0.38, color=TURUNCU, label="Toplam maliyet")
    ax[1].set_xticks(x); ax[1].set_xticklabels(etiket)
    ax[1].set_ylabel("Bosa giden omur (dongu/motor)")
    ikinci.set_ylabel("Toplam maliyet (birim)")
    ax[1].set_title("Secilen politikalar")
    for s in ("top",): ax[0].spines[s].set_visible(False); ikinci.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(ortak.CIKTILAR / "04_politika_karsilastirma.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    motorlar = motorlari_hazirla()
    n = len(motorlar)
    ortalama_omur = np.mean([m[0] for m in motorlar])
    # Bir donguluk omrun bedeli: bir bakim maliyetini ortalama omre bolduk
    birim = ortak.MALIYET_PLANLI / ortalama_omur

    print(f"{n} motor, ortalama omur {ortalama_omur:.0f} dongu")
    print(f"Maliyet varsayimlari: planli={ortak.MALIYET_PLANLI}, "
          f"ariza={ortak.MALIYET_ARIZA}, bir dongu omur={birim:.5f}\n")

    # Parametre taramasi: her iki politika icin genis bir aralik dene
    A = [politika_a(motorlar, N, birim) for N in range(60, 230, 2)]
    B = [politika_b(motorlar, T, birim) for T in range(2, 100, 1)]

    en_iyi_a, en_iyi_b = en_iyiyi_sec(A), en_iyiyi_sec(B)

    print("=== EN IYI POLITIKALAR (sifir ariza sartiyla) ===")
    for ad, r in [("Sabit takvim", en_iyi_a), ("Tahmine dayali", en_iyi_b)]:
        print(f"{ad:<16} parametre={r['parametre']:>4}  maliyet={r['maliyet']:>7.1f}  "
              f"ariza={r['ariza']}  israf={r['israf_motor_basi']:.0f} dongu/motor")

    kazanc  = 100 * (en_iyi_a["maliyet"] - en_iyi_b["maliyet"]) / en_iyi_a["maliyet"]
    israf_k = 100 * (en_iyi_a["israf_motor_basi"] - en_iyi_b["israf_motor_basi"]) \
              / en_iyi_a["israf_motor_basi"]
    print(f"\nTahmine dayali politika maliyeti %{kazanc:.1f} dusuruyor.")
    print(f"Bosa giden omur %{israf_k:.0f} azaliyor "
          f"({en_iyi_a['israf_motor_basi']:.0f} -> {en_iyi_b['israf_motor_basi']:.0f} dongu/motor).")

    grafik_ciz(A, B, en_iyi_a, en_iyi_b)
    pd.DataFrame([{"politika": "sabit_takvim", **en_iyi_a},
                  {"politika": "tahmine_dayali", **en_iyi_b}]
                 ).to_csv(ortak.CIKTILAR / "politika_sonuclari.csv", index=False)
    print(f"\nGrafik -> ciktilar/04_politika_karsilastirma.png")
