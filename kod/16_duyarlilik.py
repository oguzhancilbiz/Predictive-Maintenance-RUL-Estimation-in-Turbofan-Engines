"""
16_duyarlilik.py — Maliyet varsayimlarina DUYARLILIK analizi.

SORU: Rapordaki sonuc ("tahmine dayali esik, sabit takvimden ucuz") benim KOYDUGUM
maliyet oranlarina mi bagli? Ariza maliyetini 5, 10, 20, 50 yaparsam sonuc degisiyor mu?

YONTEM: 12_rl_karar.py'deki gibi durust degerlendirme: motorlari %70 egitim / %30 test
bol (20 farkli bolme). Her politikanin parametresini (N ya da T) SADECE egitim motorlarinda
sec, test motorlarinda olc. Planli bakim maliyeti 1, israf = 1/ortalama omur sabit.
KAPSAM: Sadece A (sabit takvim) ve B (esik). Q-ogrenme (RL) bu analize dahil DEGIL.

ONCE CALISTIR: 04_modeller.py (cv_tahminleri.csv)
CIKTI: ciktilar/duyarlilik.csv
"""
import importlib
import numpy as np
import pandas as pd
import ortak

pol = importlib.import_module("05_politika")
motorlar = pol.motorlari_hazirla()
n = len(motorlar)
birim = ortak.MALIYET_PLANLI / np.mean([m[0] for m in motorlar])
ARIZA_MALIYETLERI = [2, 5, 10, 20, 50]
DENEME = 20
satirlar = []

for ariza_m in ARIZA_MALIYETLERI:
    ortak.MALIYET_ARIZA = ariza_m                 # 05_politika bu degeri ortak'tan okur
    for d in range(DENEME):
        rng = np.random.default_rng(1000 + d)
        idx = rng.permutation(n)
        egitim = [motorlar[i] for i in idx[: int(0.7 * n)]]
        test = [motorlar[i] for i in idx[int(0.7 * n):]]
        A = [pol.politika_a(egitim, N, birim) for N in range(60, 230, 2)]
        B = [pol.politika_b(egitim, T, birim) for T in range(2, 100)]
        a = min(A, key=lambda r: r["maliyet"])["parametre"]      # sadece maliyete gore (sifir ariza sarti YOK)
        b = min(B, key=lambda r: r["maliyet"])["parametre"]
        for ad, f, p in (("A_sabit_takvim", pol.politika_a, a), ("B_esik", pol.politika_b, b)):
            r = f(test, p, birim)
            satirlar.append({"ariza_maliyeti": ariza_m, "deneme": d, "politika": ad, "parametre": p,
                             "maliyet_motor_basi": r["maliyet"] / len(test), "ariza": r["ariza"],
                             "israf_motor_basi": r["israf_motor_basi"], "test_motor": len(test)})

t = pd.DataFrame(satirlar)
t.to_csv(ortak.CIKTILAR / "duyarlilik.csv", index=False)
ozet = t.groupby(["ariza_maliyeti", "politika"]).agg(
    maliyet=("maliyet_motor_basi", "mean"), ariza_toplam=("ariza", "sum"),
    israf=("israf_motor_basi", "mean"), parametre=("parametre", "mean")).round(2)
print(ozet.to_string())
piv = t.pivot_table(index=["ariza_maliyeti", "deneme"], columns="politika", values="maliyet_motor_basi")
print("\nB'nin A'dan ucuz oldugu deneme sayisi (20 uzerinden):")
print((piv["B_esik"] < piv["A_sabit_takvim"]).groupby(level=0).sum().to_string())
