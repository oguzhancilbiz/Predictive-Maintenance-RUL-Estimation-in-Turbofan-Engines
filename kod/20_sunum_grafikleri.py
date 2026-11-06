"""
20_sunum_grafikleri.py — Sunumda kullanilan sade grafikleri ciktilar/ klasorundeki GERCEK sonuc
dosyalarindan cizer. Hicbir sayi elle yazilmadi; her grafik bir CSV'den okunur.
ONCE CALISTIR: 04, 07, 10 (ve tohum 1-9), 11, 12, 16, 17, 18, 19; RAG icin 14b ile etiketler.csv.
CIKTI: ciktilar/sunum/*.png
"""
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ortak

C = ortak.CIKTILAR
OUT = C / "sunum"
OUT.mkdir(exist_ok=True)
MAVI, TURUNCU, GRI, KOYU, ACIK = "#2a78d6", "#eb6834", "#9a9a95", "#262624", "#e6e5e0"
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": GRI, "axes.labelcolor": KOYU, "xtick.color": KOYU,
                     "ytick.color": KOYU, "axes.titleweight": "bold", "figure.dpi": 100})


def kaydet(fig, ad):
    fig.tight_layout()
    fig.savefig(OUT / ad, dpi=200, facecolor="white")
    plt.close(fig)


# 1) Sizinti deneyi
t = pd.read_csv(C / "sizinti_deneyi.csv")
fig, ax = plt.subplots(figsize=(7, 4.6))
b = ax.bar(["Satırları rastgele\nböldüm (yanlış)", "Motorlara göre\nböldüm (doğru)"], t["RMSE"],
           color=[GRI, MAVI], width=0.55)
for r, v in zip(b, t["RMSE"]):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.3, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_ylabel("Hata (RMSE, küçük iyi)"); ax.set_ylim(0, 20)
kazanc = 100 * (t["RMSE"][1] - t["RMSE"][0]) / t["RMSE"][1]
ax.text(0.5, 17.6, f"Rastgele bölme hatayı\nyaklaşık %{kazanc:.0f} daha küçük gösteriyor", ha="center", color=TURUNCU, fontweight="bold")
kaydet(fig, "01_sizinti.png")

# 2) Model karsilastirmasi (motor bazli capraz dogrulama)
t = pd.read_csv(C / "model_karsilastirma.csv").sort_values("RMSE", ascending=False)
fig, ax = plt.subplots(figsize=(7, 4.6))
renk = [MAVI if m in ("XGBoost", "LightGBM") else GRI for m in t["Model"]]
b = ax.barh(t["Model"], t["RMSE"], color=renk, height=0.6)
for r, v in zip(b, t["RMSE"]):
    ax.text(v + 0.2, r.get_y() + r.get_height() / 2, f"{v:.2f}", va="center", fontweight="bold")
ax.set_xlabel("Hata (RMSE, küçük iyi) — motor bazlı 5 katlı doğrulama"); ax.set_xlim(0, 22)
kaydet(fig, "02_modeller.png")

# 3) LSTM tohumlar: her tohumda iki kaybin NASA skoru (eslesmis)
dos = sorted(glob.glob(str(C / "lstm_sonuclari*.csv")))
satir = []
for i, f in enumerate(dos):
    d = pd.read_csv(f).set_index("Model")
    satir.append((d.loc["LSTM - MSE (standart)", "Test_NASA"], d.loc["LSTM - Asimetrik (NASA)", "Test_NASA"]))
a = np.array(satir)
fig, ax = plt.subplots(figsize=(7, 4.6))
for m, n in a:
    ax.plot([0, 1], [m, n], color=(TURUNCU if n < m else GRI), alpha=0.8, lw=1.8, marker="o", ms=6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Standart kayıp\n(MSE)", "Asimetrik kayıp\n(NASA)"])
ax.set_xlim(-0.55, 1.55); ax.set_ylabel("NASA skoru (küçük iyi)")
ax.text(-0.08, a[:, 0].mean(), f"ort. {a[:, 0].mean():.0f}", ha="right", va="center", fontweight="bold")
ax.text(1.08, a[:, 1].mean(), f"ort. {a[:, 1].mean():.0f}", ha="left", va="center", fontweight="bold", color=TURUNCU)
ax.set_title(f"Her çizgi bir tohum ({len(a)} tohum): {int((a[:, 1] < a[:, 0]).sum())}'unda düştü", fontsize=12)
kaydet(fig, "03_lstm_tohum.png")

# 4) Maliyet duyarliligi
t = pd.read_csv(C / "duyarlilik.csv").groupby(["ariza_maliyeti", "politika"])["maliyet_motor_basi"].mean().unstack()
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.plot(t.index, t["A_sabit_takvim"], color=GRI, lw=2.5, marker="o", ms=8)
ax.plot(t.index, t["B_esik"], color=MAVI, lw=2.5, marker="o", ms=8)
ax.text(t.index[0], t["A_sabit_takvim"].iloc[0] + 0.06, "Sabit takvim", color=KOYU, ha="left", fontweight="bold")
ax.text(t.index[0], t["B_esik"].iloc[0] - 0.09, "Tahmine dayalı eşik", color=MAVI, ha="left", fontweight="bold")
ax.set_xlabel("Arıza maliyeti (planlı bakım = 1 birim)"); ax.set_ylabel("Motor başına maliyet")
ax.set_xscale("log"); ax.set_xticks(t.index); ax.set_xticklabels([str(x) for x in t.index])
ax.minorticks_off(); ax.set_ylim(0.85, 2.2)
kaydet(fig, "04_maliyet_duyarlilik.png")

# 5) Politikalar (ayrilmis motorlarda, 20 bolme ortalamasi)
t = pd.read_csv(C / "rl_karsilastirma.csv").groupby("politika").agg(
    m=("maliyet_motor_basi", "mean"), a=("ariza", "sum")).reindex(
    ["A_sabit_takvim", "B_tahmine_dayali", "C1_RL_RUL_ve_yas", "C2_RL_sadece_RUL"])
ad = ["Sabit\ntakvim", "Eşik\n(tahmine dayalı)", "Q-öğrenme\n(RUL + yaş)", "Q-öğrenme\n(sadece RUL)"]
fig, ax = plt.subplots(figsize=(7.6, 4.6))
b = ax.bar(ad, t["m"], color=[GRI, MAVI, TURUNCU, TURUNCU], width=0.58)
for r, v, k in zip(b, t["m"], t["a"]):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.03, f"{v:.2f}\n{int(k)} arıza", ha="center", fontweight="bold", fontsize=11)
ax.set_ylabel("Motor başına maliyet (küçük iyi)"); ax.set_ylim(0, 1.9)
kaydet(fig, "05_politikalar.png")

# 6) Anomali: uyari payi dagilimi
t = pd.read_csv(C / "anomali_baslangic.csv")
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.hist(t["o_anda_kalan_omur"], bins=np.arange(10, 95, 5), color=MAVI, edgecolor="white", linewidth=2)
med = t["o_anda_kalan_omur"].median()
ax.axvline(med, color=TURUNCU, lw=2)
ax.text(med + 1.5, ax.get_ylim()[1] * 0.92, f"ortanca {med:.0f} döngü", color=TURUNCU, fontweight="bold")
ax.set_xlabel("Bozulma işareti verildiğinde arızaya kalan döngü"); ax.set_ylabel("Motor sayısı")
kaydet(fig, "06_anomali.png")

# 7) Alt kumeler
t = pd.read_csv(C / "alt_kumeler.csv")
fig, ax = plt.subplots(figsize=(7, 4.6))
etiket = [f"{a}\n{int(k)} koşul" for a, k in zip(t["alt_kume"], t["kosul_sayisi"])]
b = ax.bar(etiket, t["CV_RMSE"], color=[MAVI, TURUNCU, MAVI, TURUNCU], width=0.55)
for r, v in zip(b, t["CV_RMSE"]):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center", fontweight="bold")
ax.set_ylabel("Hata (RMSE) — motor bazlı doğrulama"); ax.set_ylim(0, 20)
kaydet(fig, "07_alt_kumeler.png")

# 8) RUL siniri
t = pd.read_csv(C / "rul_siniri.csv")
fig, ax = plt.subplots(figsize=(7, 4.6))
b = ax.bar(t["rul_siniri"].astype(str), t["RMSE_ortak_125"],
           color=[GRI if s not in (125, 130) else MAVI for s in t["rul_siniri"]], width=0.55)
for r, v in zip(b, t["RMSE_ortak_125"]):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.15, f"{v:.2f}", ha="center", fontweight="bold")
ax.set_xlabel("RUL üst sınırı (kırpma değeri)"); ax.set_ylabel("Hata (RMSE, küçük iyi)"); ax.set_ylim(14, 20)
kaydet(fig, "08_rul_siniri.png")

# 9) RAG sorgu bazinda precision@3
et = pd.read_csv(ortak.VERI / "etiketler.csv")
p = et.groupby("sorgu", sort=False)["ilgili"].apply(lambda s: s.iloc[:3].mean())
kisa = list(p.index)
fig, ax = plt.subplots(figsize=(9, 5.4))
o = np.argsort(p.values, kind="stable")
ax.barh(np.array(kisa)[o], p.values[o], color=[MAVI if v > 0.99 else TURUNCU for v in p.values[o]], height=0.62)
ax.axvline(p.mean(), color=KOYU, ls="--", lw=1.2)
ax.text(p.mean() + 0.01, -0.9, f"ortalama {p.mean():.2f}", fontweight="bold", fontsize=11)
ax.set_xlim(0, 1.05); ax.set_xlabel("İlk 3 sonucun kaçı ilgili (precision@3)")
ax.tick_params(axis="y", labelsize=10)
kaydet(fig, "09_rag_precision.png")
print("Grafikler ->", OUT)
