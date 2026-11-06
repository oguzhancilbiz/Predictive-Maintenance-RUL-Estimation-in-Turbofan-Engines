"""21_mimari_sekli.py — Sunumdaki sistem sekli (kutu-ok cizimi). CIKTI: ciktilar/sunum/00_mimari.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import ortak

fig, ax = plt.subplots(figsize=(13, 4.8)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.8); ax.axis("off")
def kutu(x, y, w, h, baslik, alt, renk, yazi="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=renk, ec="none"))
    ax.text(x + w / 2, y + h * 0.64, baslik, ha="center", va="center", color=yazi, fontsize=14, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, alt, ha="center", va="center", color=yazi, fontsize=10.5)
def ok(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#6b6b66", lw=2))
G, M, T = "#6b7280", "#2a78d6", "#eb6834"
kutu(0.2, 2.9, 2.6, 1.4, "NASA C-MAPSS", "100 motor · 21 sensör\n(simülasyon)", G)
kutu(0.2, 0.5, 2.6, 1.4, "FAA SDR raporları", "5.548 motor arıza anlatımı\n(2023–2025)", G)
kutu(3.9, 2.9, 2.8, 1.4, "1 · Sensör katmanı", "kalan ömür (RUL) tahmini\n+ anomali tespiti", M)
kutu(3.9, 0.5, 2.8, 1.4, "3 · Bilgi katmanı", "TF-IDF ile benzer\ngeçmiş rapor arama", M)
kutu(7.8, 1.7, 2.3, 1.5, "2 · Karar katmanı", "ne zaman bakım?\nmaliyet · eşik · RL", T)
kutu(10.9, 1.7, 2.0, 1.5, "4 · Ürün", "panel · API\nMLflow", "#262624")
ok(2.8, 3.6, 3.9, 3.6); ok(2.8, 1.2, 3.9, 1.2); ok(6.7, 3.6, 7.8, 2.9); ok(6.7, 1.2, 7.8, 2.0); ok(10.1, 2.45, 10.9, 2.45)
fig.savefig(ortak.CIKTILAR / "sunum" / "00_mimari.png", dpi=200, facecolor="white", bbox_inches="tight")
