"""
panel.py — Streamlit ile filo bakim karar paneli (4 katmanin tamami tek ekranda).

KATMANLAR:
  1 Sensor katmani : tahmini kalan omur (RUL) ve anomali baslangici
  2 Karar katmani  : maliyet temelli bakim esigi (05_politika.py'den) ve uyari esigi
  3 Bilgi katmani  : secili motora benzer gecmis FAA SDR raporlari (13/14 numarali betikler)
  4 Urun           : bu panel

NASIL CALISTIRILIR (VS Code terminali, ortam etkinken):
    streamlit run kod/panel.py
Tarayicida http://localhost:8501 acilir. Durdurmak icin terminalde Ctrl+C.

ONCE CALISTIR: 04_modeller.py (cv_tahminleri.csv), 05_politika.py, 11_anomali.py,
               13_metin_indeksi.py (benzer rapor sekmesi icin; yoksa o sekme uyari verir).

ONEMLI: Veri bir SIMULASYON. "Filonun su anki hali" gercekte her motorun en son olcumu olurdu;
burada her motordan bir an sececegiz. Panelde "gercek RUL" sutunu ancak simulasyonda bilinir;
gercek bir filoda yoktur.
"""

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# kod/ klasorunu import yoluna ekle (panel baska bir klasorden calistirilsa bile ortak.py bulunsun)
sys.path.insert(0, str(Path(__file__).parent))
import ortak

# Sayfa ayarlari — bu satir diger st.* cagrilarindan once olmali
st.set_page_config(page_title="Filo Bakım Paneli", layout="wide")

LACIVERT, TURUNCU, GRI, KIRMIZI = "#1F3A5F", "#D9822B", "#55606E", "#B3261E"


# ---- Veri yukleme -----------------------------------------------------------
# @st.cache_data: sonucu hafizada tutar; Streamlit her etkilesimde dosyayi bastan calistirir,
# bu dekorator olmasa veri her seferinde diskten okunurdu.
@st.cache_data
def tahminleri_yukle():
    return pd.read_csv(ortak.VERI / "cv_tahminleri.csv")


@st.cache_data
def anomali_yukle():
    p = ortak.CIKTILAR / "anomali_baslangic.csv"
    return pd.read_csv(p).set_index("motor") if p.exists() else None


@st.cache_data
def maliyet_esigi():
    """05_politika.py'nin sectigi tahmine dayali esik (T). Dosya yoksa 7 varsay."""
    p = ortak.CIKTILAR / "politika_sonuclari.csv"
    if p.exists():
        t = pd.read_csv(p)
        return int(t.loc[t["politika"] == "tahmine_dayali", "parametre"].iloc[0])
    return 7


@st.cache_data
def takvim_araligi():
    """05_politika.py'nin sectigi sabit takvim araligi (N). Dosya yoksa 126 varsay."""
    p = ortak.CIKTILAR / "politika_sonuclari.csv"
    if p.exists():
        t = pd.read_csv(p)
        return int(t.loc[t["politika"] == "sabit_takvim", "parametre"].iloc[0])
    return 126


@st.cache_resource
def politika_modulu():
    """05_politika.py'yi (bakim politikalari) yukler. Hata olursa hatayi dondurur."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        return importlib.import_module("05_politika")
    except Exception as hata:
        return hata


@st.cache_data
def motorlari_hazirla():
    """
    05_politika.py'nin KENDI fonksiyonunu cagirir: her motoru
    (toplam omur, tahmin dizisi, gercek RUL dizisi) ucluzune cevirir.
    Ayni fonksiyonu kullaniyoruz ki panel ile rapor ayni sayiyi uretsin.
    """
    pol = politika_modulu()
    return None if isinstance(pol, Exception) else pol.motorlari_hazirla()


@st.cache_resource
def rag_modulu():
    """14_rag.py'yi (rapor arama) yukler. Dosyalar yoksa None doner."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        return importlib.import_module("14_rag")
    except Exception as hata:
        return hata


def csv_oku(ad):
    p = ortak.CIKTILAR / ad
    return pd.read_csv(p) if p.exists() else None


df = tahminleri_yukle()
anomali = anomali_yukle()
T_MALIYET = maliyet_esigi()
T_TAKVIM = takvim_araligi()

st.title("Filo Bakım Karar Paneli")
st.caption("NASA C-MAPSS **simülasyon** verisi · 100 türbofan motor · tahmini kalan ömür (RUL) · "
           "bakım kararı: maliyet temelli · benzer geçmiş raporlar: FAA SDR")

# ---- YAN PANEL ----------------------------------------------------------------
st.sidebar.header("Ayarlar")
karar_esigi = st.sidebar.slider(
    "Bakıma alma eşiği (döngü)", 2, 40, T_MALIYET, 1,
    help=f"Tahmini RUL bu değerin altına inince bakıma al. Maliyet analizinin seçtiği değer: {T_MALIYET}. "
         "Bu eşik varsayılan maliyetlere (bakım 1, arıza 10) dayanır; gerçek eşiği bakım ekibi belirler.")
uyari_esigi = st.sidebar.slider(
    "Planlama uyarısı eşiği (döngü)", karar_esigi + 1, 80, max(30, karar_esigi + 1), 1,
    help="Parça ve iş gücü planlamak için erken uyarı. Bu eşik de insan kararıdır.")
takvim_N = st.sidebar.slider(
    "Sabit takvim aralığı (N döngü)", 60, 230, T_TAKVIM, 2,
    help="KARŞILAŞTIRMA TABANI. Klasik yöntemde her motor, yaşı N döngüye gelince bakıma alınır; "
         f"motorun durumuna bakılmaz. Maliyet analizinin seçtiği değer: {T_TAKVIM}. "
         "Kendi filonuzun bakım aralığını girip kazancı yeniden hesaplatabilirsiniz. "
         "Sadece 'Kazanç' sekmesini etkiler.")
st.sidebar.markdown("---")
mod = st.sidebar.radio("Filonun anı (simülasyon)",
                       ["Karışık filo (rastgele)", "Hepsi ömrünün %X'inde"])
if mod == "Karışık filo (rastgele)":
    tohum = st.sidebar.number_input("Rastgelelik tohumu", 0, 999, 7, 1)
else:
    yuzde = st.sidebar.slider("Ömrün yüzde kaçında", 10, 100, 85, 1)
st.sidebar.info("Panel bir karar **destek** aracıdır. Bakım kararını yetkili teknisyen verir.")

# ---- Filonun anlik goruntusu --------------------------------------------------
if mod == "Karışık filo (rastgele)":
    rng = np.random.default_rng(int(tohum))
    secili = [d.index[rng.integers(5, len(d))] for _, d in df.groupby("unit")]
else:
    secili = []
    for _, d in df.groupby("unit"):
        d = d.sort_values("cycle")
        secili.append(d.index[min(int(round(yuzde / 100 * len(d))) - 1, len(d) - 1)])
anlik = df.loc[secili].copy()


def durum(rul):
    if rul <= karar_esigi:
        return "BAKIMA AL"
    if rul <= uyari_esigi:
        return "PLANLA"
    return "İZLE"


anlik["Durum"] = anlik["tahmin"].apply(durum)
anlik = anlik.sort_values("tahmin")

sekme1, sekme_kazanc, sekme2, sekme3, sekme4 = st.tabs(
    ["Filo", "Kazanç", "Motor detayı", "Benzer raporlar", "Sonuçlar ve sınırlar"])

# ======================== SEKME 1: FILO ==========================================
with sekme1:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Toplam motor", len(anlik))
    k2.metric("Bakıma al", int((anlik["Durum"] == "BAKIMA AL").sum()))
    k3.metric("Planla", int((anlik["Durum"] == "PLANLA").sum()))
    k4.metric("Ortalama tahmini ömür", f"{anlik['tahmin'].mean():.0f} döngü")

    sol, sag = st.columns([2, 1])
    with sol:
        st.subheader("Filo sağlık görünümü")
        fig, ax = plt.subplots(figsize=(8, 3.2))
        renk = anlik["Durum"].map({"BAKIMA AL": KIRMIZI, "PLANLA": TURUNCU, "İZLE": LACIVERT})
        ax.bar(range(len(anlik)), anlik["tahmin"].clip(lower=0), color=renk)
        ax.axhline(karar_esigi, color=KIRMIZI, ls="--", lw=1)
        ax.axhline(uyari_esigi, color=TURUNCU, ls="--", lw=1)
        ax.set_xlabel("Motorlar (en riskliden en sağlama)")
        ax.set_ylabel("Tahmini RUL (döngü)")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Kırmızı çizgi: bakıma alma eşiği. Turuncu çizgi: planlama uyarısı.")
    with sag:
        st.subheader("Uyarı listesi")
        liste = anlik[anlik["Durum"] != "İZLE"]
        if liste.empty:
            st.success("Eşiklerin altında motor yok.")
        else:
            st.dataframe(
                liste[["unit", "cycle", "tahmin", "Durum", "RUL"]]
                .rename(columns={"unit": "Motor", "cycle": "Döngü", "tahmin": "Tahmini RUL",
                                 "RUL": "Gerçek RUL*"})
                .round({"Tahmini RUL": 0}),
                hide_index=True, height=300)
            st.caption("*Gerçek RUL yalnızca simülasyonda bilinir; gerçek filoda bu sütun olmaz.")

    # Karar kalitesi ozeti: bu eksiklik / fazlalik dengesini gostermek icin
    gec = int(((anlik["Durum"] == "İZLE") & (anlik["RUL"] <= karar_esigi)).sum())
    st.info(f"Bu filo anında, tahmin 'İZLE' derken gerçek ömrü eşiğin altında olan motor sayısı: **{gec}**. "
            "Bu sayı sıfırdan büyükse model o motorlar için geç uyarı vermiştir.")

# ======================== SEKME: KAZANC ==========================================
# Hocanin istegi: "sabit takvime gore kac motor kurtarildi, ne kadar maliyetten kacinildi".
# Iki kaydiriciya (N ve T) bagli olarak her iki politika BASTAN hesaplanir.
with sekme_kazanc:
    st.subheader("Sabit takvime göre kazanç")
    st.caption("Soldaki iki kaydırıcıya bağlıdır: **sabit takvim aralığı (N)** ve **bakıma alma eşiği (T)**. "
               "Her iki politika da 100 eğitim motorunun TAM ÖMRÜ üzerinde yeniden hesaplanır.")

    pol = politika_modulu()
    if isinstance(pol, Exception):
        st.warning("Politika hesabı hazır değil. Önce `python kod/04_modeller.py` çalıştır. "
                   f"(Ayrıntı: {type(pol).__name__}: {pol})")
    else:
        motorlar = motorlari_hazirla()
        n_motor = len(motorlar)
        ort_omur = float(np.mean([m[0] for m in motorlar]))
        # Bir donguluk omrun bedeli: bir planli bakim maliyetini ortalama omre bolduk (05_politika.py ile ayni)
        birim = ortak.MALIYET_PLANLI / ort_omur
        A = pol.politika_a(motorlar, takvim_N, birim)      # sabit takvim
        B = pol.politika_b(motorlar, karar_esigi, birim)   # tahmine dayali esik

        kacinilan = A["maliyet"] - B["maliyet"]
        yuzde = 100 * kacinilan / A["maliyet"] if A["maliyet"] else 0.0
        omur_motor = A["israf_motor_basi"] - B["israf_motor_basi"]
        onlenen = int(A["ariza"]) - int(B["ariza"])

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Toplam motor", n_motor)
        k2.metric("Önlenen arıza", onlenen,
                  help="Sabit takvimde arızalanıp eşik politikasında arızalanmayan motor sayısı. "
                       "Varsayılan N'de her iki politika da sıfır arıza verir; fark asıl olarak "
                       "boşa giden ömürde ortaya çıkar. Gerçek arıza farkı için alttaki dürüst ölçüme bakın.")
        k3.metric("Kaçınılan maliyet", f"{kacinilan:.1f} birim", f"%{yuzde:.0f}")
        k4.metric("Kurtarılan ömür", f"{omur_motor * n_motor:,.0f} döngü".replace(",", "."),
                  f"{omur_motor:.0f} döngü/motor")

        if kacinilan > 0:
            st.success(f"Bu ayarlarla eşik politikası, {n_motor} motorluk filoda **{kacinilan:.1f} birim** "
                       f"maliyetten kaçınıyor (yaklaşık {kacinilan:.0f} planlı bakım maliyetine denk) ve "
                       f"motor başına **{omur_motor:.0f} döngü** ömrü çöpe atmaktan kurtarıyor.")
        else:
            st.error("Bu ayarlarla sabit takvim daha ucuz. Eşiği düşürmeyi ya da N'i artırmayı deneyin.")

        # --- Maliyet nereden geliyor? ---
        def kalemler(r, ad):
            planli = n_motor - int(r["ariza"])
            israf_top = r["israf_motor_basi"] * n_motor
            return {"Politika": ad,
                    "Planlı bakım (adet)": planli,
                    "Arıza (adet)": int(r["ariza"]),
                    "Planlı bakım maliyeti": round(planli * ortak.MALIYET_PLANLI, 1),
                    "Arıza maliyeti": round(int(r["ariza"]) * ortak.MALIYET_ARIZA, 1),
                    "Boşa giden ömür (döngü)": round(israf_top),
                    "İsraf maliyeti": round(israf_top * birim, 1),
                    "TOPLAM maliyet": round(r["maliyet"], 1)}

        st.markdown("**Maliyet nereden geliyor?**")
        st.dataframe(pd.DataFrame([kalemler(A, f"A · Sabit takvim (N={takvim_N})"),
                                   kalemler(B, f"B · Eşik politikası (T={karar_esigi})")]),
                     hide_index=True)

        fig, ax = plt.subplots(figsize=(8, 2.4))
        adlar = [f"B · Eşik (T={karar_esigi})", f"A · Sabit takvim (N={takvim_N})"]
        planli_m = [(n_motor - int(r["ariza"])) * ortak.MALIYET_PLANLI for r in (B, A)]
        ariza_m = [int(r["ariza"]) * ortak.MALIYET_ARIZA for r in (B, A)]
        israf_m = [r["israf_motor_basi"] * n_motor * birim for r in (B, A)]
        ax.barh(adlar, planli_m, color=LACIVERT, label="Planlı bakım", height=0.55)
        ax.barh(adlar, ariza_m, left=planli_m, color=KIRMIZI, label="Arıza", height=0.55)
        ax.barh(adlar, israf_m, left=[p + a for p, a in zip(planli_m, ariza_m)],
                color=TURUNCU, label="Boşa giden ömür", height=0.55)
        ax.set_xlabel("Toplam maliyet (birim)")
        ax.legend(frameon=False, fontsize=9, ncol=3, loc="lower right")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

        # --- Durust olcum: hic gorulmemis motorlar ---
        st.markdown("---")
        st.markdown("**Hiç görülmemiş motorlarda (dürüst ölçüm)**")
        rl = csv_oku("rl_karsilastirma.csv")
        if rl is None:
            st.info("`ciktilar/rl_karsilastirma.csv` yok; `python kod/12_rl_karar.py` çalıştır.")
        else:
            oz = (rl[rl["politika"].isin(["A_sabit_takvim", "B_tahmine_dayali"])]
                  .groupby("politika")
                  .agg(motor_basina_maliyet=("maliyet_motor_basi", "mean"),
                       toplam_ariza=("ariza", "sum"),
                       israf_motor_basi=("israf_motor_basi", "mean"),
                       motor_denemesi=("test_motor", "sum"))
                  .round(3).reset_index())
            st.dataframe(oz, hide_index=True)
            a2 = oz[oz["politika"] == "A_sabit_takvim"]
            b2 = oz[oz["politika"] == "B_tahmine_dayali"]
            if len(a2) and len(b2):
                a2, b2 = a2.iloc[0], b2.iloc[0]
                d = a2["motor_basina_maliyet"] - b2["motor_basina_maliyet"]
                st.info(f"Eğitimde hiç görülmemiş {int(a2['motor_denemesi'])} motor denemesinde: "
                        f"sabit takvim **{int(a2['toplam_ariza'])} arıza**, eşik politikası "
                        f"**{int(b2['toplam_ariza'])} arıza**. Motor başına maliyet "
                        f"{a2['motor_basina_maliyet']:.2f} → {b2['motor_basina_maliyet']:.2f} "
                        f"(%{100 * d / a2['motor_basina_maliyet']:.0f} düşüş). "
                        "Parametreler motorların %70'inde seçilip kalan %30'unda ölçüldü, 20 kez tekrarlandı.")

        st.warning(
            "**Bu sayılar nasıl okunmalı?**\n\n"
            f"- **Birim bir para birimi değildir.** Planlı bakım 1, arıza {ortak.MALIYET_ARIZA:.0f}, "
            f"boşa giden her döngü 1/{ort_omur:.0f} birim kabul edilmiştir. Bu oranlar benim "
            "varsayımımdır; gerçek oranları bakım ekibi belirler.\n"
            "- **Üstteki tablo eğitim motorlarında hesaplanır**, yani iyimser taraftadır. "
            "Dürüst rakam alttaki tablodadır.\n"
            "- **N bir karşılaştırma tabanıdır.** Varsayılan değer maliyet analizinin seçtiği değerdir; "
            "gerçek bir filonun bakım aralığı farklı olabilir.\n"
            "- Veri bir **simülasyondur**; sonuçlar gerçek bir filoya doğrudan genellenemez.")


# ======================== SEKME 2: MOTOR DETAYI ==================================
with sekme2:
    motor_no = st.selectbox("Motor seç", sorted(df["unit"].unique()))
    d = df[df["unit"] == motor_no].sort_values("cycle")
    tetik = d[d["tahmin"] <= karar_esigi]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(d["cycle"], d["RUL_kirpik"], color=GRI, lw=1.6, label="Gerçek RUL (125'te kırpık)")
    ax.plot(d["cycle"], d["tahmin"], color=TURUNCU, lw=1.2, label="Modelin tahmini")
    ax.axhline(karar_esigi, color=KIRMIZI, ls="--", lw=1, label="Bakıma alma eşiği")
    if anomali is not None and motor_no in anomali.index and pd.notna(anomali.loc[motor_no, "baslangic_dongusu"]):
        ax.axvline(anomali.loc[motor_no, "baslangic_dongusu"], color=LACIVERT, lw=1.5,
                   label="Anomali başlangıcı")
    if len(tetik):
        ax.axvline(tetik["cycle"].iloc[0], color=KIRMIZI, lw=1.5, alpha=0.6, label="Bakım kararı anı")
    ax.set_xlabel("Döngü")
    ax.set_ylabel("Kalan ömür (döngü)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close(fig)

    omur = int(d["omur"].iloc[0])
    if len(tetik):
        c = int(tetik["cycle"].iloc[0])
        israf = int(d.loc[d["cycle"] == c, "RUL"].iloc[0])
        st.success(f"Bu eşikle motor **{c}. döngüde** bakıma alınırdı. Motorun gerçek ömrü {omur} döngü; "
                   f"bakım anında gerçekte **{israf} döngü** ömrü kalmıştı (boşa giden ömür).")
    else:
        st.error(f"Bu eşikle hiç uyarı gelmedi: motor {omur}. döngüde arızalandı.")
    if anomali is not None and motor_no in anomali.index and pd.notna(anomali.loc[motor_no, "o_anda_kalan_omur"]):
        st.caption(f"Anomali tespiti bozulmayı {int(anomali.loc[motor_no, 'baslangic_dongusu'])}. döngüde "
                   f"yakaladı (arızadan yaklaşık {int(anomali.loc[motor_no, 'o_anda_kalan_omur'])} döngü önce). "
                   "Bu, 11_anomali.py'deki '5 döngü üst üste eşik aşımı' tanımıdır (benim varsayımım).")

# ======================== SEKME 3: BENZER RAPORLAR ===============================
with sekme3:
    st.subheader("Benzer geçmiş arıza raporları (FAA SDR)")
    rag = rag_modulu()
    if isinstance(rag, Exception):
        st.warning("Rapor arama hazır değil. Önce `python kod/13a_sdr_hazirla.py`, "
                   "`python kod/13_metin_indeksi.py` komutlarını çalıştır. "
                   f"(Ayrıntı: {type(rag).__name__}: {rag})")
    else:
        c1, c2, c3 = st.columns(3)
        m_sec = c1.selectbox("Motor", sorted(anlik["unit"].unique()), key="rag_motor")
        satir = anlik[anlik["unit"] == m_sec].iloc[0]
        c2.metric("Tahmini RUL (filo anında)", f"{satir['tahmin']:.0f} döngü")
        sistem = c3.selectbox("İlgili sistem", ["compressor", "turbine", "fan", "bearing oil", "fuel", "bleed air"],
                              help="C-MAPSS FD001'de arıza modu tek: yüksek basınç kompresörü. "
                                   "Sistem bilgisini şimdilik elle seçiyorsun.")
        yontemler = ["tfidf"] + (["gomme"] if rag.VEKTORLER is not None else [])
        yontem = st.radio("Arama yöntemi", yontemler, horizontal=True,
                          help="'gomme' için sentence-transformers kurulu olmalı (13_metin_indeksi.py).")
        sorgu = rag.sorgu_uret(float(satir["tahmin"]), sistem)
        st.caption(f"Arama sorgusu (RUL'a göre otomatik): *{sorgu}*")
        for i, (_, r) in enumerate(rag.benzer_bul(sorgu, 3, yontem).iterrows(), 1):
            with st.expander(f"{i}. [{r['id']}] benzerlik {r['benzerlik']:.2f} · {r.get('motor_markasi', '')}", expanded=(i == 1)):
                st.write(r["temiz"])
                st.caption(f"Kaynak: {r.get('kaynak', 'bilinmiyor')}")
        st.warning("Bu raporlar gerçek uçaklardan gelmektedir ve bu motoru anlatmamaktadır. "
                   "Benzer arıza türünden geçmiş deneyimdir. Özet otomatik üretilmiştir; "
                   "bakım kararı yetkili teknisyen tarafından verilir.")
        if bool(rag.raporlar["sentetik"].iloc[0]):
            st.error("DİKKAT: Raporlar SENTETİK ÖRNEKTİR (kod testi). Gerçek veri değildir.")

# ======================== SEKME 4: SONUCLAR VE SINIRLAR ==========================
with sekme4:
    st.subheader("Model sonuçları (gerçek çıktı dosyalarından)")
    a, b = st.columns(2)
    t = csv_oku("model_karsilastirma.csv")
    if t is not None:
        a.markdown("**Motor bazlı 5 katlı çapraz doğrulama (RMSE, düşük iyi)**")
        a.dataframe(t, hide_index=True)
    t = csv_oku("test_sonuclari.csv")
    if t is not None:
        b.markdown("**Standart protokol: 100 test motoru, son döngü**")
        b.dataframe(t, hide_index=True)
    ls = [pd.read_csv(p) for p in sorted(ortak.CIKTILAR.glob("lstm_sonuclari*.csv"))]
    if ls:
        ort = pd.concat(ls).groupby("Model").mean(numeric_only=True).round(1).reset_index()
        st.markdown(f"**LSTM ({len(ls)} farklı rastgelelikle ortalama)**")
        st.dataframe(ort, hide_index=True)
    rl = csv_oku("rl_karsilastirma.csv")
    if rl is not None:
        ozet = (rl.groupby("politika")
                .agg(maliyet_ort=("maliyet_motor_basi", "mean"), maliyet_sapma=("maliyet_motor_basi", "std"),
                     ariza_toplam=("ariza", "sum"), israf_ort=("israf_motor_basi", "mean")).round(2).reset_index())
        st.markdown("**Bakım politikaları: hiç görülmemiş test motorlarında (20 denemenin ortalaması)**")
        st.dataframe(ozet, hide_index=True)
    sz = csv_oku("sizinti_deneyi.csv")
    if sz is not None:
        st.markdown("**Veri sızıntısı deneyi:** rastgele bölme modeli olduğundan iyi gösteriyor.")
        st.dataframe(sz, hide_index=True)

    st.subheader("Sınırlılıklar (insan dikkati gerektirenler)")
    st.markdown("""
- **Veri bir simülasyondur** ve sivil ticari motora aittir; savunma sistemleri için vekil örnektir. Gerçek filoya doğrudan genellenemez.
- **Maliyet oranları varsayımdır** (bakım 1, arıza 10, boşa giden ömür 1/206). Gerçek oranları bakım ekibi belirler; sonuçlar bunlara duyarlıdır.
- **Eşikler insan kararıdır.** Panelde kaydırıcılarla değiştirilebilir.
- **Tahminler kat dışı (out-of-fold)** tahminlerdir; LSTM sonuçları rastgeleliğe göre yaklaşık 2 döngü oynar.
- **Benzer raporlar başka uçaklardan gelir.** Sistem "bu rapor bu motoru anlatıyor" demez. Rapor özeti şablon tabanlıdır, dil modeli yoktur.
- **Anomali tanımı** (5 döngü üst üste eşik aşımı) benim varsayımımdır; veride bozulmanın gerçek başlangıcı yoktur.
""")
