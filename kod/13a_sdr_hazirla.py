"""
13a_sdr_hazirla.py — FAA SDR dosyalarindan MOTOR ile ilgili ariza raporlarini ayiklar.

SDR = Service Difficulty Report. Havayollari ve bakim istasyonlari, ucaklarda
gordukleri ariza/bozukluklari FAA'ya bildirir. Her satir bir bildirim; serbest metin
'Discrepancy' sutunundadir (bildirimin anlatimi).

NE YAPAR:
  1) veri/sdr/ klasorundeki SDR-*.csv dosyalarini okur (FAA sitesinden indirdigin yillik dosyalar)
  2) Sadece MOTOR sistemleriyle ilgili satirlari tutar: JASC kodu 72-79 ile baslayanlar
       72 motor, 73 yakit ve kontrol, 74 atesleme, 75 hava sistemi (sizdirma),
       76 motor kontrolleri, 77 gostergeler, 78 egzoz, 79 yag
  3) Cok kisa anlatimlari atar (en az 20 kelime)
  4) KISISEL / TANIMLAYICI bilgileri maskeler: e-posta, telefon, uçak tescil numarasi (N-numarasi),
     seri numarasi, is emri numarasi. Operator kodunu ve kontrol numarasini cikti dosyasina KOYMAZ.
  5) veri/raporlar.csv dosyasini yazar (13_metin_indeksi.py bunu okur)

NEDEN MASKELEME: Veri kamuya aciktir, ama metinde bazen kisi ve ucak tanimlayan ayrintilar var.
Projemin ihtiyaci "ne tur ariza, nasil cozuldu"; kimlik gerekmiyor. Gereksiz bilgiyi tutmam.

KULLANIM SARTI: FAA sayfasinda acik bir kullanim sarti yazmiyordu. Veriyi YENIDEN DAGITMA,
raporda kaynagi (FAA SDR, yil araligi) belirt. Emin degilsen FAA'ya sor.

ONCE: FAA sayfasindan SDR-2023.csv, SDR-2024.csv, SDR-2025.csv'yi indirip veri/sdr/ klasorune koy.
CIKTI: veri/raporlar.csv
"""

import re
import pandas as pd
import ortak

SDR_KLASOR = ortak.VERI / "sdr"
MOTOR_JASC = ("72", "73", "74", "75", "76", "77", "78", "79")
EN_AZ_KELIME = 20

# Maskeleme kurallari: (desen, yerine yazilacak)
MASKELER = [
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"),                                  # e-posta
    (r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b", "[TEL]"),                            # telefon 555-123-4567
    # Ucak tescil numarasi: N + en az 3 rakam ya da rakam+harf (N123AB). "N1", "N2" (fan devri) etkilenmez.
    (r"\bN(?:\d{3,5}[A-Z]{0,2}|\d{1,4}[A-Z]{1,2})\b", "[TESCIL]"),
    (r"\b(?:ESN|S/N|SN)\s*[:#]?\s*[A-Z0-9-]{4,}\b", "[SERI]"),                # seri numarasi
    (r"\b(?:W/O|WO)\s*[:#]?\s*\d{4,}\b", "W/O [NO]"),                         # is emri
    (r"\b\d{7,}\b", "[NO]"),                                                  # diger uzun rakam dizileri
]


def maskele(metin):
    for desen, yerine in MASKELER:
        metin = re.sub(desen, yerine, metin)
    return metin


def dosyalari_oku():
    dosyalar = sorted(SDR_KLASOR.glob("SDR-*.csv"))
    if not dosyalar:
        raise SystemExit(f"HATA: {SDR_KLASOR} icinde SDR-*.csv yok. FAA sayfasindan indirip oraya koy.")
    parcalar = []
    for f in dosyalar:
        yil = int(re.search(r"(\d{4})", f.name).group(1))
        d = pd.read_csv(f, low_memory=False)
        print(f"  {f.name}: {len(d):,} satir")
        parcalar.append(d.assign(yil=yil))
    return pd.concat(parcalar, ignore_index=True)


if __name__ == "__main__":
    print("SDR dosyalari okunuyor...")
    d = dosyalari_oku()
    print(f"Toplam bildirim: {len(d):,}")

    # Motor sistemleri: JASC kodunun ilk iki hanesi 72-79
    jasc = d["JASCCode"].astype(str).str.strip()
    motor = d[jasc.str.startswith(MOTOR_JASC)].copy()
    print(f"Motor ile ilgili (JASC 72-79): {len(motor):,}")

    # Anlatimi olmayan ya da cok kisa olanlari at
    motor["Discrepancy"] = motor["Discrepancy"].fillna("").astype(str)
    motor = motor[motor["Discrepancy"].str.split().str.len() >= EN_AZ_KELIME]
    print(f"En az {EN_AZ_KELIME} kelimelik anlatimi olan: {len(motor):,}")

    # Birebir ayni anlatimlari (tekrarlar) at: ayni metin iki kez arama sonucunu bozar
    once = len(motor)
    motor = motor.drop_duplicates(subset="Discrepancy")
    print(f"Birebir tekrar eden {once - len(motor)} anlatim atildi")

    # Yakin tekrarlar: ayni operator ayni sablonu yuzlerce kez gonderebiliyor, sadece sonu degisiyor.
    # Ilk 120 karakteri (buyuk/kucuk harf ve bosluk farki yok sayilarak) ayni olanlardan birini tutariz.
    once = len(motor)
    anahtar = motor["Discrepancy"].str.lower().str.replace(r"\s+", " ", regex=True).str[:120]
    motor = motor[~anahtar.duplicated()]
    print(f"Yakin tekrar eden {once - len(motor)} anlatim atildi")

    cikti = pd.DataFrame({
        # Gercek kontrol numarasi yerine kendi numaramiz: operatoru ele vermesin
        "id": [f"SDR{y}-{i:06d}" for y, i in zip(motor["yil"], motor.index)],
        "yil": motor["yil"].to_numpy(),
        "jasc": motor["JASCCode"].astype(str).str.strip().to_numpy(),
        "motor_markasi": motor["EngineMake"].fillna("").to_numpy(),
        "motor_modeli": motor["EngineModel"].fillna("").to_numpy(),
        "parca": motor["PartName"].fillna("").to_numpy(),
        "metin": motor["Discrepancy"].apply(maskele).str.replace(r"\s+", " ", regex=True).str.strip().to_numpy(),
        "kaynak": "FAA SDR " + motor["yil"].astype(str).to_numpy(),
    })
    cikti.to_csv(ortak.VERI / "raporlar.csv", index=False)

    print(f"\nKaydedildi -> veri/raporlar.csv  ({len(cikti):,} rapor)")
    print("Yillara gore:", cikti["yil"].value_counts().sort_index().to_dict())
    kelime = cikti["metin"].str.split().str.len()
    print(f"Anlatim uzunlugu (kelime): ortanca {kelime.median():.0f}, ortalama {kelime.mean():.0f}")
    print("En sik motor markalari:", cikti["motor_markasi"].replace("", "bos").value_counts().head(5).to_dict())
    print("\nORNEK (ilk 2 rapor):")
    for _, r in cikti.head(2).iterrows():
        print(f"  [{r['id']}] {r['metin'][:200]}...")
