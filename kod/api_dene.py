"""
api_dene.py — Calisan API'ye ornek bir istek gonderir ve cevabi okunur sekilde yazar.

NE YAPAR: veri/test_FD001.txt'den bir test motorunun gecmis olcumlerini alir, API'nin /tahmin
adresine gonderir, tahmini kalan omru (RUL), durumu ve uyarilari ekrana basar. Ayrica gercek
kalan omru (veri/RUL_FD001.txt) ile karsilastirir.
NEDEN: /docs sayfasinda 21 sensoru elle girmek zor; bu dosya istegi senin yerine hazirlar.
ONCE: API calismali (baslat_api.bat cift tikla ya da: uvicorn kod.api:app --port 8000).
KULLANIM (ikinci bir terminalde, PROJE klasorunde):
    python kod/api_dene.py            -> motor 1
    python kod/api_dene.py 25         -> motor 25
    python kod/api_dene.py 25 8000    -> motor 25, port 8000
Sadece standart Python kutuphaneleri kullanir; ek kurulum gerekmez.
"""
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
motor = int(sys.argv[1]) if len(sys.argv) > 1 else 1
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000

# Ham dosya: bosluklarla ayrilmis sayilar. Sutunlar: motor, dongu, 3 ayar, 21 sensor.
adlar = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]
satirlar = []
for s in (KOK / "veri" / "test_FD001.txt").read_text().splitlines():
    d = s.split()
    if d and int(float(d[0])) == motor:
        satirlar.append(dict(zip(adlar, [float(x) for x in d])))
if not satirlar:
    sys.exit(f"Motor {motor} test dosyasinda yok (1-100 arasi sec).")
for k in satirlar:
    k["cycle"] = int(k["cycle"])
    k.pop("unit")

gercek = int(float((KOK / "veri" / "RUL_FD001.txt").read_text().split()[motor - 1]))
istek = {"motor_id": f"test-motor-{motor}", "olcumler": satirlar,
         "karar_esigi": 7, "uyari_esigi": 30, "rapor_sayisi": 2}

try:
    r = urllib.request.Request(f"http://127.0.0.1:{port}/tahmin", data=json.dumps(istek).encode(),
                               headers={"Content-Type": "application/json"})
    cevap = json.load(urllib.request.urlopen(r, timeout=60))
except urllib.error.URLError as e:
    sys.exit(f"API'ye ulasilamadi ({e}). Once API'yi baslat (baslat_api.bat).")

print(f"Motor: {cevap['motor_id']}  (gonderilen gecmis: {len(satirlar)} dongu)")
print(f"Tahmini kalan omur : {cevap['tahmini_rul']} dongu")
print(f"Gercek kalan omur  : {gercek} dongu  (not: modelin ust siniri 125)")
print(f"Durum              : {cevap['durum']}  (esikler: {cevap['esikler']})")
print("Uyarilar:")
for u in cevap["uyarilar"]:
    print("  -", u)
for rp in cevap["benzer_raporlar"]:
    print(f"Benzer rapor {rp['rapor_id']} (benzerlik {rp['benzerlik']}): {rp['metin'][:120]}...")
