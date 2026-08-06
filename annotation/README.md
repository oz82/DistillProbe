# Polysemy human annotation (P1-5)

Bu klasör, polysemy probe'una **insan altın-standardı + insan tavanı** eklemek için
gereken anotasyon materyalini içerir (revizyon item P1-5).

## Dosyalar
- **`polysemy_annotation_GUIDE.md`** — işaretleyici kılavuzu. **Önce bunu okuyun.**
- **`polysemy_form_annotator_A.xlsx` … `…_E.xlsx`** — boş şablon formlar. PII içermez, **ama *Form* sekmesi 346 TDK tanımını ve örnek cümlesini birebir gömer** → telif nedeniyle **release arşivine girmez** (`.gitattributes` export-ignore). Dev repo'da yerel üretim için durur; arşivi alan `scripts/build_polysemy_form.py` ile sense manifest'inden yeniden üretir.
- **`polysemy_ratings_anon.csv`** — anonim (R1–R4) 0–3 derecelendirmeler; yayımlanan veri.
- **`ethics/etik_kurul_raporu.docx`** — etik kurul başvuru/bilgilendirme raporu (Türkçe). **Yalnız dev repo**: doldurulmamış bir başvuru formudur, arşive girmez (makale kurul incelemesinin aranmadığını/gerekmediğini beyan ediyor; boş bir başvuru formu bununla çelişir görünür).
- **`responses/`** — ham dolu formlar + `_coder_map.txt` (PII; **gitignored**, release edilmez).
- İlgili: `data/polysemy/tr_polysemy.v3.sense_manifest.json` (+`.sha256`) — TDK datasetinin verbatim metin olmadan yeniden üretim reçetesi.

Excel formu (.xlsx; Excel, Google Sheets, LibreOffice, Numbers — hepsinde açılır)
iki sekmeli: *Yönerge* (kısa talimat + **işaretleyici bilgileri & onay** alanları) ve
*Form*. `PUAN_0_3` sütununda **0–3 açılır menü** (yanlış değer girilemez), başlık satırı
sabit, puan sütunu sarı ile vurgulu, kelimeler dönüşümlü gölgeli. Formlar
`data/polysemy/tr_polysemy.v3.json` (leak-free primary, 50 kelime / 346 anlam)
içeriğinden üretildi.

## Akış
1. **Türkçe anadil konuşucuları** (hedef 5; toplanan 4 bağımsız kodlayıcı), kılavuzu
   okuyup Yönerge'deki bilgi+onay alanlarını doldurur, sonra **bağımsız** olarak kendi
   Excel formunu doldurur (`PUAN_0_3`: her anlama 0–3).
2. Doldurulmuş ham Excel'ler **`annotation/responses/`** altına konur. Bu klasör
   **gitignore'lu**: formların Yönerge sayfası işaretleyici **ad/yaş (PII)** içerir,
   commit/release edilmez.
3. **Anonimleştirme:** ham formlardan PII'siz `annotation/polysemy_ratings_anon.csv`
   üretilir. Sütunlar **`R1..R4`** (ad/baş-harf yok); anahtar `kelime_no, anlam_no`.
   **TDK metni (`tanim`) CSV'de YOK** — telif için dağıtılmaz; gerekirse build
   script/manifest ile join edilir. Commit/release edilen bilimsel artifact budur.
   Ad↔R-kodu eşlemesi yalnızca yerel `responses/_coder_map.txt`'de (gitignored,
   submit edilmez).
4. Skorlama: **`python scripts/score_polysemy_annotation.py --encoder-metrics
   results/polysemy/<primary_run>/metrics.json`**. Varsayılan girdi = anonim CSV
   (PII'siz, repo'dan tam reprodüktif); ham xlsx için `--forms-glob annotation/responses/...`.
   Hesaplar: İnsan-tavanı = işaretleyiciler arası uyum (**Krippendorff α**, ordinal);
   gold = anlam başına medyan/ortalama; her encoder'ın (M2V, LaBSE) sense-sıralamasının
   gold'a uyumu (Spearman ρ + winner-agreement, bootstrap CI). Boş/kısmi formlarda
   fill-status raporlar. (Gerekirse: `pip install -e ".[annotation]"`.)

**Sonuç (2026-06-30, 4 kodlayıcı):** α=0.477 (insan tavanı); gold-vs-encoder LaBSE
ρ=0.10, M2V ρ=0.06 — ikisi de ~0 → statik embedding insan salience'ını izlemiyor.
Detay: paper §4.3 "Human ceiling" paragrafı.

## Önemli
- İşaretleyiciler **birbirinden bağımsız** çalışmalı (uyum ölçümü için şart).
- İsteğe bağlı **kalibrasyon turu:** herkes önce aynı 5 kelimeyi yapıp *ölçeği* (0–3'ün
  anlamını) bir kez hizalasın, sonra gerisini bağımsız doldursun.
- Formlar yeniden üretilebilir: `python scripts/build_polysemy_form.py -n 5`.
