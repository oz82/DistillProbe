# DistillProbe

> Bağlamsal sentence transformer'lardan damıtılmış statik embedding'lerin ne kaybettiğini ve ne koruduğunu ölçen modüler bir probing framework'ü.

DistillProbe, bag-of-words distill statik embedding'leri (Model2Vec ailesi, `minishlab/potion-multilingual-128M` — teacher'ı `BAAI/bge-m3`, Tokenlearn ile C4 üzerinde damıtılmış) beş eksende student üzerinde + bir teacher-internal idiomaticity probe'uyla sistematik olarak test eder. Karşılaştırma `sentence-transformers/LaBSE` (**reference contextual encoder**; potion'ın teacher'ı *değil*, yalnızca eski `M2V_multilingual_output` variantının teacher'ı) ve gerektiğinde `cc.tr.300.bin` (fastText subword baseline) ile yapılır. Türkçe odaklı; veri kaynakları ve örnekler Türkçe (PARSEME 1.2 TR, UD-Turkish-IMST, TDK GTS API, FLORES-200), framework dil-agnostik.

**Ana bulgu — distillation kayıp yelpazesi:**

| Eksen | Distill statik retention |
|---|---|
| Cross-lingual hizalanma | **~%66** (17 dil V3; tipoloji-bağlı: %94-97 yakın IE → %28-34 Türkmence outlier. Not: 8-dil yüksek-kaynak alt kümesi ~%83) |
| Polysemy sense-ranking agreement | **ρ≈0.34** (leak-free v3, perm p=0.001; random ~%15). Not: LaBSE-augmented v3_aug ρ=0.520 — augmentation leak'i, sadece üst-sınır olarak raporlanır |
| Morfolojik suffix tutarlılığı | kategoriye göre değişken (-OGlu 0.88 outlier; -DA 0.23) |
| Idiomaticity gap (teacher-içi spektrum) | doğru ve monoton (`gap(VID) > gap(LVC) > gap(LIT)`, p<10⁻⁹) |
| Pozisyona-özgü syntactic role | **~%0** (BoW yapısı gereği imkânsız) |
| Negasyon polarity flip | **~%0** ("iyi değil" hâlâ "iyi"'ye yakın) |

Detay yorum için: [`findings.md`](findings.md). Mühendislik kararları için: [`PLAN.md`](PLAN.md). Araştırma motivasyonu için: [`migration.md`](migration.md).

---

## Doküman çatısı

| Doküman | Soru |
|---|---|
| [`migration.md`](migration.md) | **Neden?** Araştırma motivasyonu, hipotezler, ön bulgular |
| [`PLAN.md`](PLAN.md) | **Nasıl?** Mimari, kararlar, faz tablosu, ilerleme kaydı |
| [`findings.md`](findings.md) | **Ne?** Sonuçlar, modül-bazlı bulgular, sentez |
| [`README.md`](README.md) | **Hızlı başlangıç** (bu doküman) |

> **Arşiv sürümü notu.** Yukarıdaki üç doküman ve dahili planlama/inceleme
> notları **geliştirme reposuna aittir; alıntılanabilir arşiv sürümüne (Zenodo)
> dahil edilmez.** Bunlar sürüm-içi çalışma günlükleridir ve yer yer makalede
> geri çekilmiş ara sonuçları (ör. yalnızca yüksek-kaynaklı 8 dilde ölçülen eski
> %83 cross-lingual retention sayısı; yayımlanan değer 17 dilde **%66**)
> bağlamıyla birlikte kaydeder. Arşivde bu dosyaları göremiyorsanız beklenen
> durum budur; makalenin her sayısı `configs/` + `paper/canonical_runs.md` +
> `scripts/rerun_canonical.sh` ile yeniden üretilebilir.

---

## Kurulum

```bash
pip install -e .                  # ana paket
pip install -e ".[teacher]"       # + sentence-transformers (LaBSE)
pip install -e ".[dev]"           # + pytest, ruff
```

Python ≥ 3.10. CUDA opsiyonel; LaBSE GPU'da ~10× hızlı (Faz 2 V2 / Faz 4 / Faz 5'te belirgin fark).

**fastText (yalnız morphology için):** Türkçe pretrained `cc.tr.300.bin` ~6.8 GB. `data/fasttext/cc.tr.300.bin` altına yerleştirilmeli.

**FLORES-200 (yalnız xlingual için):** Build script `scripts/build_xlingual_dataset.py` ilk çalıştırıldığında otomatik indirir (~70 MB).

---

## Hızlı başlangıç

```bash
# Mevcut modülleri listele
distillprobe list-modules

# Bir modülü tanıt
distillprobe describe morphology

# Bir veri setine sanity-check uygula
distillprobe sanity-check data/morphology/tr_categories.v1.json

# Bir probe çalıştır
distillprobe run --config configs/morph_v2.yaml
```

Her run otomatik olarak şunu üretir:
```
results/<module>/<UTC-timestamp>_<short-hash>/
├── run_manifest.json         # config + git rev + dep versions + seed
├── metrics.json              # tam sayısal sonuç
├── per_category.csv          # insan-okunaklı tablo
├── tokenization_audit.csv    # ilgili modüller için
└── plots/*.png               # bar charts, summary
```

Mevcut bir run dizini için plotları yeniden üretmek:

```bash
python scripts/replot.py results/morphology/<run_id>/
```

---

## Modüller

Tümü ✓ tamamlandı. Her satırda: temel soru, kullanılan teacher/baseline karşılaştırması, ana bulgu.

| Modül | Soru | Karşılaştırma | Bulgu |
|---|---|---|---|
| **morphology** | Suffix → fark vektörü tutarlılığı korunuyor mu? | M2V vs fastText (8 kategori, 215 çift) | `-OGlu` M2V 0.881 vs FT 0.196; pattern kategori-spesifik |
| **idiomaticity** (V2) | Compositionality gap teacher'da spektrum hâlinde mi? | LaBSE subword-pool vs input-embedding (PARSEME-tr, 11.219 phrase) | `gap(VID) > gap(LVC.full) > gap(LITERAL)`, p<10⁻⁹ |
| **polysemy** (V2) | Distill statik vektör hangi sense'e kayar? | M2V vs LaBSE (TDK API + PARSEME-augment, 19 kelime / 118 sense / ~400 örnek) | Winner agreement %47, Spearman ρ=0.37 [0.11, 0.60] |
| **syntax** | Pozisyona-özgü role bilgisi distill'de var mı? | LaBSE target-pool vs M2V sentence BoW (PARSEME UD, 107 lemma) | M2V'nin pozisyonel temsili %0 (BoW gereği); LaBSE 0.058 saf-positional sinyal silinir |
| **xlingual** | Cross-lingual hizalanma korunuyor mu? | M2V vs LaBSE (FLORES-200 dev, 17 dil, 136 çift) | LaBSE P@1=0.999 vs M2V P@1=0.660; retention ~%66; tipoloji-bağlı |
| **negation** | "X değil" antonyme yaklaşıyor mu? | M2V vs LaBSE (25 sıfat tripleti) | M2V 0/25 polarity flip; LaBSE 4/25; mean_diff +0.50, p=3×10⁻⁸ |

---

## 6 modülü baştan üretmek

Tüm runlar ~30 dakikada (CUDA ile) bitter. fastText (~6.8 GB RAM) sadece morphology'de yüklenir; LaBSE (~2 GB) sadece V2 idiom + polysemy + syntax + xlingual + negation'da.

```bash
# 1. Morphology — fastText gerekir
distillprobe run --config configs/morph_v2.yaml

# 2. Idiomaticity V1 (sentence-pool)
distillprobe run --config configs/idiom_v1.yaml

# 2b. Idiomaticity V2 (subword-level + input embedding)
python scripts/build_idiomaticity_dataset.py \
  data/idiomaticity/parseme_tr/{train,dev,test}.cupt \
  --out data/idiomaticity/tr_parseme.v2.json --with-literal --max-literal 3500
distillprobe run --config configs/idiom_v2.yaml

# 3. Polysemy V1 (TDK only)
python scripts/build_polysemy_dataset.py \
  --out data/polysemy/tr_polysemy.v1.json --min-examples 1 --min-senses 2
distillprobe run --config configs/polysemy_v1.yaml

# 3b. Polysemy V2 (+ PARSEME augmentation)
python scripts/augment_polysemy_with_parseme.py \
  --in data/polysemy/tr_polysemy.v1.json \
  --out data/polysemy/tr_polysemy.v2.json \
  --parseme data/idiomaticity/parseme_tr/{train,dev,test}.cupt \
  --top-k 3 --min-cos 0.30
distillprobe run --config configs/polysemy_v2.yaml

# 4. Syntax
python scripts/build_syntax_dataset.py \
  --parseme data/idiomaticity/parseme_tr/{train,dev,test}.cupt \
  --out data/syntax/tr_syntax.v1.json
distillprobe run --config configs/syntax_v1.yaml

# 5. Cross-lingual (FLORES-200 ilk çağrıda inilir)
python scripts/build_xlingual_dataset.py --out data/xlingual/tr_flores.v1.json
distillprobe run --config configs/xlingual_v1.yaml

# 6. Negation
distillprobe run --config configs/negation_v1.yaml
```

Modeller (Model2Vec, LaBSE) HuggingFace cache'inden ilk çağrıda otomatik inilir.

---

## Encoder'lar

| Encoder | Rol | Kaynak |
|---|---|---|
| **`Model2VecEncoder`** | Distill statik student (256-d, BoW lookup) | `minishlab/potion-multilingual-128M` |
| **`TransformerEncoder`** | Teacher / contextual reference | `sentence-transformers/LaBSE` (768-d, mean pooling, L2-norm) |
| **`FastTextEncoder`** | Static subword baseline (sadece morphology) | Facebook `cc.tr.300.bin` (300-d) |

Her `Encoder` aynı sözleşmeyi sağlar: `name`, `dim`, `encode(text) -> ndarray (L2-normalized)`, `tokenize(text) -> list[str]`. Yeni bir backend eklemek ABC'yi implement etmekten ibaret.

Faz 2 V2 ve Faz 4'te subword-level erişim için HuggingFace `AutoTokenizer` + `AutoModel` direkt kullanılır (encoder abstraction'ı yetmez); ilgili modüller modül-içi yüklemeyi kendisi yönetir.

---

## Proje yapısı

```
DistillProbe/
├── pyproject.toml
├── README.md, migration.md, PLAN.md, findings.md
├── distillprobe/
│   ├── encoders/        # base.py + 3 backend (model2vec, fasttext, transformer)
│   ├── metrics/         # consistency, similarity, baselines, stats (bootstrap, Wilcoxon, MannWhitney, Cliff δ)
│   ├── data/            # schemas (pydantic), loaders, parseme.py CUPT parser
│   ├── modules/         # 6 modül + IdiomaticityModuleV2 (toplam 7 registry entry)
│   ├── reporting/       # manifest, plots (modül başına özelleşmiş)
│   ├── config.py        # YAML config + encoder builder
│   └── cli.py           # `distillprobe run / sanity-check / list-modules / describe`
├── data/
│   ├── morphology/                      # tr_categories.v1.json
│   ├── idiomaticity/parseme_tr/         # PARSEME 1.2 TR train/dev/test.cupt
│   ├── idiomaticity/                    # tr_parseme.v1.json, .v2.json
│   ├── polysemy/                        # tr_polysemy.v1.json, .v2.json
│   ├── syntax/                          # tr_syntax.v1.json
│   ├── xlingual/                        # tr_flores.v1.json + flores200/ extracted
│   ├── negation/                        # tr_negation.v1.json
│   └── fasttext/cc.tr.300.bin           # gitignore (6.8 GB)
├── configs/                             # her modül için YAML run config
├── results/                             # gitignore — run outputs
├── tests/                               # 29 test
└── scripts/                             # build_*, replot, smoke_test
```

---

## Test ve smoke

```bash
pytest -q                         # 29 test, hepsi yeşil; ~0.3 saniye, model yüklemez
python scripts/smoke_test.py      # Model2Vec gerçek inference; opsiyonel fastText (~6.8 GB RAM)
```

Encoder kontrat testleri stub-based (gerçek model yüklemez); modül smoke testleri stub encoder + mini dataset üzerinden.

---

## Reproducibility detayları

- Random seed sabit (`42`); her bootstrap, sampling, augmentation deterministic.
- Bootstrap CI 10.000 resample, %95 percentile.
- Encoder L2-norm zorunlu (ABC'de test edilir).
- Run manifest: timestamp UTC, hostname, platform, Python sürümü, kurulu paket versiyonları, git rev, config dump, seed.
- Dataset versioning: `tr_X.v1.json` → değişiklikte `v2`, eski silinmez. v1 ve v2 farklı `data_path`'larda yan yana yaşar.

---

## Bilinmesi gereken sınırlar

- **Single-token sentence noise**: Sentence transformer'lar 1-2 token "cümleleri" kötü pool eder; idiom V1, syntax V1, negation V1 bu artefakttan etkilenir. Idiom V2 subword-level erişimle bunu temizledi; negation V2 (cümle bağlamında test) sonraya bırakıldı.
- **Cross-space comparison çıkmazı**: M2V (256d), LaBSE (768d), fastText (300d) farklı uzaylarda; doğrudan cosine yapılamaz. Modüller within-encoder ranking + cross-encoder agreement protokolü kullanır.
- **Negasyon veri seti küçük**: 25 el-yapımı triplet; Türkçe için hazır negation dataset yok. V2'de Türkçe-bilen review + verbal negation testi adayı.
- **PARSEME-tr lemma kalitesi değişken**: Bazı LVC.full lemma'ları bozuk ("zulü etmeyiniz", "öde yapıl"). Yüzey formları sağlam; modüller phrase/text üzerinden çalıştığı için sorun değil, ama lemma-bazlı analiz dikkat ister.

Tüm sınırlar ve V2 adayları [`findings.md`](findings.md) §4 ve §5'te işaretli.

---

## Bağımlılıklar

```toml
python = ">=3.10"
model2vec = "0.8.1"
gensim = ">=4.3,<5"
sentence-transformers = ">=2.7,<3"      # extras: teacher
numpy = ">=1.24,<3"
pandas = ">=2.0"
pydantic = ">=2.0"
pyyaml = ">=6"
matplotlib = ">=3.7"
scipy = ">=1.10"
# dev: pytest, ruff
```

---

## Lisans

MIT (kod). Veri setlerinin kendi lisansları:
- PARSEME 1.2 TR: CC BY-SA 4.0 (https://gitlab.com/parseme/sharedtask-data)
- FLORES-200: CC BY-SA 4.0 (Meta NLLB)
- TDK GTS API: TDK kamu API'si (kullanım şartları sozluk.gov.tr)
- KeNet: GPL-3.0 (StarlangSoftware/TurkishWordNet — referans için, doğrudan dağıtılmıyor)
