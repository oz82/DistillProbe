"""Build a cross-lingual probe dataset from FLORES-200 dev split.

Downloads the FLORES-200 archive (~70 MB compressed) on first run, extracts the
per-language dev files for the requested codes, and writes a single JSON
dataset where each pair is a parallel sentence aligned by line index across
all configured languages.

Default language set: TR + 7 high/medium-resource languages spanning multiple
scripts and families.

Usage:
    python scripts/build_xlingual_dataset.py \\
        --out data/xlingual/tr_flores.v1.json \\
        --flores-dir data/xlingual/flores200
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

from distillprobe.data.schemas import XLingualDataset, XLingualPair

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

DEFAULT_LANGS = ["tr", "en", "de", "fr", "ar", "zh", "es", "ru"]

LANG_CODE_MAP = {
    "tr": "tur_Latn",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "fi": "fin_Latn",
    "hu": "hun_Latn",
    "el": "ell_Grek",
    # Turkic family — added for V2 typology coverage
    "kk": "kaz_Cyrl",  # Kazakh (Cyrillic; Kazakhstan still mostly uses Cyrillic in 2026)
    "ky": "kir_Cyrl",  # Kyrgyz
    "uz": "uzn_Latn",  # Uzbek (Northern, Latin)
    "az": "azj_Latn",  # Azerbaijani (Northern, Latin)
    "tk": "tuk_Latn",  # Turkmen
}


def ensure_flores(out_dir: Path) -> Path:
    """Download + extract FLORES-200 if absent. Return the dev/ directory."""
    extracted = out_dir / "flores200_dataset" / "dev"
    if extracted.exists() and any(extracted.iterdir()):
        return extracted
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / "flores200_dataset.tar.gz"
    if not archive.exists():
        print(f"downloading FLORES-200 → {archive}", file=sys.stderr)
        urllib.request.urlretrieve(FLORES_URL, archive)
    print(f"extracting {archive} → {out_dir}", file=sys.stderr)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(out_dir)
    return extracted


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--flores-dir", default="data/xlingual/flores200")
    p.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    p.add_argument("--n", type=int, default=None, help="cap number of sentences (default: all)")
    p.add_argument("--version", default="v1-2026-05-05")
    args = p.parse_args()

    for short in args.langs:
        if short not in LANG_CODE_MAP:
            print(f"unknown lang code: {short}", file=sys.stderr)
            return 2

    dev_dir = ensure_flores(Path(args.flores_dir))
    print(f"using {dev_dir}", file=sys.stderr)

    lang_lines: dict[str, list[str]] = {}
    for short in args.langs:
        long = LANG_CODE_MAP[short]
        f = dev_dir / f"{long}.dev"
        if not f.exists():
            print(f"missing: {f}", file=sys.stderr)
            return 2
        lang_lines[short] = f.read_text(encoding="utf-8").splitlines()

    counts = {l: len(v) for l, v in lang_lines.items()}
    if len(set(counts.values())) != 1:
        print(f"line count mismatch: {counts}", file=sys.stderr)
        return 2
    N = next(iter(counts.values()))
    if args.n and args.n < N:
        N = args.n

    pairs: list[XLingualPair] = []
    for i in range(N):
        pairs.append(
            XLingualPair(
                pair_id=str(i),
                text_by_lang={lang: lang_lines[lang][i] for lang in args.langs},
            )
        )

    ds = XLingualDataset(
        version=args.version,
        source=f"FLORES-200 dev split (n={N}, langs={args.langs})",
        languages=list(args.langs),
        pairs=pairs,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out} ({N} pairs, {len(args.langs)} langs)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
