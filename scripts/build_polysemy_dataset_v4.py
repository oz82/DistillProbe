"""Build the Turkish polysemy v4 dataset with the same selection rule as English v2.

Usage:
    python scripts/build_polysemy_dataset_v4.py \
        --freq-conllu data/idiomaticity/parseme_tr/{train,dev,test}.cupt \
        --cache-dir data/polysemy/tdk_cache \
        --out data/polysemy/tr_polysemy.v4.json

Selection mirrors build_polysemy_dataset_en.py rule for rule, with TDK GTS in
the role of WordNet (both supply a sense inventory with a gloss and example
sentences per sense):

  1. Vocabulary: NOUN lemmas occurring >= `--min-freq` times in the given UD-style
     corpus, alphabetic, length >= `--min-len` (English: UD-EWT).
  2. Senses: the noun senses of TDK entries whose headword is exactly the lemma
     (English: noun synsets listing the exact lemma), each with >= `--min-examples`
     example sentence(s). TDK tags a part of speech only where it changes within an
     entry, so a sense inherits the last tag seen; only `isim` senses are kept.
     Proper-name entries are skipped.
  3. Words with >= `--min-senses` such senses qualify.
  4. A seeded random sample of `--n-words` qualifying words, drawn by walking the
     vocabulary in a seeded random order and stopping at the n-th qualifying word
     (equivalent to sampling from the full qualifying pool, which English computes
     outright; querying TDK for every lemma is too slow).

Unlike v1-v3, there are no manual edits. TDK glosses that open with a deictic
("Bu ...") refer back to the preceding sense; when that sense was filtered out the
dangling sense is dropped as well (v3 rewrote such glosses by hand).

Every TDK response is cached under `--cache-dir` so the build is reproducible
without re-querying the dictionary. The cache holds TDK text and is never
released (gitignore + export-ignore), like the dataset itself.

`--manifest-out` records the result as a sense manifest (word -> kept TDK sense
IDs, in order; no TDK text), which is released. `--manifest` replays it: only the
recorded headwords are queried, and the build aborts if TDK no longer returns
exactly the recorded senses, so a rebuild either reproduces the published dataset
or fails loudly.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_polysemy_dataset import ANAPHORA_RE, API_BASE  # noqa: E402

from distillprobe.data.parseme import iter_many_cupt  # noqa: E402
from distillprobe.data.schemas import PolysemyDataset, PolysemySense, PolysemyWord  # noqa: E402

POS_FEATURE = "3"  # TDK `ozelliklerListe` type code for part of speech
NOUN = "isim"


def tr_lower(s: str) -> str:
    return s.replace("I", "ı").replace("İ", "i").lower()


def decirc(s: str) -> str:
    return s.translate(str.maketrans("âîûêôÂÎÛÊÔ", "aiueoAIUEO"))


def norm_gloss(t: str) -> str:
    import re

    t = (t or "").replace("►", "")
    t = re.sub(r"\s*\((?:I{1,3}|IV|V|VI{0,3})\)", "", t)
    return re.sub(r"\s{2,}", " ", t).strip()


def noun_vocabulary(paths: list[Path], min_freq: int, min_len: int) -> list[str]:
    freq: Counter[str] = Counter()
    for sent in iter_many_cupt(paths):
        for tok in sent.tokens:
            if not tok.is_range and tok.upos == "NOUN" and tok.lemma:
                freq[tr_lower(tok.lemma)] += 1
    return sorted(w for w, n in freq.items() if n >= min_freq and w.isalpha() and len(w) >= min_len)


def fetch_tdk(word: str, timeout: float = 15.0, retries: int = 2) -> list[dict] | None:
    """Return TDK records ([] if the word has no entry) or None on a network failure."""
    import urllib.parse
    import urllib.request

    url = f"{API_BASE}?ara={urllib.parse.quote(word)}"
    req = urllib.request.Request(url, headers={"User-Agent": "DistillProbe/0.1 (research)"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []
        except Exception as e:  # noqa: BLE001
            if attempt == retries:
                print(f"  [{word}] network failure, not cached: {e}", file=sys.stderr)
                return None
            time.sleep(1.0 + attempt)
    return None


def cached_madde(word: str, cache_dir: Path, sleep: float) -> list[dict] | None:
    path = cache_dir / f"{word}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    records = fetch_tdk(word)
    time.sleep(sleep)
    if records is not None:
        # Unknown words are cached as [] so a rebuild does not query them again.
        path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return records


def noun_senses(
    records: list[dict], word: str, min_examples: int
) -> tuple[list[PolysemySense], int]:
    """Return (kept senses, number of dangling senses dropped)."""
    kept: list[PolysemySense] = []
    n_dangling = 0
    for rec in records:
        if str(rec.get("ozel_mi")) == "1" or tr_lower(decirc(rec.get("madde") or "")) != decirc(
            word
        ):
            continue
        pos: str | None = None
        prev_kept = False
        for s in rec.get("anlamlarListe") or []:
            for feat in s.get("ozelliklerListe") or []:
                if str(feat.get("tur")) == POS_FEATURE:
                    pos = feat.get("tam_adi")
            definition = norm_gloss(s.get("anlam") or "")
            examples = [
                t
                for t in (norm_gloss(e.get("ornek") or "") for e in s.get("orneklerListe") or [])
                if t
            ]
            keep = pos == NOUN and bool(definition) and len(examples) >= min_examples
            if keep and ANAPHORA_RE.match(definition) and not prev_kept:
                keep = False
                n_dangling += 1
            if keep:
                kept.append(
                    PolysemySense(
                        sense_id=f"tdk:{s.get('anlam_id', 'unknown')}",
                        homograph_id=str(rec.get("madde_id"))
                        if rec.get("madde_id") is not None
                        else None,
                        definition=definition,
                        pos="noun",
                        examples=examples,
                    )
                )
            prev_kept = keep
    return kept, n_dangling


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--freq-conllu", nargs="+", help="required unless --manifest is given")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--n-words", type=int, default=100)
    p.add_argument("--min-senses", type=int, default=3)
    p.add_argument("--min-examples", type=int, default=1)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--min-freq", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=0.3, help="seconds between uncached API calls")
    p.add_argument("--version", default="v4-tdk-random-2026-11-11")
    p.add_argument("--manifest-out", help="write the released sense manifest here")
    p.add_argument("--manifest", help="replay a released sense manifest instead of sampling")
    args = p.parse_args()
    if not args.manifest and not args.freq_conllu:
        p.error("--freq-conllu is required unless --manifest is given")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.manifest:
        return replay_manifest(Path(args.manifest), cache_dir, Path(args.out), args)
    vocab = noun_vocabulary([Path(s) for s in args.freq_conllu], args.min_freq, args.min_len)
    print(f"{len(vocab)} NOUN lemmas with frequency >= {args.min_freq}", file=sys.stderr)

    # Seeded random order over the whole vocabulary, stopping at the n-th qualifying
    # word: a uniform random sample of the qualifying nouns, without querying the
    # dictionary for every lemma (TDK can take 10-20 s per request).
    order = vocab[:]
    random.Random(args.seed).shuffle(order)
    candidates: list[tuple[str, list[PolysemySense]]] = []
    seen: set[str] = set()
    n_examined = 0
    n_dangling = 0
    for word in order:
        if len(candidates) >= args.n_words:
            break
        if decirc(word) in seen:
            continue
        seen.add(decirc(word))
        records = None
        for attempt in range(4):
            records = cached_madde(word, cache_dir, args.sleep)
            if records is not None:
                break
            time.sleep(5.0 * (attempt + 1))
        if records is None:
            # Skipping would make the sample depend on network luck; stop instead.
            # Re-running resumes from the cache.
            raise SystemExit(f"[{word}] could not be downloaded; re-run to resume")
        n_examined += 1
        senses, dangling = noun_senses(records, word, args.min_examples)
        if len(senses) >= args.min_senses:
            candidates.append((word, senses))
            n_dangling += dangling
        if n_examined % 50 == 0:
            print(
                f"  {n_examined} examined, {len(candidates)}/{args.n_words} qualify",
                file=sys.stderr,
            )
    if len(candidates) < args.n_words:
        raise SystemExit(f"only {len(candidates)} qualifying nouns in the whole vocabulary")
    print(
        f"{n_examined} lemmas examined in seeded random order; {len(candidates)} qualify "
        f"({len(candidates) / n_examined:.1%}); {n_dangling} dangling-anaphora senses dropped",
        file=sys.stderr,
    )

    chosen = sorted(candidates, key=lambda t: t[0])
    kept = [PolysemyWord(word=w, senses=s) for w, s in chosen]
    for w in kept:
        print(f"  [{w.word}] {len(w.senses)} senses", file=sys.stderr)

    ds = PolysemyDataset(
        language="tr",
        version=args.version,
        source=(
            "TDK GTS API (https://sozluk.gov.tr/gts); noun senses of entries whose headword is "
            f"exactly the lemma, with >= {args.min_examples} example(s); words with >= "
            f"{args.min_senses} such senses, NOUN lemma frequency >= {args.min_freq} in "
            f"{len(args.freq_conllu)} PARSEME-tr file(s); the first {args.n_words} qualifying "
            f"nouns in a seeded random order of the vocabulary (seed={args.seed}; "
            f"{n_examined} lemmas examined)"
        ),
        words=kept,
    )
    write_dataset(ds, Path(args.out))
    if args.manifest_out:
        manifest = {
            "language": "tr",
            "version": args.version,
            "source": ds.source,
            "params": {
                k: getattr(args, k)
                for k in ("n_words", "min_senses", "min_examples", "min_len", "min_freq", "seed")
            },
            "n_examined": n_examined,
            "note": (
                "Regeneration recipe for the Turkish polysemy v4 set: word -> kept TDK sense "
                "IDs, in order, without TDK text. Replay with "
                "scripts/build_polysemy_dataset_v4.py --manifest."
            ),
            "words": [{"word": w.word, "sense_ids": [s.sense_id for s in w.senses]} for w in kept],
        }
        Path(args.manifest_out).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"wrote {args.manifest_out}", file=sys.stderr)
    return 0


def write_dataset(ds: PolysemyDataset, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(
        f"wrote {out}: {len(ds.words)} words / {sum(len(w.senses) for w in ds.words)} senses",
        file=sys.stderr,
    )


def replay_manifest(path: Path, cache_dir: Path, out: Path, args: argparse.Namespace) -> int:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    min_examples = manifest["params"]["min_examples"]
    kept: list[PolysemyWord] = []
    problems: list[str] = []
    for entry in manifest["words"]:
        word = entry["word"]
        records = cached_madde(word, cache_dir, args.sleep)
        if records is None:
            raise SystemExit(f"[{word}] could not be downloaded; re-run to resume")
        senses, _ = noun_senses(records, word, min_examples)
        got = [s.sense_id for s in senses]
        if got != entry["sense_ids"]:
            problems.append(f"  [{word}] recorded {entry['sense_ids']}, TDK now gives {got}")
            continue
        kept.append(PolysemyWord(word=word, senses=senses))
    if problems:
        print("MANIFEST DRIFT - the rebuild does not match the published dataset:", file=sys.stderr)
        for line in problems:
            print(line, file=sys.stderr)
        raise SystemExit("aborting: TDK's entries have changed since the manifest was recorded")
    ds = PolysemyDataset(
        language="tr", version=manifest["version"], source=manifest["source"], words=kept
    )
    write_dataset(ds, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
