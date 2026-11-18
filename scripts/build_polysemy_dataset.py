"""Build a Turkish polysemy probe dataset from the TDK GTS public API.

Usage:
    python scripts/build_polysemy_dataset.py --out data/polysemy/tr_polysemy.v1.json

For each candidate word we hit `https://sozluk.gov.tr/gts?ara=<word>` (the API
backing TDK's online dictionary), parse the returned `madde` records, and keep
the senses that have at least `--min-examples` example sentences. Words with
fewer than `--min-senses` qualifying senses are dropped.

Rate-limit policy: sequential requests with a small inter-call sleep — TDK
blocks IPs that hit them with >16 parallel queries (per documentation).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import urllib.request
import urllib.parse

from distillprobe.data.schemas import PolysemyDataset, PolysemySense, PolysemyWord

# Turkish-speaker draft of polysemic candidates, mixed POS, mixed concrete/abstract,
# mixed homograph/genuine-polysemy. Will be filtered by TDK availability.
CANDIDATES: list[str] = [
    "yüz",   # face / 100 / swim
    "çay",   # tea / stream
    "ay",    # moon / month
    "kazak", # sweater / Kazakh
    "kurt",  # wolf / worm
    "dil",   # tongue / language
    "baş",   # head / chief / beginning
    "göz",   # eye / drawer / spring source
    "el",    # hand / others / strange
    "kaba",  # rude / coarse
    "taş",   # stone / overflow (verb)
    "sap",   # handle / stalk / get-confused
    "sıra",  # row / turn / desk
    "gül",   # rose / laugh (verb)
    "sol",   # left / pale (adj)
    "yer",   # place / earth / floor
    "yol",   # road / way / method
    "gün",   # day / sun
    "ot",    # grass / weed
    "kara",  # black / land
    "yaş",   # age / wet
    "aç",    # hungry / open (verb)
    "sağ",   # right / alive / robust
    "bin",   # thousand / mount (verb)
    "kaz",   # goose / dig (verb)
]

API_BASE = "https://sozluk.gov.tr/gts"


def fetch_madde(word: str, timeout: float = 15.0, retries: int = 2) -> list[dict] | None:
    """Hit TDK GTS, return list of madde records or None on failure."""
    url = f"{API_BASE}?ara={urllib.parse.quote(word)}"
    req = urllib.request.Request(url, headers={"User-Agent": "DistillProbe/0.1 (research)"})
    body: str | None = None
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read().decode("utf-8")
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(1.0 + attempt)
                continue
    if body is None:
        print(f"  [{word}] HTTP error after {retries + 1} attempts: {last_err}", file=sys.stderr)
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print(f"  [{word}] non-JSON response (probably not in dict)", file=sys.stderr)
        return None
    if isinstance(data, dict) and "error" in data:
        print(f"  [{word}] api error: {data.get('error')}", file=sys.stderr)
        return None
    if not isinstance(data, list):
        print(f"  [{word}] unexpected top-level type: {type(data).__name__}", file=sys.stderr)
        return None
    return data


def extract_senses(records: list[dict], word: str) -> list[PolysemySense]:
    senses: list[PolysemySense] = []
    for rec in records:
        homograph = rec.get("madde_id")
        pos = rec.get("ozelliklerListe", [{}])[0].get("tam_adi") if isinstance(rec.get("ozelliklerListe"), list) else None
        # Iterate senses in this madde
        sense_list = rec.get("anlamlarListe") or []
        import re as _re

        def _norm(t: str) -> str:
            # Strip TDK metadata that leaks into glosses: the '►' cross-reference
            # marker and the (I)/(II)/(III) homograph tags; collapse whitespace.
            t = (t or "").replace("►", "")
            t = _re.sub(r"\s*\((?:I{1,3}|IV|V|VI{0,3})\)", "", t)
            return _re.sub(r"\s{2,}", " ", t).strip()

        for s in sense_list:
            sense_id = f"tdk:{s.get('anlam_id', 'unknown')}"
            definition = _norm(s.get("anlam") or "")
            examples_raw = s.get("orneklerListe") or []
            examples: list[str] = []
            for ex in examples_raw:
                txt = _norm(ex.get("ornek") or "")
                if txt:
                    examples.append(txt)
            sense_pos = s.get("ozellikleriListe", [{}])
            if isinstance(sense_pos, list) and sense_pos:
                # 'ozellikleriListe' (typo or different spelling) sometimes used
                pass
            senses.append(
                PolysemySense(
                    sense_id=sense_id,
                    homograph_id=str(homograph) if homograph is not None else None,
                    definition=definition,
                    pos=pos,
                    examples=examples,
                )
            )
    return senses


def filter_word(senses: list[PolysemySense], min_examples: int, min_senses: int) -> list[PolysemySense] | None:
    """Keep only senses with enough examples; drop the word if too few qualify."""
    qualifying = [s for s in senses if len(s.examples) >= min_examples]
    if len(qualifying) < min_senses:
        return None
    return qualifying


# TDK GTS glosses use intra-entry forward anaphora: a sense beginning with a
# deictic ("Bu/Bunlar/Böyle/Şu/Söz konusu + X") refers back to the head noun of
# the IMMEDIATELY PRECEDING sense. When filter_word drops that preceding sense
# (too few examples), the reference is left dangling — the kept gloss encodes a
# meaningless prototype and misleads one-shot annotators. This detector flags
# exactly that case: a kept anaphor-initial sense whose original predecessor was
# dropped. (A kept predecessor → reference resolves → not flagged.)
ANAPHORA_RE = re.compile(r"^(bu|bunlar|bunun|bunları|bunların|böyle|böylesi|şu|söz konusu)\b", re.I)


def detect_dangling_anaphora(
    full_senses: list[PolysemySense], kept_senses: list[PolysemySense]
) -> list[tuple[PolysemySense, PolysemySense]]:
    """Return (kept_dangling_sense, dropped_antecedent) pairs.

    A kept sense is dangling iff its definition opens with a deictic cue AND the
    sense immediately preceding it in the *unfiltered* list was dropped.
    """
    kept_ids = {s.sense_id for s in kept_senses}
    dangling: list[tuple[PolysemySense, PolysemySense]] = []
    for idx, s in enumerate(full_senses):
        if idx == 0 or s.sense_id not in kept_ids:
            continue
        if not ANAPHORA_RE.match(s.definition or ""):
            continue
        prev = full_senses[idx - 1]
        if prev.sense_id not in kept_ids:
            dangling.append((s, prev))
    return dangling


def _apply_manifest(
    built: list[PolysemyWord],
    manifest_path: Path,
    fetched: dict[str, list[PolysemySense]],
) -> list[PolysemyWord]:
    """Reconcile a freshly fetched build against a released sense manifest.

    The manifest records exactly which TDK sense IDs the published dataset kept,
    in order, plus the manual edits applied to them. Edits are recorded as rules
    rather than as dictionary text: `definition_from` + `replace` substitutes the
    antecedent sense's definition for a deictic phrase in the edited gloss (for a
    dangling "Bu ..." reference), and `field` + `replace` = [old, new] fixes a typo
    in that field's text. `new_definition` (literal text) is still accepted.
    `fetched` holds every sense TDK returned per word, before filtering, so a
    `definition_from` antecedent can be a sense the filter dropped. Replaying it turns the
    build into a deterministic regeneration of the published dataset rather than
    a fresh crawl whose contents drift with the dictionary. This is what makes
    the "cited, not redistributed" release honest: the recipe reproduces the
    dataset without shipping TDK's text.

    Fails loudly on any drift — a silently different rebuild would be worse than
    no rebuild at all.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_word = {w.word: w for w in built}
    edits: dict[tuple[str, str], list[dict]] = {}
    for e in manifest.get("manual_edits", []):
        edits.setdefault((e["word"], e["sense_id"]), []).append(e)

    def apply_edit(word: str, sense: PolysemySense, e: dict) -> PolysemySense:
        if "new_definition" in e:
            return sense.model_copy(update={"definition": e["new_definition"]})
        if "definition_from" in e:
            source = {x.sense_id: x for x in fetched.get(word, [])}.get(e["definition_from"])
            if source is None or e["replace"] not in sense.definition:
                raise SystemExit(f"[{word}] manual edit {e} no longer applies to TDK's text")
            definition = sense.definition.replace(e["replace"], source.definition, 1)
            return sense.model_copy(update={"definition": definition})
        old, new = e["replace"]
        field = e["field"]
        value = getattr(sense, field)
        texts = value if isinstance(value, list) else [value]
        if not any(old in t for t in texts):
            raise SystemExit(f"[{word}] manual edit {e} no longer applies to TDK's text")
        fixed = [t.replace(old, new) for t in texts]
        return sense.model_copy(update={field: fixed if isinstance(value, list) else fixed[0]})

    out: list[PolysemyWord] = []
    problems: list[str] = []
    for entry in manifest["words"]:
        word = entry["word"]
        want = [s["sense_id"] for s in entry["senses"]]
        got = by_word.get(word)
        if got is None:
            problems.append(f"  [{word}] absent from this build (TDK entry changed or fetch failed)")
            continue
        have = {s.sense_id: s for s in got.senses}
        missing = [sid for sid in want if sid not in have]
        if missing:
            problems.append(f"  [{word}] {len(missing)} recorded sense(s) no longer returned: {missing}")
            continue
        senses = []
        for sid in want:
            s = have[sid]
            for e in edits.get((word, sid), []):
                s = apply_edit(word, s, e)
            senses.append(s)
        out.append(PolysemyWord(word=word, senses=senses))

    n_senses = sum(len(w.senses) for w in out)
    print(
        f"\n=== manifest replay: {len(out)}/{len(manifest['words'])} words, "
        f"{n_senses} senses, {len(edits)} manual edit(s) applied ===",
        file=sys.stderr,
    )
    if problems:
        print("MANIFEST DRIFT — the rebuild does not match the published dataset:", file=sys.stderr)
        for line in problems:
            print(line, file=sys.stderr)
        raise SystemExit(
            "aborting: TDK's entries have changed since the manifest was recorded, so this "
            "rebuild would not reproduce the published dataset. Report the drift rather than "
            "publishing a silently different dataset."
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument(
        "--manifest",
        default=None,
        help="released sense manifest (data/polysemy/tr_polysemy.v3.sense_manifest.json). "
             "When given, the build is restricted and reordered to exactly the recorded "
             "sense IDs and the recorded manual edits are applied, reproducing the "
             "published dataset deterministically. Aborts on any drift.",
    )
    p.add_argument("--version", default="v1-2026-05-05")
    p.add_argument("--min-examples", type=int, default=2)
    p.add_argument("--min-senses", type=int, default=2)
    p.add_argument("--sleep", type=float, default=0.3, help="seconds between API calls")
    p.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="override default candidate list (space-separated words)",
    )
    args = p.parse_args()

    if args.manifest:
        # A manifest records the parameters it was built with; replaying it with
        # the defaults would filter out recorded senses.
        recorded = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        for key, value in recorded.get("params", {}).items():
            setattr(args, key, value)
        args.version = recorded.get("version", args.version)

    if args.candidates:
        candidates = args.candidates
    elif args.manifest:
        # Replaying a manifest: query exactly its headwords, not the full
        # candidate pool the original crawl started from.
        candidates = [w["word"] for w in
                      json.loads(Path(args.manifest).read_text(encoding="utf-8"))["words"]]
    else:
        candidates = CANDIDATES
    print(f"querying TDK GTS for {len(candidates)} candidates", file=sys.stderr)

    kept: list[PolysemyWord] = []
    fetched: dict[str, list[PolysemySense]] = {}
    dropped: list[tuple[str, str]] = []
    dangling_report: list[str] = []

    for w in candidates:
        records = fetch_madde(w)
        time.sleep(args.sleep)
        if records is None:
            dropped.append((w, "fetch failed"))
            continue
        senses = extract_senses(records, w)
        fetched[w] = senses
        n_total = len(senses)
        n_qualifying = sum(1 for s in senses if len(s.examples) >= args.min_examples)
        kept_senses = filter_word(senses, args.min_examples, args.min_senses)
        if kept_senses is not None:
            for kept_s, ante in detect_dangling_anaphora(senses, kept_senses):
                dangling_report.append(
                    f"  [{w}] dangling anaphora: kept sense {kept_s.sense_id} "
                    f"«{kept_s.definition}» references dropped antecedent "
                    f"{ante.sense_id} «{ante.definition}» — resolve the gloss manually"
                )
        if kept_senses is None:
            dropped.append(
                (
                    w,
                    f"{n_qualifying}/{n_total} senses have ≥{args.min_examples} examples "
                    f"(need ≥{args.min_senses})",
                )
            )
            continue
        kept.append(PolysemyWord(word=w, senses=kept_senses))
        print(
            f"  [{w}] kept {len(kept_senses)}/{n_total} senses; "
            f"examples per sense: {[len(s.examples) for s in kept_senses]}",
            file=sys.stderr,
        )

    print("", file=sys.stderr)
    print(f"=== summary: kept {len(kept)} / {len(candidates)} ===", file=sys.stderr)
    if dropped:
        print("dropped:", file=sys.stderr)
        for w, reason in dropped:
            print(f"  [{w}] {reason}", file=sys.stderr)
    if dangling_report:
        print(
            f"\n⚠️  {len(dangling_report)} DANGLING ANAPHORA — filtering dropped an "
            f"antecedent referenced by a kept sense:",
            file=sys.stderr,
        )
        for line in dangling_report:
            print(line, file=sys.stderr)

    # Deduplicate near-homograph spellings (e.g. 'alem' / 'âlem'): if two kept
    # words share a circumflex-normalized form, keep the standard (non-circumflex)
    # spelling and drop the variant.
    def _decirc(s: str) -> str:
        table = str.maketrans("âîûêôÂÎÛÊÔ", "aiueoAIUEO")
        return s.lower().translate(table)

    # Drop non-noun headwords flagged by the annotation-readiness audit
    # (adjective/adverb/interjection lemmas that break a noun-polysemy task).
    NON_NOUN = {"geri", "ileri", "kırık", "kesme", "usul", "güç"}
    kept = [w for w in kept if w.word not in NON_NOUN]

    seen: dict[str, str] = {}
    deduped: list[PolysemyWord] = []
    for w in kept:
        key = _decirc(w.word)
        prev = seen.get(key)
        if prev is None:
            seen[key] = w.word
            deduped.append(w)
        else:
            # prefer the spelling without circumflex
            keep_new = (w.word == _decirc(w.word)) and (prev != _decirc(prev))
            print(f"  [dedup] '{w.word}' ~ '{prev}' (same normalized form); "
                  f"keeping '{w.word if keep_new else prev}'", file=sys.stderr)
            if keep_new:
                deduped = [x for x in deduped if x.word != prev]
                seen[key] = w.word
                deduped.append(w)
    kept = deduped

    if args.manifest:
        kept = _apply_manifest(kept, Path(args.manifest), fetched)

    ds = PolysemyDataset(
        language="tr",
        version=args.version,
        source=f"TDK GTS API ({API_BASE}); min_examples={args.min_examples}, min_senses={args.min_senses}",
        words=kept,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
