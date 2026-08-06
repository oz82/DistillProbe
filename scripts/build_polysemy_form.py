"""Generate the polysemy human-annotation forms (P1-5) from the primary dataset.

Produces, for each of two independent annotators (A/B), an Excel workbook (.xlsx):
annotator-friendly with 0-3 dropdown validation, frozen header, highlighted score
column, per-word shading, and a short instructions sheet.

One row per (word, sense); the annotator fills only PUAN_0_3 (0-3 salience) and an
optional NOT. See annotation/polysemy_annotation_GUIDE.md for the full task.

Usage: python scripts/build_polysemy_form.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "polysemy" / "tr_polysemy.v3.json"
OUTDIR = ROOT / "annotation"

COLS = ["kelime_no", "kelime", "anlam_no", "toplam_anlam",
        "tanim", "ornek_cumle", "PUAN_0_3", "NOT"]


def _rows():
    words = json.loads(DATA.read_text(encoding="utf-8"))["words"]
    out = []
    for wi, w in enumerate(words, 1):
        ns = len(w["senses"])
        for si, s in enumerate(w["senses"], 1):
            ex = [e.strip() for e in (s.get("examples") or []) if e and e.strip()]
            out.append({
                "kelime_no": wi, "kelime": w["word"],
                "anlam_no": si, "toplam_anlam": ns,
                "tanim": (s.get("definition") or "").strip(),
                "ornek_cumle": " / ".join(ex),   # tüm örnekler
                "PUAN_0_3": "", "NOT": "",
            })
    return out


def _write_xlsx(rows, who):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Protection, Side
    from openpyxl.worksheet.datavalidation import DataValidation

    wb = Workbook()

    # --- Sheet 1: instructions + annotator info + consent ------------------
    g = wb.active
    g.title = "Yönerge"
    g.sheet_view.showGridLines = False
    g.column_dimensions["A"].width = 36
    g.column_dimensions["B"].width = 50
    box_fill = PatternFill("solid", fgColor="FFFDE7")
    bthin = Side(style="thin", color="BFBFBF")
    box_border = Border(left=bthin, right=bthin, top=bthin, bottom=bthin)

    state = {"r": 1}

    def span(text, bold=False, size=11, height=None):
        r = state["r"]
        c = g.cell(row=r, column=1, value=text)
        c.font = Font(bold=bold, size=size)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        g.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)
        if height:
            g.row_dimensions[r].height = height
        state["r"] = r + 1

    def field(label):
        r = state["r"]
        lc = g.cell(row=r, column=1, value=label)
        lc.font = Font(bold=True)
        lc.alignment = Alignment(vertical="center")
        inp = g.cell(row=r, column=2)
        inp.fill = box_fill
        inp.border = box_border
        inp.protection = Protection(locked=False)   # annotator-editable
        state["r"] = r + 1

    def blank():
        state["r"] += 1

    span("Polysemy anotasyon — kısa yönerge", bold=True, size=13)
    blank()
    span("Soru: Bu kelimeyi TEK BAŞINA (cümlesiz) duysam, bu anlam ne kadar aklıma gelir?")
    blank()
    span("PUAN_0_3 sütununa yaz (açılır menüden seç):", bold=True)
    span("  3 = ilk akla gelen, baskın anlam")
    span("  2 = yaygın ama ikincil anlam")
    span("  1 = nadir / teknik / dar anlam")
    span("  0 = bu anlamı biliyorum ama kelime için neredeyse hiç akla gelmez (marjinal)")
    span("  Bu anlamı BİLMİYORSAN: PUAN'ı boş bırak + NOT'a 'bilmiyorum' yaz (0 verme).", bold=True)
    blank()
    span("Kurallar:", bold=True)
    span("  • Sadece PUAN_0_3 (ve istersen NOT) sütununu doldur. Diğerlerine dokunma.")
    span("  • Örnek cümleyi anlamı ANLAMAK için oku; ama puanı KELİME için ver, cümle için değil.")
    span("  • Her kelimede en az bir anlam 3 almalı; eşit puan vermek serbest.")
    span("  • Bu görevi diğer kişilerle KONUŞMADAN, bağımsız doldur.")
    span("  • Sözlüğe/internete bakma; bilmediğin anlamda PUAN'ı boş bırak (0 verme).")
    blank()
    span("'Form' sekmesine geç ve doldurmaya başla. Aynı kelimenin anlamları peş peşedir.")
    blank()
    span("─────────  İŞARETLEYİCİ BİLGİLERİ (lütfen doldurun)  ─────────", bold=True)
    field("Ad–Soyad:")
    field("Tarih:")
    field("Ana diliniz Türkçe mi? (Evet/Hayır):")
    field("Yaş (isteğe bağlı):")
    blank()
    span("─────────  ONAY  ─────────", bold=True)
    span("Aşağıdaki ifadeyi okuduktan sonra onay hücresine 'Evet' yazın:", bold=True)
    span("«Yönergeyi okudum ve görevi anladım. Puanları, başka biriyle konuşmadan, "
         "kendi bağımsız yargımla verdim.»", height=32)
    field("Onaylıyorum (yazın: Evet):")
    field("İmza / baş harfler:")
    g.protection.sheet = True   # lock instructions; only the fillable boxes are editable

    # --- Sheet 2: the form -------------------------------------------------
    ws = wb.create_sheet("Form")
    header_fill = PatternFill("solid", fgColor="2F5496")
    header_font = Font(bold=True, color="FFFFFF")
    score_fill = PatternFill("solid", fgColor="FFF2CC")     # light yellow: fill here
    shade_fill = PatternFill("solid", fgColor="F2F2F2")     # alternate word shading
    thin = Side(style="thin", color="D9D9D9")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    wrap_top = Alignment(wrap_text=True, vertical="top")
    center = Alignment(horizontal="center", vertical="center")

    ws.append(COLS)
    for ci, _ in enumerate(COLS, 1):
        c = ws.cell(row=1, column=ci)
        c.fill = header_fill
        c.font = header_font
        c.alignment = center
        c.border = border

    for r in rows:
        ws.append([r[k] for k in COLS])

    widths = {"kelime_no": 8, "kelime": 13, "anlam_no": 8, "toplam_anlam": 11,
              "tanim": 58, "ornek_cumle": 46, "PUAN_0_3": 11, "NOT": 26}
    from openpyxl.utils import get_column_letter
    for ci, col in enumerate(COLS, 1):
        ws.column_dimensions[get_column_letter(ci)].width = widths[col]

    puan_idx = COLS.index("PUAN_0_3") + 1
    puan_letter = get_column_letter(puan_idx)

    for ri, r in enumerate(rows, start=2):
        shade = (int(r["kelime_no"]) % 2 == 0)
        for ci, col in enumerate(COLS, 1):
            c = ws.cell(row=ri, column=ci)
            c.border = border
            if col in ("tanim", "ornek_cumle"):
                c.alignment = wrap_top
            elif col == "NOT":
                c.alignment = wrap_top
            else:
                c.alignment = center
            if col in ("PUAN_0_3", "NOT"):
                c.protection = Protection(locked=False)   # annotator-editable
            if col == "PUAN_0_3":
                c.fill = score_fill
            elif shade:
                c.fill = shade_fill

    # 0-3 dropdown validation on the whole PUAN column
    dv = DataValidation(type="list", formula1='"0,1,2,3"', allow_blank=True,
                        showErrorMessage=True, showInputMessage=True,
                        errorTitle="Geçersiz puan",
                        error="Lütfen yalnızca 0, 1, 2 veya 3 seç. (Bilmiyorsan boş bırak + NOT'a yaz.)")
    dv.prompt = "0=marjinal · 1=nadir · 2=yaygın ikincil · 3=baskın · bilmiyorsan boş bırak"
    dv.promptTitle = "Salience puanı"
    ws.add_data_validation(dv)
    dv.add(f"{puan_letter}2:{puan_letter}{len(rows) + 1}")

    ws.freeze_panes = "A2"           # keep header visible
    ws.auto_filter.ref = f"A1:{get_column_letter(len(COLS))}1"
    ws.protection.sheet = True       # lock all cells except PUAN/NOT (unlocked above)

    fn = OUTDIR / f"polysemy_form_annotator_{who}.xlsx"
    wb.save(fn)
    return fn


def main() -> int:
    import argparse
    import string

    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--annotators", type=int, default=2,
                    help="number of independent annotators (one .xlsx each, labelled A, B, C, ...)")
    args = ap.parse_args()

    OUTDIR.mkdir(exist_ok=True)
    rows = _rows()
    for who in list(string.ascii_uppercase)[: args.annotators]:
        x = _write_xlsx(rows, who)
        print(f"wrote {x.name} — {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
