"""
classify_debug.py — Módulo de instrumentación del pipeline de clasificación v6.

PROPÓSITO:
  Genera una traza JSON completa de cada decisión tomada por classify(),
  apta para ser visualizada con scorer_visualizer.html.

USO:
  # Opción A — traza de un POI individual:
  from classify_debug import classify_debug
  import json

  trace = classify_debug(
      nombre="Fuente del Pino",
      categoria="espacio natural",
      descripcion="Fuente histórica de aguas minerales..."
  )
  print(json.dumps(trace, ensure_ascii=False, indent=2))

  # Opción B — procesar CSV completo y guardar trazas en JSONL:
  python classify_debug.py --csv Servicios_Limpio.csv --out trazas.jsonl

  # Opción C — traza de un POI desde línea de comandos:
  python classify_debug.py --nombre "Fuente del Pino" \
                           --categoria "espacio natural" \
                           --desc "Fuente histórica de aguas minerales"

IMPORTANTE:
  Este módulo importa directamente del script principal (classifier_v6.py).
  Asegúrate de que el fichero se llama classifier_v6.py o ajusta el import.
  Si el script principal tiene otro nombre, modifica la línea de import abajo.
"""

import json
import re
import sys
import argparse
import unicodedata
from copy import deepcopy

# ── Import del script principal ──────────────────────────────────────────────
# Ajusta el nombre si tu script tiene otro nombre de fichero.
try:
    import classifier_v6 as clf
except ModuleNotFoundError:
    # Intentar importar el script directamente por si tiene guiones en el nombre
    import importlib.util, os
    _candidates = [
        "classifier_v6.py", "clasificador_v6.py", "classifier.py",
        "clasificador.py", "main.py"
    ]
    _found = None
    for _c in _candidates:
        if os.path.exists(_c):
            _found = _c
            break
    if not _found:
        sys.exit(
            "[ERROR] No se encontró el script principal del clasificador.\n"
            "Ajusta el import en classify_debug.py o coloca ambos ficheros "
            "en el mismo directorio."
        )
    _spec = importlib.util.spec_from_file_location("clf", _found)
    clf   = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(clf)


# ════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL DE TRAZA
# ════════════════════════════════════════════════════════════════════════════

def classify_debug(nombre, categoria, descripcion):
    """
    Ejecuta el pipeline completo de classify() y captura una traza detallada.

    Devuelve un dict JSON-serializable con:
      input       — valores de entrada normalizados
      steps       — dict con la salida de cada etapa del pipeline
      scores_raw  — ranking completo ANTES de bonus dominante
      scores_final— ranking completo DESPUÉS de bonus dominante
      result      — resultado final idéntico al de classify()

    La traza no altera el resultado: es read-only.
    """
    import pandas as pd

    trace = {
        "input":        {},
        "steps":        {},
        "scores_raw":   [],
        "scores_final": [],
        "result":       {},
    }

    # ── PASO 1 — Normalización de tipos ──────────────────────────────────
    if categoria is None or (isinstance(categoria, float) and pd.isna(categoria)):
        categoria = ""
    else:
        categoria = str(categoria)
    categoria_lower = categoria.strip().lower()

    trace["input"] = {
        "nombre":     str(nombre) if isinstance(nombre, str) else "",
        "categoria":  categoria,
        "descripcion": str(descripcion)[:300] if isinstance(descripcion, str) else "",
    }

    # ── PASO 2 — Limpieza HTML + abreviaturas ─────────────────────────────
    name_raw      = clf.expand_abbrev(clf.strip_html(nombre if isinstance(nombre, str) else ""))
    desc_html_raw = descripcion if isinstance(descripcion, str) else ""
    desc_raw      = clf.expand_abbrev(clf.strip_html(desc_html_raw))
    name_clean    = clf.strip_address_suffix(name_raw, categoria_lower)

    trace["steps"]["normalization"] = {
        "name_raw":    name_raw,
        "name_clean":  name_clean,
        "desc_raw":    desc_raw[:200],
        "address_stripped": name_raw != name_clean,
    }

    # ── PASO 3 — Tokenización ─────────────────────────────────────────────
    tok_n               = clf.tokenize(name_clean)
    tok_d               = clf.tokenize(desc_raw)
    tok_f               = tok_n | tok_d
    name_tokens_ordered = clf.TOKEN_REGEX.findall(name_clean.lower())

    trace["steps"]["tokenization"] = {
        "name_tokens_ordered": name_tokens_ordered,
        "tok_n":  sorted(tok_n),
        "tok_d":  sorted(tok_d)[:30],
        "tok_f":  sorted(tok_f)[:40],
    }

    # ── PASO 4 — Separación genitiva  [Cambio C] ──────────────────────────
    tok_n_head, tok_n_complement = clf.split_name_head_complement(name_tokens_ordered)
    tok_n_clean = tok_n_head

    # Detectar si se activó la excepción de label multipalabra
    multiword_exception = False
    for _it in clf.ONTO:
        lw = clf.label_words(_it["label"])
        if len(lw) >= 2 and lw.issubset(set(name_tokens_ordered)):
            multiword_exception = True
            break

    # Detectar el conector usado
    connector_found = None
    for i, t in enumerate(name_tokens_ordered):
        if t in clf._GENITIVE_CONNECTORS and i > 0:
            connector_found = t
            break

    trace["steps"]["genitive_split"] = {
        "head_tokens":        sorted(tok_n_head),
        "complement_tokens":  sorted(tok_n_complement),
        "connector_found":    connector_found,
        "multiword_exception": multiword_exception,
        "split_applied":      bool(tok_n_complement),
    }

    # ── PASO 5 — Primera palabra significativa  [Cambio A] ────────────────
    _raw_first = clf.first_significant_token(name_tokens_ordered)
    _is_sem    = clf.is_first_tok_semantic(_raw_first)
    first_tok  = _raw_first if _is_sem else None

    trace["steps"]["first_token"] = {
        "raw_first":      _raw_first,
        "is_semantic":    _is_sem,
        "effective_first_tok": first_tok,
        "bonus_will_apply": first_tok is not None,
    }

    # ── PASO 6 — Candidatos del mapping ───────────────────────────────────
    mapping_labels, ruta = clf.mapping_candidate_labels(categoria)

    trace["steps"]["mapping"] = {
        "categoria_origen": categoria,
        "candidates":       sorted(mapping_labels),
        "ruta":             ruta,
        "level_used":       _detect_mapping_level(ruta, mapping_labels),
        "deterministic":    len(mapping_labels) == 1,
    }

    # ── PASO 7 — Reglas duras ─────────────────────────────────────────────
    hard_entry, hard_reason = clf.apply_hard_rules(
        name_clean, categoria_lower,
        desc_html_raw, desc_raw,
        tok_n, tok_d, tok_f,
    )

    trace["steps"]["hard_rules"] = {
        "fired":       hard_entry is not None,
        "rule":        hard_reason,
        "label":       hard_entry.get("label") if hard_entry else None,
    }

    if hard_entry:
        result = clf.build_result(hard_entry, 0.92, False, False, hard_reason)
        trace["result"]  = result
        trace["shortcircuit"] = "hard_rule"
        return trace

    # ── PASO 7b — Cortocircuito mapping determinista  [FIX-MAP-2] ─────────
    if len(mapping_labels) == 1:
        sole_label = next(iter(mapping_labels))
        entry = clf.ontology_entry(sole_label)
        if entry and not clf.has_negative(tok_f, entry):
            reason = f"Mapping determinista (1 label): '{categoria}' → {sole_label}"
            result = clf.build_result(entry, 0.95, False, False, reason)
            trace["steps"]["hard_rules"]["mapping_deterministic"] = True
            trace["result"]     = result
            trace["shortcircuit"] = "mapping_deterministic"
            return trace

    # ── PASO 8 — Scoring ontología ────────────────────────────────────────
    scores:        dict = {}
    name_map_labels: set = set()
    signal_detail: dict = {}   # Para la traza: señales disparadas por label

    def add(label, pts, reason, signal_type="scoring"):
        if label not in scores:
            scores[label]        = [pts, [reason]]
            signal_detail[label] = {"signals": [(signal_type, pts, reason)], "in_map": False}
        else:
            scores[label][0] += pts
            scores[label][1].append(reason)
            signal_detail[label]["signals"].append((signal_type, pts, reason))

    for it in clf.ONTO:
        lab    = it["label"]
        lw     = clf.label_words(lab)
        in_map = lab in mapping_labels

        if clf.has_negative(tok_f, it):
            signal_detail[lab] = {"signals": [("negative_filter", 0, "keywords_negativas match")], "in_map": in_map, "filtered": True}
            continue

        kw_n = clf.onto_keywords_match(tok_n_clean, it)
        kw_d = clf.onto_keywords_match(tok_d, it)
        kw_f = clf.onto_keywords_match(tok_n_clean | tok_d, it)

        # SEÑAL 1
        if lw and lw.issubset(tok_n_clean):
            if in_map:
                name_map_labels.add(lab)
                add(lab, 15, f"'{lab}' en nombre (mapping)", "signal1_name_map")
            elif len(lw) > 1:
                bonus_multi = 2
                add(lab, 7 + bonus_multi, f"'{lab}' en nombre (multipalabra)", "signal1_multiword")
            else:
                lw_token = next(iter(lw))
                leading  = lw_token in name_tokens_ordered[:2]
                first_bonus = 4 if (first_tok and lw_token == first_tok) else 0
                pts  = (7 if leading else 3) + first_bonus
                stype = "signal1_leading" if leading else "signal1_trailing"
                add(lab, pts, f"'{lab}' en nombre ({'primera' if first_bonus else 'leading' if leading else 'trailing'})", stype)
            if lab in signal_detail:
                signal_detail[lab]["in_map"] = in_map
            continue

        # SEÑAL 2
        if kw_f >= 2:
            pts = 8 if in_map else 5
            add(lab, pts, f"keywords en nombre/desc ({kw_f})", "signal2_kw_multi")
            if lab in signal_detail:
                signal_detail[lab]["in_map"] = in_map
            continue

        # SEÑAL 3
        if kw_f == 1 and lw:
            if len(lw) == 1:
                pts = 9 if in_map else 6
                add(lab, pts, f"keyword en nombre/desc", "signal3_kw_single")
            else:
                kw_multi_bonus = 2
                pts = (5 if in_map else 3) + kw_multi_bonus
                add(lab, pts, f"1 keyword multipalabra", "signal3_kw_multi")
            if lab in signal_detail:
                signal_detail[lab]["in_map"] = in_map
            continue

        # SEÑAL 4
        if lw.issubset(tok_f) and not lw.issubset(tok_n_clean):
            if in_map and len(lw) >= 2:
                add(lab, 2, f"'{lab}' en descripción (mapping, multipalabra)", "signal4_desc")
            elif len(lw) >= 3:
                add(lab, 1, f"'{lab}' en descripción (multipalabra)", "signal4_desc")

        no_mapping = not mapping_labels
        if kw_d >= 2 and kw_n == 0:
            if in_map:
                add(lab, 2, f"keywords en desc ({kw_d}) (mapping)", "signal4_kw_desc")
            elif no_mapping and kw_d >= 3:
                add(lab, 2, f"keywords en desc ({kw_d}) (sin mapping)", "signal4_kw_desc")
        if no_mapping and lw.issubset(tok_d) and not lw.issubset(tok_n_clean) and len(lw) >= 2:
            add(lab, 2, f"'{lab}' en descripción (sin mapping)", "signal4_desc")

    # ── PASO 8b — Aliases ─────────────────────────────────────────────────
    aliases_fired = []
    for alias, target in clf.ALIASES.items():
        alias_tokens = set(clf.TOKEN_REGEX.findall(alias.lower()))
        if not alias_tokens:
            continue
        if alias_tokens.issubset(tok_n_clean):
            entry = clf.ontology_entry(target)
            if entry and not clf.has_negative(tok_f, entry):
                in_map = entry["label"] in mapping_labels
                bonus  = 1 if len(alias_tokens) > 1 else 0
                pts    = (9 if in_map else 7) + bonus
                add(entry["label"], pts, f"alias '{alias}'→{entry['label']}", "alias")
                aliases_fired.append({"alias": alias, "target": entry["label"], "pts": pts})

    # Snapshot ANTES de bonus dominante
    scores_before_bonus = {k: v[0] for k, v in scores.items()}

    # ── PASO 9 — Ancla de categoría  [FIX-10] ────────────────────────────
    anchors_applied = []
    if mapping_labels and len(mapping_labels) <= 5:
        for lab in mapping_labels:
            if lab not in scores:
                entry_check = clf.ontology_entry(lab)
                if entry_check and not clf.has_negative(tok_f, entry_check):
                    add(lab, 8, "ancla de categoría (mapping directo, sin señal de texto)", "anchor")
                    anchors_applied.append(lab)

    # ── PASO 10 — Bonus dominante  [FIX-8] ───────────────────────────────
    dominant_applied = []
    for lab in name_map_labels:
        if lab in scores:
            scores[lab][0] += clf._DOMINANT_BONUS
            scores[lab][1].append(f"+{clf._DOMINANT_BONUS} (acuerdo nombre+mapping)")
            if lab in signal_detail:
                signal_detail[lab]["signals"].append(("dominant_bonus", clf._DOMINANT_BONUS, "acuerdo nombre+mapping"))
            dominant_applied.append({"label": lab, "bonus": clf._DOMINANT_BONUS})

    # ── Snapshot DESPUÉS de bonus dominante
    scores_after_bonus = {k: v[0] for k, v in scores.items()}

    # ── PASO 11 — Desambiguaciones manuales ──────────────────────────────
    disambig_applied = []
    _before_dis = deepcopy(scores)

    if "Plaza de toros" in scores and "Plaza" in scores and {"plaza", "toros"}.issubset(tok_n_clean):
        scores["Plaza"][0] = 0
        disambig_applied.append("Plaza→0 (Plaza de toros ganó)")

    if "Bar de vinos" in scores and "Bar" in scores and "vinos" in tok_n_clean:
        scores["Bar"][0] = 0
        disambig_applied.append("Bar→0 (Bar de vinos ganó)")

    if "Parque natural" in scores and "Parque infantil" in scores:
        if "natural"    in tok_n_clean:
            scores["Parque infantil"][0] = 0
            disambig_applied.append("Parque infantil→0 ('natural' en nombre)")
        elif "infantil" in tok_n_clean:
            scores["Parque natural"][0]  = 0
            disambig_applied.append("Parque natural→0 ('infantil' en nombre)")

    if "Mirador natural" in scores and "Mirador urbano" in scores:
        if "urbano" in tok_n_clean:
            scores["Mirador natural"][0] = 0
            disambig_applied.append("Mirador natural→0 ('urbano' en nombre)")
        else:
            scores["Mirador urbano"][0]  = 0
            disambig_applied.append("Mirador urbano→0 (sin 'urbano' en nombre)")

    if "Cine" in scores and "cine" not in tok_n_clean:
        scores["Cine"][0] = 0
        disambig_applied.append("Cine→0 ('cine' no en nombre)")

    scores = {k: v for k, v in scores.items() if v[0] > 0}

    # ── Construir rankings completos para la traza ─────────────────────────
    def _rank(scores_dict, detail_dict):
        rows = []
        for lab, (sc, reasons) in sorted(scores_dict.items(), key=lambda kv: -kv[1][0]):
            sd    = detail_dict.get(lab, {})
            sigs  = sd.get("signals", [])
            rows.append({
                "label":    lab,
                "score":    sc,
                "in_map":   lab in mapping_labels,
                "reasons":  reasons[:6],
                "signals":  [(s[0], s[1], s[2]) for s in sigs],
            })
        return rows

    trace["scores_raw"]   = _rank(
        {k: [scores_before_bonus[k], scores[k][1]] for k in scores_before_bonus if k in scores},
        signal_detail
    )
    trace["scores_final"] = _rank(scores, signal_detail)

    trace["steps"]["scoring"] = {
        "entries_evaluated":  len(clf.ONTO),
        "labels_with_score":  len(scores),
        "name_map_labels":    sorted(name_map_labels),
        "anchors_applied":    anchors_applied,
        "dominant_applied":   dominant_applied,
        "aliases_fired":      aliases_fired,
        "disambiguation":     disambig_applied,
        "mapping_candidates": sorted(mapping_labels),
    }

    # ── PASO 12 — Fallback ────────────────────────────────────────────────
    if not scores:
        default = clf.DEFAULTS.get(categoria_lower)
        if default:
            entry = clf.ontology_entry(default)
            if entry:
                reason = f"Sin señales específicas; default de '{categoria}' → {entry['label']}"
                result = clf.build_result(entry, 0.5, False, False, reason)
                trace["result"]      = result
                trace["shortcircuit"] = "fallback_default"
                return trace
        result = clf.build_unclassified()
        trace["result"]      = result
        trace["shortcircuit"] = "no_clasificado"
        return trace

    # ── PASO 13 — Selección ganador ───────────────────────────────────────
    ranked       = sorted(scores.items(), key=lambda kv: -kv[1][0])
    best_label,  (best_score,  best_reasons) = ranked[0]
    second_label = ranked[1][0]    if len(ranked) > 1 else None
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0

    ambig     = bool(second_label and (best_score - second_score) < 2 and best_score < 10)
    in_map_f  = best_label in mapping_labels
    conflicto = bool(ruta and not in_map_f)

    conf = min(0.98, 0.4 + best_score * 0.05)
    if ambig:     conf = max(0.35, conf - 0.10)
    if conflicto: conf = max(0.45, conf - 0.05)

    entry       = clf.ontology_entry(best_label)
    explicacion = "; ".join(best_reasons[:3])
    if ambig and second_label:
        explicacion += f" | alt: {second_label}"
    if conflicto:
        explicacion = f"[Conflicto con '{categoria}'] " + explicacion

    result = clf.build_result(entry, conf, conflicto, ambig, explicacion)

    trace["steps"]["winner"] = {
        "best_label":    best_label,
        "best_score":    best_score,
        "second_label":  second_label,
        "second_score":  second_score,
        "gap":           best_score - second_score,
        "ambiguous":     ambig,
        "conflict":      conflicto,
        "in_map":        in_map_f,
        "confidence":    round(conf, 2),
    }
    trace["result"] = result

    return trace


# ════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ════════════════════════════════════════════════════════════════════════════

def _detect_mapping_level(ruta, mapping_labels):
    """Detecta qué nivel del mapping resolvió los candidatos."""
    if not ruta or not mapping_labels:
        return None
    for key in ("nivel_4", "nivel_3", "nivel_2"):
        vals = ruta.get(key, []) or []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            if v and v.strip().lower() in {m.lower() for m in mapping_labels}:
                return key
    return "cluster_expand"


def batch_debug(csv_path, out_path, max_rows=None, sep=";"):
    """
    Procesa el CSV y guarda una traza JSONL (una línea por fila).
    Útil para análisis masivo o para alimentar el visualizador.
    """
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=sep, encoding="latin-1")
    df.columns = [c.strip() for c in df.columns]

    cols = clf.detect_columns(df.columns)
    if max_rows:
        df = df.head(max_rows)

    total = len(df)
    with open(out_path, "w", encoding="utf-8") as f_out:
        for i, row in enumerate(df.itertuples(index=False), 1):
            nombre     = getattr(row, cols["nombre"]    or "", "") or ""
            categoria  = getattr(row, cols["categoria"] or "", "") or ""
            descripcion= getattr(row, cols["descripcion"] or "", "") or ""
            trace = classify_debug(str(nombre), str(categoria), str(descripcion))
            f_out.write(json.dumps(trace, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{total} filas procesadas…")

    print(f"[OK] Trazas guardadas en '{out_path}' ({total} filas)")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT CLI
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera trazas de debug del clasificador v6."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nombre",  help="Nombre del POI (modo individual)")
    group.add_argument("--csv",     help="Ruta al CSV (modo batch)")

    parser.add_argument("--categoria", default="", help="Categoría del POI")
    parser.add_argument("--desc",      default="", help="Descripción del POI")
    parser.add_argument("--out",  default="trazas.jsonl",
                        help="Fichero de salida para modo batch (defecto: trazas.jsonl)")
    parser.add_argument("--max",  type=int, default=None,
                        help="Número máximo de filas a procesar en modo batch")
    parser.add_argument("--pretty", action="store_true",
                        help="Pretty-print JSON en modo individual")

    args = parser.parse_args()

    if args.nombre:
        trace = classify_debug(args.nombre, args.categoria, args.desc)
        indent = 2 if args.pretty else None
        print(json.dumps(trace, ensure_ascii=False, indent=indent))
    else:
        batch_debug(args.csv, args.out, max_rows=args.max)