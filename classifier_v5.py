"""
Clasificador v6 — Primera palabra, multipalabra y complementos genitivos.

CAMBIOS RESPECTO A v5:

  [A - FIRST_TOKEN]
    La primera palabra significativa del nombre recibe un bonus de +4 pts.
    Ejemplo: 'Arquería del Ayuntamiento' → 'arquería' (primera palabra) puntúa
    más que 'ayuntamiento' (complemento).
    Condiciones:
      · Solo aplica en el nombre (no en descripción).
      · Se saltan stopwords (artículos, preposiciones, 'nuevo', 'viejo'…).
      · Si la primera palabra NO aparece en ningún label ni keyword de la
        ontología, se ignora (no es un descriptor semántico: topónimo, nombre
        propio…).
      · Excepción: si hay un label multipalabra que matchea completo
        (ej: 'Centro de interpretación'), el bonus de primera palabra
        no interfiere.

  [B - MULTI_BONUS]
    Los labels multipalabra y las keywords multipalabra reciben un bonus
    adicional (+2 pts) cuando matchean en su totalidad.
    Ejemplo: 'pinturas rupestres' (keyword de Yacimiento arqueológico) que
    matchea entera puntúa más que dos keywords monopalabra independientes.

  [C - GENITIVE]
    Tokens que aparecen en el complemento genitivo del nombre
    ('del / de / al + …') se excluyen del scoring del nombre.
    Ejemplo: 'Fuente del Pino' → 'pino' no entra en scoring (es el nombre
    del lugar, no el descriptor del POI).
    Condiciones:
      · Solo aplica en el NOMBRE.
      · tok_n original (sin filtrar) sigue disponible para hard rules,
        keywords negativas y tok_f.
      · Excepción: si los tokens completos contienen un label multipalabra
        reconocido ('centro de interpretación'), no se divide el nombre.

  [FIX-MAP-1] (de la sesión anterior)
    mapping_candidate_labels() usa estrategia 'hoja más profunda primero':
    recorre nivel_4 → nivel_3 → nivel_2 y devuelve el primer nivel que
    resuelve labels concretos, descartando los niveles superiores.
    Evita que 'servicio médico' (nivel_3) compita con 'farmacia' (nivel_4).

  [FIX-MAP-2] (de la sesión anterior)
    Si el mapping resuelve a exactamente 1 label concreto, la clasificación
    es determinista: se devuelve directamente sin pasar por scoring.
    Confianza: 0.95.

  Hereda todos los cambios de v5:
    [PERF-1..3] Expresiones regulares precompiladas, expand_abbrev con
                patrones compilados, itertuples en main().
    [API-1]     apply_hard_rules() recibe tok_f como argumento.
    [FIX-7..10] E.S. → estación de servicio, bonus dominante nombre+mapping,
                protección topónimos monopalabra, ancla de categoría.
"""

import json
import re
import sys
import unicodedata
from html import unescape
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — RUTAS DE FICHEROS
# ════════════════════════════════════════════════════════════════════════════

ONTO_PATH    = 'Ontology_SEGITTUR_2.json'
ALIASES_PATH = 'Ontology_SEGITTUR_2.json'
MAPPING_PATH = 'Mapping.json'
CSV_PATH     = 'Servicios_Limpio.csv'


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — EXPRESIONES REGULARES PRECOMPILADAS  [PERF-1]
# ════════════════════════════════════════════════════════════════════════════

TOKEN_REGEX      = re.compile(r"[a-záéíóúüñç]+")
HTML_TAG_REGEX   = re.compile(r"<[^>]+>")
WHITESPACE_REGEX = re.compile(r"\s+")

_ADDRESS_INDICATORS = (
    r"calle|avenida|plaza|paseo|bulevar|rua|"
    r"carretera|camino|travesía|travesia|callejón|callejon|"
    r"urbanización|urbanizacion|polígono|poligono|barrio|paraje"
)
ADDRESS_SUFFIX_REGEX = re.compile(
    r"(?<=\S)\s+(" + _ADDRESS_INDICATORS + r")\b",
    re.IGNORECASE,
)

# Patrones de abreviaturas  [PERF-2]
_ABBREV_PATTERNS = [
    (re.compile(r"\b[Cc]/\s*"),                   "calle "),
    (re.compile(r"\b[Aa]vda\.?\s*"),              "avenida "),
    (re.compile(r"\b[Aa]v\.\s*"),                 "avenida "),
    (re.compile(r"\b[Pp]za\.?\s*"),               "plaza "),
    (re.compile(r"\b[Pp]l\.\s*"),                 "plaza "),
    (re.compile(r"\b[Ss]ra\.?\s*"),               "señora "),
    (re.compile(r"\bE\.S\.\s*", re.IGNORECASE),   "estación de servicio "),
]


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — CARGA SEGURA DE DATOS
# ════════════════════════════════════════════════════════════════════════════

def _load_json(path, label):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"[ERROR] No se encontró el fichero '{label}' en:\n  {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] El fichero '{label}' no es JSON válido:\n  {e}")


ONTO         = _load_json(ONTO_PATH,    "Ontología SEGITTUR")
ALIASES_LIST = _load_json(ALIASES_PATH, "Ontología de aliases")
MAPPING_LIST = _load_json(MAPPING_PATH, "Mapping de categorías")


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — ÍNDICES EN MEMORIA
# ════════════════════════════════════════════════════════════════════════════

MAPPING     = {m["categoria_origen"].strip().lower(): m["ruta"] for m in MAPPING_LIST}
LABEL_INDEX = {it["label"].strip().lower(): it for it in ONTO if it.get("label")}

_LABEL_INDEX_NORM: dict = {}
_LABEL_INDEX_NORM_BUILT = False


def ontology_entry(label):
    """Busca una entrada de ontología: exacta primero, normalizada si falla."""
    global _LABEL_INDEX_NORM, _LABEL_INDEX_NORM_BUILT
    if not label:
        return None
    result = LABEL_INDEX.get(label.strip().lower())
    if result:
        return result
    if not _LABEL_INDEX_NORM_BUILT:
        _LABEL_INDEX_NORM = {normalize_text(k): v for k, v in LABEL_INDEX.items()}
        _LABEL_INDEX_NORM_BUILT = True
    return _LABEL_INDEX_NORM.get(normalize_text(label))


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — LIMPIEZA Y NORMALIZACIÓN DE TEXTO
# ════════════════════════════════════════════════════════════════════════════

def strip_html(s):
    """Elimina etiquetas HTML, entidades y espacios extra."""
    if not isinstance(s, str):
        return ""
    s = unescape(s)
    s = HTML_TAG_REGEX.sub(" ", s)
    s = s.replace("\xa0", " ")
    return WHITESPACE_REGEX.sub(" ", s).strip()


def expand_abbrev(s):
    """Expande abreviaturas de vías y estaciones. Usa patrones precompilados [PERF-2]."""
    if not isinstance(s, str):
        return ""
    for pattern, replacement in _ABBREV_PATTERNS:
        s = pattern.sub(replacement, s)
    return s


def strip_address_suffix(name_expanded, categoria_lower):
    """[FIX-4] Elimina la dirección postal concatenada al nombre."""
    if categoria_lower == "punto de ruta":
        return name_expanded
    m = ADDRESS_SUFFIX_REGEX.search(name_expanded)
    if m:
        return name_expanded[: m.start()].strip()
    return name_expanded


def tokenize(s):
    """Set de tokens en minúscula (solo letras). Usa TOKEN_REGEX [PERF-1]."""
    return set(TOKEN_REGEX.findall(s.lower()))


def normalize_text(s):
    """Minúsculas sin acentos para comparaciones tolerantes."""
    s = s.strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return WHITESPACE_REGEX.sub(" ", s).strip()


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — PRIMERA PALABRA SIGNIFICATIVA  [Cambio A]
# ════════════════════════════════════════════════════════════════════════════
# La primera palabra descriptiva del nombre es el descriptor principal del POI.
# Palabras funcionales (artículos, preposiciones, adjetivos genéricos) se saltan.

_STOPWORDS = {
    # Artículos
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    # Preposiciones y conjunciones
    "de", "del", "al", "a", "en", "con", "por", "para",
    "y", "e", "o", "u", "ni", "que", "se", "su", "sus",
    # Adjetivos genéricos que no discriminan
    "nuevo", "nueva", "nuevos", "nuevas",
    "viejo", "vieja", "viejos", "viejas",
    "antiguo", "antigua", "antiguos", "antiguas",
    # Hagiónimos muy frecuentes como prefijos de topónimos
    "san", "santa", "santo", "santos",
}


def first_significant_token(name_tokens_ordered):
    """
    Devuelve el primer token del nombre que no sea stopword.
    Devuelve None si todos son stopwords o la lista está vacía.

    Ejemplo:
      ['arqueria', 'del', 'ayuntamiento'] → 'arqueria'
      ['san', 'pedro', 'de', 'merida']    → 'pedro'  (pero luego se comprueba
                                                        si es semántico)
    """
    for tok in name_tokens_ordered:
        if tok not in _STOPWORDS:
            return tok
    return None


def is_first_tok_semantic(first_tok):
    """
    Comprueba si first_tok aparece en algún label o keyword de la ontología.
    Si no aparece en ningún lado, es un topónimo o nombre propio sin valor
    semántico para la clasificación → se ignora el bonus.

    Se llama una sola vez por fila, antes del loop de scoring.
    """
    if not first_tok:
        return False
    for _it in ONTO:
        # ¿Está en las palabras del label?
        if first_tok in label_words(_it["label"]):
            return True
        # ¿Está en alguna keyword?
        for _kw in (_it.get("keywords") or []):
            if first_tok in TOKEN_REGEX.findall(_kw.lower()):
                return True
    return False


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — COMPLEMENTOS GENITIVOS  [Cambio C]
# ════════════════════════════════════════════════════════════════════════════
# Separa los tokens del nombre en "head" (descriptor del POI) y "complemento"
# (topónimo o especificador que no describe la categoría).
#
# Ejemplos:
#   "Fuente del Pino"           → head={"fuente"},      complement={"pino"}
#   "Castillo de la Mota"       → head={"castillo"},    complement={"mota"}
#   "Iglesia de San Pedro"      → head={"iglesia"},     complement={"san","pedro"}
#   "Centro de interpretación"  → NO se divide (label multipalabra reconocido)
#   "Arquería del Ayuntamiento" → head={"arqueria"},    complement={"ayuntamiento"}

_GENITIVE_CONNECTORS = {"del", "de", "al"}


def split_name_head_complement(name_tokens_ordered):
    """
    Devuelve (head_tokens: set, complement_tokens: set).

    EXCEPCIÓN: si los tokens completos contienen un label multipalabra
    reconocido de la ontología, no se divide (se devuelve set completo, set vacío).
    Esto protege expresiones como 'centro de interpretación' o 'casa rural'.

    Los complement_tokens se usan SOLO para excluirlos de tok_n_clean.
    tok_n original (con todos los tokens) sigue disponible para las
    hard rules y las keywords negativas.
    """
    full_set = set(name_tokens_ordered)

    # ── Excepción: label multipalabra reconocido ─────────────────────────
    # Si el nombre (o una parte) contiene un label multipalabra de la ontología,
    # no dividir para que ese label pueda matchear completo.
    for _it in ONTO:
        lw = label_words(_it["label"])
        if len(lw) >= 2 and lw.issubset(full_set):
            return full_set, set()

    # ── Buscar el primer conector genitivo ───────────────────────────────
    connector_idx = None
    for i, tok in enumerate(name_tokens_ordered):
        # El conector debe aparecer DESPUÉS de al menos una palabra (i > 0)
        # para no dividir nombres que empiezan con "De la Fuente…"
        if tok in _GENITIVE_CONNECTORS and i > 0:
            connector_idx = i
            break

    if connector_idx is None:
        # Sin conector → todos los tokens son head
        return full_set, set()

    head_tokens       = set(name_tokens_ordered[:connector_idx])
    complement_tokens = set(name_tokens_ordered[connector_idx + 1:])
    # Quitamos stopwords del complemento (no aportan al scoring de todas formas)
    complement_tokens -= _STOPWORDS

    return head_tokens, complement_tokens


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — ALIASES
# ════════════════════════════════════════════════════════════════════════════

def _build_aliases(alias_ontology):
    result  = {}
    skipped = 0
    for entry in alias_ontology:
        label = (entry.get("label") or "").strip()
        if not label:
            continue
        for kw in entry.get("keywords") or []:
            kw_norm = kw.strip().lower()
            if not kw_norm:
                continue
            if kw_norm not in result:
                result[kw_norm] = label
            else:
                skipped += 1
    print(f"[INFO] Aliases cargados: {len(result)} "
          f"({skipped} keywords duplicadas ignoradas por precedencia)")
    return result


ALIASES = _build_aliases(ALIASES_LIST)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — DEFAULTS
# ════════════════════════════════════════════════════════════════════════════

DEFAULTS = {
    "edificio religioso":         "iglesia",
    "construcción civil":         "arquitectura vernácula",
    "monumento":                  "monumento",
    "espacio natural":            "parque natural",
    "castillos y fortalezas":     "castillo",
    "punto de ruta":              "ruta",
    "museo":                      "museo",
    "casa solariega":             "arquitectura vernácula",
    "parques y jardines":         "jardín",
    "zona arqueológica":          "yacimiento arqueológico",
    "paraje pintoresco":          "mirador natural",
    "arte rupestre":              "yacimiento arqueológico",
    "jardín histórico":           "jardín",
    "conjunto histórico":         "conjunto de interés artístico",
    "vía histórica":              "ruta",
    "escudos":                    "conjunto de interés artístico",
    "construcción militar":       "castillo",
    "oficina de turismo":         "oficina de turismo",
    "zona comercial":             "centro comercial",
    "otros núcleos de población": "destino turístico",
    "otros":                      "conjunto de interés artístico",
    "":                           "conjunto de interés artístico",
    "farmacia":                   "farmacia",
    "ayuntamiento":               "ayuntamiento",
    "supermercado":               "supermercado",
    "cajero":                     "cajero automático",
    "bomberos":                   "estación de bomberos",
    "instalación deportiva":      "polideportivo",
    "policía y/o guardía civil":  "comisaría de policía",
    "centro de salud":            "centro de atención primaria",
    "oficina de información turística": "oficina de turismo",
    "taller de automóvil":        "taller mecánico",
    "consulta médica":            "hospital público",
    "estación de servicio":       "gasolinera",
    "educación":                  "centro educativo",
    "cultura":                    "centro cultural",
}


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — EXPANSIÓN DE CLUSTERS
# ════════════════════════════════════════════════════════════════════════════

_CLUSTER_INDEX_N3: dict = {}
_CLUSTER_INDEX_N2: dict = {}
for _it in ONTO:
    _n2 = normalize_text(_it.get("nivel_2") or "")
    _n3 = normalize_text(_it.get("nivel_3") or "")
    _n4 = normalize_text(_it.get("nivel_4") or "")
    if _n3 and _n4:
        _CLUSTER_INDEX_N3.setdefault(_n3, []).append(_it["label"])
    if _n2 and _n3 and not _n4:
        _CLUSTER_INDEX_N2.setdefault(_n2, []).append(_it["label"])

_EXPAND_CACHE:    dict = {}
_WARNED_CLUSTERS: set  = set()


def expand_cluster(label):
    global _WARNED_CLUSTERS
    lab = normalize_text(label)
    if lab in _EXPAND_CACHE:
        return _EXPAND_CACHE[lab]
    out = list(_CLUSTER_INDEX_N3.get(lab, []))
    if not out:
        out = list(_CLUSTER_INDEX_N2.get(lab, []))
    if not out and lab not in _WARNED_CLUSTERS:
        _WARNED_CLUSTERS.add(lab)
        print(f"[WARN] expand_cluster: '{label}' no produjo hojas en la ontología.")
    _EXPAND_CACHE[lab] = out
    return out


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11 — CANDIDATOS DEL MAPPING  [FIX-MAP-1]
# ════════════════════════════════════════════════════════════════════════════

def mapping_candidate_labels(cat_origen):
    """
    Dado el valor de 'categoria' del CSV, devuelve:
      (set_de_labels_candidatos, objeto_ruta_del_mapping)

    [FIX-MAP-1] Estrategia 'hoja más profunda primero':
      Recorre nivel_4 → nivel_3 → nivel_2. En cuanto un nivel resuelve
      labels concretos de la ontología, los devuelve y descarta los
      niveles superiores. Esto evita que niveles intermedios (ej:
      'servicio médico') compitan con la hoja específica ('farmacia').
    """
    if not cat_origen or (isinstance(cat_origen, float) and pd.isna(cat_origen)):
        return set(), None
    cat_origen = str(cat_origen)
    ruta = MAPPING.get(cat_origen.strip().lower())
    if not ruta:
        return set(), None

    def _resolve_level(vals):
        """Convierte una lista de valores de nivel a labels concretos de la ontología."""
        concrete = set()
        clusters = []
        if isinstance(vals, str):
            vals = [vals]
        for v in (vals or []):
            if not v:
                continue
            for part in re.split(r"[|,]", v):
                part = part.strip()
                if not part:
                    continue
                entry = ontology_entry(part)
                if entry:
                    concrete.add(entry["label"])
                else:
                    clusters.append(part)
        if not concrete:
            for c in clusters:
                for sub in expand_cluster(c):
                    concrete.add(sub)
        return concrete

    # Recorrer de más específico a más general; devolver al primer nivel con resultados
    for key in ("nivel_4", "nivel_3", "nivel_2"):
        vals = ruta.get(key, []) or []
        resolved = _resolve_level(vals)
        if resolved:
            return resolved, ruta

    return set(), ruta


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12 — FUNCIONES DE SCORING
# ════════════════════════════════════════════════════════════════════════════

def onto_keywords_match(tokens, entry):
    """
    Cuenta keywords de la entrada presentes en el set de tokens.
    Keywords multipalabra requieren que TODOS sus tokens estén presentes.
    Usa TOKEN_REGEX precompilada [PERF-1].
    """
    kws = entry.get("keywords") or []
    if not kws:
        return 0
    count = 0
    for k in kws:
        sub = TOKEN_REGEX.findall(k.lower())
        if sub and all(s in tokens for s in sub):
            count += 1
    return count


def has_negative(tokens, entry):
    """True si alguna keyword negativa de la entrada aparece en los tokens."""
    for k in entry.get("keywords_negativas") or []:
        sub = TOKEN_REGEX.findall(k.lower())
        if sub and all(s in tokens for s in sub):
            return True
    return False


def label_words(label):
    """Set de tokens del texto del label. Usa TOKEN_REGEX [PERF-1]."""
    return set(TOKEN_REGEX.findall(label.lower()))


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13 — REGLAS DURAS (HARD RULES)
# ════════════════════════════════════════════════════════════════════════════

_CAPILLA_TOKENS   = {"capilla", "ermita", "iglesia", "oratorio", "culto", "altar", "retablo"}
_HISTORICO_TOKENS = {"medieval", "histórico", "historico", "antiguo", "siglo",
                     "patrimonio", "renacentista", "gótico", "gotico", "barroco",
                     "románico", "romanico", "arquitectura", "monumental"}
_CUEVA_TOKENS     = {"cueva", "cuevas", "gruta", "grutas", "caverna", "cavernas",
                     "espeleon", "espeleología", "espeleologia"}


def apply_hard_rules(name_clean, categoria_lower,
                     desc_html_raw, desc_raw,
                     tok_n, tok_d, tok_f):   # [API-1] tok_f como argumento
    """
    Reglas de negocio deterministas evaluadas antes del scoring.
    Devuelve (entry, explicacion) o (None, None).

    tok_n aquí es el set COMPLETO (sin filtrar complementos) para que las
    reglas puedan detectar palabras como 'hospital' o 'cueva' aunque estén
    en posición de complemento.

    HR-0  'Centro de interpretación' en nombre o desc → Centro de interpretación.
    HR-1  URL fexme.com en HTML crudo de desc → Ruta.
    HR-2  Edificio religioso + 'colegio' en nombre → Centro educativo.
    HR-3  Edificio religioso + 'hospital' en nombre → Capilla /
          Arquitectura vernácula / Conjunto artístico (según desc).
    HR-3b Hospital en nombre, cat vacía u 'otros'.
    HR-4  Espacio natural / Punto de ruta + cueva en nombre → Cueva o caverna.
    HR-5  Casa solariega → siempre Arquitectura vernácula.
    """

    # ── HR-0 ────────────────────────────────────────────────────────────
    if ({"centro", "interpretación"}.issubset(tok_f)
            or {"centro", "interpretacion"}.issubset(tok_f)):
        entry = ontology_entry("centro de interpretación")
        if entry:
            origin = "nombre" if (
                {"centro", "interpretación"}.issubset(tok_n)
                or {"centro", "interpretacion"}.issubset(tok_n)
            ) else "descripción"
            return entry, f"HR-0: 'centro de interpretación' en {origin}"

    # ── HR-1: fexme.com en HTML crudo → Ruta ────────────────────────────
    if "fexme.com" in desc_html_raw.lower():
        entry = ontology_entry("ruta")
        if entry:
            return entry, "HR-1: URL fexme.com en descripción (HTML crudo) → Ruta"

    # ── HR-2 ────────────────────────────────────────────────────────────
    if categoria_lower == "edificio religioso" and "colegio" in tok_n:
        entry = ontology_entry("centro educativo")
        if entry:
            return entry, "HR-2: edificio religioso + 'colegio' en nombre → Centro educativo"

    # ── HR-3 ────────────────────────────────────────────────────────────
    if categoria_lower == "edificio religioso" and "hospital" in tok_n:
        if tok_d & _CAPILLA_TOKENS:
            entry = ontology_entry("capilla")
            if entry:
                return entry, f"HR-3: edificio religioso + hospital + {tok_d & _CAPILLA_TOKENS} → Capilla"
        if tok_d & _HISTORICO_TOKENS:
            entry = ontology_entry("arquitectura vernácula")
            if entry:
                return entry, f"HR-3: edificio religioso + hospital + {tok_d & _HISTORICO_TOKENS} → Arq. vernácula"
        entry = ontology_entry("conjunto de interés artístico")
        if entry:
            return entry, "HR-3: edificio religioso + hospital (sin señales) → Conjunto artístico"

    # ── HR-3b ───────────────────────────────────────────────────────────
    if categoria_lower in ("", "otros") and "hospital" in tok_n:
        if tok_d & _CAPILLA_TOKENS:
            entry = ontology_entry("capilla")
            if entry:
                return entry, "HR-3b: hospital + tokens religiosos → Capilla"
        entry = ontology_entry("conjunto de interés artístico")
        if entry:
            return entry, "HR-3b: hospital en nombre (cat vacía/otros) → Conjunto artístico"

    # ── HR-4 ────────────────────────────────────────────────────────────
    if categoria_lower in ("espacio natural", "punto de ruta"):
        matched = tok_n & _CUEVA_TOKENS
        if matched:
            entry = ontology_entry("cueva o caverna")
            if entry:
                return entry, f"HR-4: {categoria_lower} + {matched} → Cueva o caverna"

    # ── HR-5 ────────────────────────────────────────────────────────────
    if categoria_lower == "casa solariega":
        entry = ontology_entry("arquitectura vernácula")
        if entry:
            return entry, "HR-5: categoría 'casa solariega' → Arquitectura vernácula"

    return None, None


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 14 — FUNCIÓN PRINCIPAL DE CLASIFICACIÓN
# ════════════════════════════════════════════════════════════════════════════

# Bonus extra para labels con acuerdo nombre+mapping [FIX-8]
_DOMINANT_BONUS = 10


def classify(nombre, categoria, descripcion):
    """
    Clasifica un punto de interés turístico.

    Pipeline:
      1.  Normalización de tipos
      2.  Limpieza HTML + expansión de abreviaturas (guarda desc_html_raw)
      3.  Eliminación de dirección postal del nombre
      4.  Tokenización completa
      5.  [Cambio C] Separación head/complemento genitivo
      6.  [Cambio A] Primera palabra significativa + verificación semántica
      7.  Candidatos del mapping
      8.  Reglas duras (cortocircuito si aplican)
      9.  [FIX-MAP-2] Cortocircuito determinista si mapping da 1 label
     10.  Scoring ontología:
            · SEÑAL 1 in_map → 15 pts + name_map_labels  [FIX-8]
            · SEÑAL 1 multipalabra → +2 bonus  [Cambio B]
            · SEÑAL 1 monopalabra primera → +4 bonus  [Cambio A]
            · SEÑAL 1 monopalabra posición trasera → 3 pts  [FIX-9]
            · kw_f calculado sobre tok_n_clean (sin complementos) [Cambio C]
     11.  Scoring aliases (sobre tok_n_clean)
     12.  Ancla de categoría  [FIX-10]
     13.  Bonus dominante nombre+mapping  [FIX-8]
     14.  Desambiguaciones manuales
     15.  Fallback / "No clasificado"
     16.  Selección del ganador y confianza
    """

    # ── 1. Normalización de tipos ───────────────────────────────────────
    if categoria is None or (isinstance(categoria, float) and pd.isna(categoria)):
        categoria = ""
    else:
        categoria = str(categoria)
    categoria_lower = categoria.strip().lower()

    # ── 2. Limpieza ─────────────────────────────────────────────────────
    name_raw      = expand_abbrev(strip_html(nombre if isinstance(nombre, str) else ""))
    desc_html_raw = descripcion if isinstance(descripcion, str) else ""
    desc_raw      = expand_abbrev(strip_html(desc_html_raw))

    # ── 3. Eliminación de dirección postal ──────────────────────────────
    name_clean = strip_address_suffix(name_raw, categoria_lower)

    # ── 4. Tokenización completa ─────────────────────────────────────────
    # tok_n: set COMPLETO del nombre (para hard rules, negativos y tok_f)
    # name_tokens_ordered: lista ordenada (para posición y head/complement)
    tok_n               = tokenize(name_clean)
    tok_d               = tokenize(desc_raw)
    tok_f               = tok_n | tok_d
    name_tokens_ordered = TOKEN_REGEX.findall(name_clean.lower())

    # ── 5. Separación head/complemento  [Cambio C] ───────────────────────
    # tok_n_clean: solo tokens del "head" del nombre (antes del genitivo).
    # Estos son los que se usan en SEÑAL 1, 2, 3 del scoring.
    # tok_n original sigue disponible para hard rules y tok_f.
    tok_n_head, tok_n_complement = split_name_head_complement(name_tokens_ordered)
    tok_n_clean = tok_n_head

    # ── 6. Primera palabra significativa  [Cambio A] ─────────────────────
    # Se calcula una vez antes del loop para eficiencia.
    # first_tok = None si la primera palabra no es semántica (topónimo, etc.)
    _raw_first = first_significant_token(name_tokens_ordered)
    first_tok  = _raw_first if is_first_tok_semantic(_raw_first) else None

    # ── 7. Candidatos del mapping ────────────────────────────────────────
    mapping_labels, ruta = mapping_candidate_labels(categoria)

    # ── 8. Reglas duras ──────────────────────────────────────────────────
    # Reciben tok_n COMPLETO (sin filtrar) para detectar palabras clave
    # aunque estén en posición de complemento ('hospital', 'cueva'…).
    hard_entry, hard_reason = apply_hard_rules(
        name_clean, categoria_lower,
        desc_html_raw, desc_raw,
        tok_n, tok_d, tok_f,
    )
    if hard_entry:
        return build_result(hard_entry, 0.92, False, False, hard_reason)

    # ── 9. Cortocircuito determinista  [FIX-MAP-2] ───────────────────────
    # Si el mapping resuelve a exactamente 1 label, la clasificación es
    # determinista: no tiene sentido pasar por el scoring.
    if len(mapping_labels) == 1:
        sole_label = next(iter(mapping_labels))
        entry = ontology_entry(sole_label)
        if entry and not has_negative(tok_f, entry):
            return build_result(
                entry, 0.95, False, False,
                f"Mapping determinista (1 label): '{categoria}' → {sole_label}"
            )

    # ── 10. Scoring: ontología ───────────────────────────────────────────
    scores: dict         = {}
    name_map_labels: set = set()   # labels con acuerdo nombre+mapping [FIX-8]

    def add(label, pts, reason):
        if label not in scores:
            scores[label] = [pts, [reason]]
        else:
            scores[label][0] += pts
            scores[label][1].append(reason)

    for it in ONTO:
        lab    = it["label"]
        lw     = label_words(lab)
        in_map = lab in mapping_labels

        if has_negative(tok_f, it):
            continue

        # Keywords calculadas sobre tok_n_clean (sin complemento) [Cambio C]
        kw_n = onto_keywords_match(tok_n_clean, it)
        kw_d = onto_keywords_match(tok_d, it)
        # kw_f: nombre_clean OR descripción (no complemento) [FIX-5]
        kw_f = onto_keywords_match(tok_n_clean | tok_d, it)

        # ── SEÑAL 1 — Palabras del label ⊆ tokens del nombre (head) ──────
        # Usa tok_n_clean para no matchear con el complemento genitivo.
        if lw and lw.issubset(tok_n_clean):
            if in_map:
                # [FIX-8] Acuerdo nombre+mapping: 15 pts y se marca como dominante
                name_map_labels.add(lab)
                add(lab, 15, f"'{lab}' en nombre (mapping)")

            elif len(lw) > 1:
                # [Cambio B] Label multipalabra sin mapping: bonus +2 por especificidad
                # Un label de 2+ palabras que matchea completo en el nombre es
                # mucho más preciso que dos labels monopalabra independientes.
                bonus_multi = 2
                add(lab, 7 + bonus_multi, f"'{lab}' en nombre (multipalabra)")

            else:
                # Label monopalabra sin mapping
                lw_token = next(iter(lw))
                leading  = lw_token in name_tokens_ordered[:2]

                # [Cambio A] Bonus si es la primera palabra significativa del nombre
                # Solo aplica si:
                #   · first_tok no es None (fue verificado como semántico)
                #   · El token del label coincide con first_tok
                #   · El label NO es multipalabra (ya cubierto arriba)
                first_bonus = 4 if (first_tok and lw_token == first_tok) else 0

                # [FIX-9] Posición: tokens en la parte trasera del nombre suelen
                # ser topónimos, no descriptores del POI.
                pts = (7 if leading else 3) + first_bonus

                tag_parts = []
                if first_bonus:
                    tag_parts.append("+primera palabra")
                if not leading:
                    tag_parts.append("posición trasera→topónimo?")
                tag = f" ({', '.join(tag_parts)})" if tag_parts else ""
                add(lab, pts, f"'{lab}' en nombre{tag}")

            continue  # No seguir evaluando señales para este label

        # ── SEÑAL 2 — ≥2 keywords en nombre_clean O descripción ─────────
        if kw_f >= 2:
            pts = 8 if in_map else 5
            add(lab, pts,
                f"keywords en nombre/desc ({kw_f})" + (" (mapping)" if in_map else ""))
            continue

        # ── SEÑAL 3 — 1 keyword en nombre_clean O descripción ────────────
        if kw_f == 1 and lw:
            if len(lw) == 1:
                pts = 9 if in_map else 6
                add(lab, pts,
                    f"keyword '{next(iter(lw))}' en nombre/desc" + (" (mapping)" if in_map else ""))
            else:
                # [Cambio B] 1 keyword de label multipalabra: bonus +2 por especificidad
                kw_multi_bonus = 2
                pts = (5 if in_map else 3) + kw_multi_bonus
                add(lab, pts,
                    f"1 keyword multipalabra en nombre/desc" + (" (mapping)" if in_map else ""))
            continue

        # ── SEÑAL 4 — Solo descripción (señal débil de respaldo) ─────────
        # Usa tok_f (que incluye tok_n completo) para descripción, pero
        # la señal fuerte de nombre ya usa tok_n_clean.
        if lw.issubset(tok_f) and not lw.issubset(tok_n_clean):
            if in_map and len(lw) >= 2:
                add(lab, 2, f"'{lab}' en descripción (mapping, multipalabra)")
            elif len(lw) >= 3:
                add(lab, 1, f"'{lab}' en descripción (multipalabra)")

        no_mapping = not mapping_labels
        if kw_d >= 2 and kw_n == 0:
            if in_map:
                add(lab, 2, f"keywords en desc ({kw_d}) (mapping)")
            elif no_mapping and kw_d >= 3:
                add(lab, 2, f"keywords en desc ({kw_d}) (sin mapping)")
        if no_mapping and lw.issubset(tok_d) and not lw.issubset(tok_n_clean) and len(lw) >= 2:
            add(lab, 2, f"'{lab}' en descripción (sin mapping)")

    # ── 11. Scoring: aliases (sobre tok_n_clean)  [Cambio C] ─────────────
    # Los aliases también usan el head del nombre sin complementos genitivos.
    for alias, target in ALIASES.items():
        alias_tokens = set(TOKEN_REGEX.findall(alias.lower()))
        if not alias_tokens:
            continue
        if alias_tokens.issubset(tok_n_clean):
            entry = ontology_entry(target)
            if entry and not has_negative(tok_f, entry):
                in_map = entry["label"] in mapping_labels
                bonus  = 1 if len(alias_tokens) > 1 else 0
                pts    = (9 if in_map else 7) + bonus
                add(entry["label"], pts, f"alias '{alias}'→{entry['label']}")

    # ── 12. Ancla de categoría  [FIX-10] ─────────────────────────────────
    # Si el mapping resuelve a ≤5 labels concretos y ninguno recibió señal
    # de texto, darles 8 pts base para que no pierdan ante topónimos
    # coincidentales (que sin mapping obtienen 3–7 pts).
    if mapping_labels and len(mapping_labels) <= 5:
        for lab in mapping_labels:
            if lab not in scores:
                entry_check = ontology_entry(lab)
                if entry_check and not has_negative(tok_f, entry_check):
                    add(lab, 8, "ancla de categoría (mapping directo, sin señal de texto)")

    # ── 13. Bonus dominante para nombre+mapping  [FIX-8] ─────────────────
    # Labels con SEÑAL 1 + in_map reciben _DOMINANT_BONUS pts adicionales.
    # Hace imposible que señales puramente de descripción los superen.
    for lab in name_map_labels:
        if lab in scores:
            scores[lab][0] += _DOMINANT_BONUS
            scores[lab][1].append(f"+{_DOMINANT_BONUS} (acuerdo nombre+mapping)")

    # ── 14. Desambiguaciones manuales ─────────────────────────────────────
    if "Plaza de toros" in scores and "Plaza" in scores \
            and {"plaza", "toros"}.issubset(tok_n_clean):
        scores["Plaza"][0] = 0

    if "Bar de vinos" in scores and "Bar" in scores and "vinos" in tok_n_clean:
        scores["Bar"][0] = 0

    if "Parque natural" in scores and "Parque infantil" in scores:
        if "natural"    in tok_n_clean: scores["Parque infantil"][0] = 0
        elif "infantil" in tok_n_clean: scores["Parque natural"][0]  = 0

    if "Mirador natural" in scores and "Mirador urbano" in scores:
        if "urbano" in tok_n_clean: scores["Mirador natural"][0] = 0
        else:                       scores["Mirador urbano"][0]  = 0

    if "Cine" in scores and "cine" not in tok_n_clean:
        scores["Cine"][0] = 0

    scores = {k: v for k, v in scores.items() if v[0] > 0}

    # ── 15. Fallback ──────────────────────────────────────────────────────
    if not scores:
        default = DEFAULTS.get(categoria_lower)
        if default:
            entry = ontology_entry(default)
            if entry:
                return build_result(
                    entry, 0.5, False, False,
                    f"Sin señales específicas; default de '{categoria}' → {entry['label']}"
                )
        return build_unclassified()

    # ── 16. Selección del ganador y confianza ─────────────────────────────
    ranked       = sorted(scores.items(), key=lambda kv: -kv[1][0])
    best_label,  (best_score,  best_reasons) = ranked[0]
    second_label = ranked[1][0]    if len(ranked) > 1 else None
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0

    ambig = bool(
        second_label
        and (best_score - second_score) < 2
        and best_score < 10
    )

    in_map_final = best_label in mapping_labels
    conflicto    = bool(ruta and not in_map_final)

    conf = min(0.98, 0.4 + best_score * 0.05)
    if ambig:     conf = max(0.35, conf - 0.10)
    if conflicto: conf = max(0.45, conf - 0.05)

    entry       = ontology_entry(best_label)
    explicacion = "; ".join(best_reasons[:3])
    if ambig and second_label:
        explicacion += f" | alt: {second_label}"
    if conflicto:
        explicacion = f"[Conflicto con '{categoria}'] " + explicacion

    return build_result(entry, conf, conflicto, ambig, explicacion)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 15 — CONSTRUCTORES DE RESULTADO
# ════════════════════════════════════════════════════════════════════════════

def build_result(entry, conf, conflicto, ambig, explicacion):
    return {
        "categoria_final":     entry.get("label", ""),
        "Nivel 1":             entry.get("nivel_2") or "",
        "Nivel 2":             entry.get("nivel_3") or "",
        "Nivel 3":             entry.get("nivel_4") or "",
        "Nivel 4":             entry.get("nivel_5") or "",
        "confianza":           round(conf, 2),
        "conflicto_semantico": bool(conflicto),
        "ambiguedad":          bool(ambig),
        "sin_clasificar":      False,
        "explicacion":         explicacion[:500],
    }


def build_unclassified():
    return {
        "categoria_final":     "No clasificado",
        "Nivel 1": "", "Nivel 2": "", "Nivel 3": "", "Nivel 4": "",
        "confianza":           0.2,
        "conflicto_semantico": False,
        "ambiguedad":          False,
        "sin_clasificar":      True,
        "explicacion":         "Sin información suficiente para clasificar.",
    }


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 16 — DETECCIÓN DINÁMICA DE COLUMNAS Y MAIN
# ════════════════════════════════════════════════════════════════════════════

COLUMN_ROLES = {
    "nombre":      ["titulo", "nombre", "name", "denominacion", "title"],
    "categoria":   ["categoria", "category", "tipo_categoria", "tipo", "type"],
    "descripcion": ["descripcion", "descripción", "long_description",
                    "description", "texto", "text", "resumen"],
}


def detect_columns(df_columns):
    col_lower = {c.strip().lower(): c for c in df_columns}
    result = {}
    for role, candidates in COLUMN_ROLES.items():
        found = None
        for cand in candidates:
            if cand.lower() in col_lower:
                found = col_lower[cand.lower()]
                break
        result[role] = found
        if found is None:
            print(f"[WARN] Columna para '{role}' no encontrada. "
                  f"Buscados: {candidates}. Disponibles: {list(col_lower.keys())}")
    return result


def main():
    # Lectura con fallback de codificación  [FIX-2]
    try:
        df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
    except UnicodeDecodeError:
        print("[WARN] UTF-8 falló, reintentando con latin-1…")
        df = pd.read_csv(CSV_PATH, sep=";", encoding="latin-1")

    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] Columnas CSV: {list(df.columns)}")

    cols = detect_columns(df.columns)
    print(f"[INFO] Mapeo: nombre='{cols['nombre']}' | "
          f"categoria='{cols['categoria']}' | descripcion='{cols['descripcion']}'")

    # [PERF-3] itertuples(name=None): tuplas planas, acceso por índice numérico.
    col_pos = {
        role: (list(df.columns).index(col_name) + 1
               if col_name is not None and col_name in df.columns
               else None)
        for role, col_name in cols.items()
    }

    def _tval(tup, pos):
        """Acceso seguro a columna de itertuples por posición."""
        if pos is None:
            return ""
        val = tup[pos]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return str(val)

    rows = [
        classify(
            _tval(r, col_pos["nombre"]),
            _tval(r, col_pos["categoria"]),
            _tval(r, col_pos["descripcion"]),
        )
        for r in df.itertuples(index=True, name=None)   # [PERF-3]
    ]

    out   = pd.DataFrame(rows)
    final = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    output_cols = [
        "categoria_final", "Nivel 1", "Nivel 2", "Nivel 3", "Nivel 4",
        "confianza", "conflicto_semantico", "ambiguedad",
        "sin_clasificar", "explicacion",
    ]
    existing_orig    = [c for c in df.columns    if c in final.columns]
    existing_classif = [c for c in output_cols   if c in final.columns]
    final = final[existing_orig + existing_classif]

    print(f"\n[INFO] Filas clasificadas: {len(final)}")
    print("=== Distribución categoria_final (top 35) ===")
    print(final["categoria_final"].value_counts().head(35))
    print(f"\nconflicto_semantico: {final['conflicto_semantico'].sum()}")
    print(f"ambiguedad:          {final['ambiguedad'].sum()}")
    print(f"sin_clasificar:      {final['sin_clasificar'].sum()}")
    print(f"confianza media:     {final['confianza'].mean():.3f}")
    return final


if __name__ == "__main__":
    final = main()
    final.to_excel("clasificacion_output.xlsx", index=False)
    print("Guardado en clasificacion_output.xlsx")