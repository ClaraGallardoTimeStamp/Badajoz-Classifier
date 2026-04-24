"""
Clasificador v5 — Correcciones y mejoras de robustez.

CAMBIOS RESPECTO A v2:
  [FIX-1]  Carga segura de ficheros con try/except (ya no rompe silenciosamente).
  [FIX-2]  Lectura del CSV con fallback a latin-1 si UTF-8 falla.
  [FIX-3]  Detección de columnas del CSV insensible a mayúsculas y espacios.
  [FIX-4]  strip_address_suffix(): elimina la dirección postal del nombre
           (ej: "CASA SOLARIEGA CALLE SANTA ANA 1" → "CASA SOLARIEGA"),
           excepto si la categoría de origen es "punto de ruta".
  [HR-1]   URL fexme.com/senderos en descripción → SIEMPRE Sendero.
  [HR-2]   Edificio religioso + "colegio" en nombre → SIEMPRE Centro educativo.
  [HR-3]   Edificio religioso + "hospital" en nombre → busca pistas en descripción
           para decidir entre Capilla, Arquitectura vernácula u Hospital público.
  [HR-4]   Espacio natural o Punto de ruta + cueva/gruta en nombre →
           SIEMPRE Cueva o caverna.
  [ALIAS]  Aliases construidos dinámicamente desde Ontology_SEGITTUR_2.json:\n           
  cada keyword del JSON se convierte en alias → label.\n           
  Ya no hay aliases hardcodeados: para mejorar el vocabulario\n           
  solo hay que editar el fichero JSON.
  [WARN]   expand_cluster() ya no falla silenciosamente cuando un cluster
           no produce hojas: emite un aviso por consola.
"""

import json
import re
import sys
import unicodedata
from functools import lru_cache
from html import unescape
import pandas as pd


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — RUTAS DE FICHEROS
# ════════════════════════════════════════════════════════════════════════════
# Centralizar las rutas aquí facilita cambiarlas sin tocar el resto del código.

ONTO_PATH    = 'Ontology_SEGITTUR_2.json'
ALIASES_PATH = 'Ontology_SEGITTUR_2.json'   # fuente de aliases (keywords)
MAPPING_PATH = 'Mapping.json'
CSV_PATH     = 'Servicios_Limpio.csv'

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — CARGA SEGURA DE DATOS  [FIX-1]
# ════════════════════════════════════════════════════════════════════════════
# En v2 los open() estaban a nivel de módulo sin manejo de errores:
# si un fichero faltaba o estaba corrupto, el script petaba con un traceback
# ininteligible. Ahora damos mensajes claros y paramos el proceso.

def _load_json(path, label):
    """Carga un JSON con mensajes de error comprensibles."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"[ERROR] No se encontró el fichero '{label}' en:\n  {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"[ERROR] El fichero '{label}' no es JSON válido:\n  {e}")


ONTO         = _load_json(ONTO_PATH,    "Ontología SEGITTUR")
ALIASES_LIST = _load_json(ALIASES_PATH, "Ontología de aliases (v2)")
MAPPING_LIST = _load_json(MAPPING_PATH, "Mapping de categorías")


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ÍNDICES EN MEMORIA
# ════════════════════════════════════════════════════════════════════════════
# Construimos diccionarios una sola vez al cargar el módulo para que las
# búsquedas sean O(1) en lugar de O(n) en cada clasificación.

# Diccionario: "categoria_origen_en_minúscula" → objeto ruta del mapping
MAPPING = {m['categoria_origen'].strip().lower(): m['ruta'] for m in MAPPING_LIST}

# Diccionario: "label_en_minúscula" → entrada completa de la ontología
LABEL_INDEX = {it['label'].strip().lower(): it for it in ONTO if it.get('label')}

# Índice adicional normalizado (sin acentos) para búsquedas tolerantes.
# Se usa solo como fallback cuando la búsqueda exacta falla.
# Se construye DESPUÉS de que normalize_text esté disponible (ver sección 7);
# por eso se llena con una función lazy en ontology_entry().
_LABEL_INDEX_NORM: dict = {}   # se llena en la primera llamada a ontology_entry()
_LABEL_INDEX_NORM_BUILT = False


def ontology_entry(label):
    """
    Devuelve la entrada de la ontología para un label dado.
    Busca en dos pasos:
      1. Búsqueda exacta insensible a mayúsculas (rápida, sin pérdida).
      2. Búsqueda normalizada sin acentos (fallback para textos del mapping
         que no tienen tilde: 'comisaria' → 'Comisaría').
    """
    global _LABEL_INDEX_NORM, _LABEL_INDEX_NORM_BUILT
    if not label:
        return None
    # Paso 1: búsqueda exacta (mayúsculas/minúsculas)
    result = LABEL_INDEX.get(label.strip().lower())
    if result:
        return result
    # Paso 2: búsqueda normalizada (sin acentos)
    # Construir el índice normalizado la primera vez que se necesite
    if not _LABEL_INDEX_NORM_BUILT:
        # normalize_text está definida en Sección 7; como este módulo
        # se ejecuta de arriba a abajo, al llegar aquí ya está disponible.
        _LABEL_INDEX_NORM = {
            normalize_text(k): v
            for k, v in LABEL_INDEX.items()
        }
        _LABEL_INDEX_NORM_BUILT = True
    return _LABEL_INDEX_NORM.get(normalize_text(label))


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — LIMPIEZA Y NORMALIZACIÓN DE TEXTO
# ════════════════════════════════════════════════════════════════════════════

def strip_html(s):
    """
    Elimina etiquetas HTML, entidades HTML (&amp;, &nbsp;…) y espacios extra.
    Si la entrada no es string, devuelve ''.
    """
    if not isinstance(s, str):
        return ''
    s = unescape(s)                       # &amp; → &, &nbsp; → espacio, etc.
    s = re.sub(r'<[^>]+>', ' ', s)        # <b>texto</b> → texto
    s = s.replace('\xa0', ' ')            # non-breaking space
    return re.sub(r'\s+', ' ', s).strip() # espacios múltiples → uno


def expand_abbrev(s):
    """
    Expande abreviaturas comunes de vías públicas.
    Esto es NECESARIO antes de tokenizar para que "C/ Mayor" matchee con
    el alias 'calle' y las keywords de la ontología.

    C/ → calle   |  Avda. → avenida  |  Pza. → plaza
    Av. → avenida|  Pl.   → plaza    |  Sra. → señora
    """
    if not isinstance(s, str):
        return ''
    s = re.sub(r'\b[Cc]/\s*',      'calle ',   s)
    s = re.sub(r'\b[Aa]vda\.?\s*', 'avenida ', s)
    s = re.sub(r'\b[Aa]v\.\s*',    'avenida ', s)
    s = re.sub(r'\b[Pp]za\.?\s*',  'plaza ',   s)
    s = re.sub(r'\b[Pp]l\.\s*',    'plaza ',   s)
    s = re.sub(r'\b[Ss]ra\.?\s*',  'señora ',  s)
    return s


# Palabras que indican el inicio de una dirección postal dentro del nombre.
# Se aplican DESPUÉS de expand_abbrev, por eso aquí ya están expandidas.
_ADDRESS_INDICATORS = (
    r'calle|avenida|plaza|paseo|bulevar|rua|'
    r'carretera|camino|travesía|travesia|callejón|callejon|'
    r'urbanización|urbanizacion|polígono|poligono|barrio|paraje'
)


def strip_address_suffix(name_expanded, categoria_lower):
    """
    [FIX-4] Elimina la parte de dirección postal que a veces se concatena
    al nombre del elemento en el CSV.

    Ejemplo:
      "CASA SOLARIEGA CALLE SANTA ANA 1"  →  "CASA SOLARIEGA"
      "ERMITA CALLE DEL PINO S/N"         →  "ERMITA"

    Lógica:
      Busca el primer indicador de vía que aparezca DESPUÉS de al menos
      una palabra (es decir, no al principio del nombre). Si lo encuentra,
      trunca el texto ahí.

    Excepción:
      Si la categoría de origen es "punto de ruta", el nombre PUEDE y DEBE
      contener "Calle de X", "Avenida de Y", etc., porque el punto ES esa vía.
      En ese caso no se modifica nada.
    """
    if categoria_lower == 'punto de ruta':
        return name_expanded   # Las rutas son vías: no tocar

    # Buscamos el indicador PRECEDIDO de al menos un carácter no-espacio
    # (para no truncar si el nombre EMPIEZA con la vía)
    pattern = r'(?<=\S)\s+(' + _ADDRESS_INDICATORS + r')\b'
    m = re.search(pattern, name_expanded, re.IGNORECASE)
    if m:
        return name_expanded[:m.start()].strip()
    return name_expanded


def tokenize(s):
    """
    Devuelve un set de tokens en minúscula.
    Solo letras (incluye vocales acentuadas, ü y ñ).
    Un set permite hacer operaciones de intersección/subconjunto eficientemente.
    """
    return set(re.findall(r"[a-záéíóúüñç]+", s.lower()))


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — ALIASES: sinónimos → label canónico de la ontología
# ════════════════════════════════════════════════════════════════════════════
# Los aliases se construyen DINÁMICAMENTE desde Ontology_SEGITTUR_2.json:
# cada keyword de cada entrada se convierte en un alias que apunta al label
# de esa entrada.  Para añadir o mejorar aliases solo hay que editar el JSON.
#
# Ejemplo en el JSON:
#   { "label": "Castillo", "keywords": ["castillo","fortaleza","fuerte"], … }
# Produce:
#   "castillo"  → "Castillo"
#   "fortaleza" → "Castillo"
#   "fuerte"    → "Castillo"
#
# Regla de precedencia:
#   Si una misma keyword aparece en varios labels, gana el label que aparece
#   ANTES en el fichero (el orden del JSON es significativo).
#   Esto permite priorizar deliberadamente ciertos labels poniendo su entrada
#   más arriba en el fichero.
#
# Nota sobre keywords multipalabra:
#   Si una keyword tiene varias palabras (ej: "parque nacional"), se registra
#   tal cual en el diccionario. En la fase de scoring (sección 11) se compara
#   token a token, así que el match requiere que TODOS los tokens de la keyword
#   estén en tok_n (comportamiento idéntico al que ya tenía onto_keywords_match).

def _build_aliases(alias_ontology):
    """
    Lee ALIASES_LIST (Ontology_SEGITTUR_2.json) y construye el diccionario
    de aliases.

    Proceso:
      Para cada entrada del JSON:
        Para cada keyword de esa entrada:
          Normalizar la keyword a minúscula y eliminar espacios extra.
          Si la keyword NO está ya registrada (precedencia: primera aparición),
          añadir  keyword_normalizada → label_de_la_entrada.

    El resultado es un dict:  str → str
      "ermita" → "Capilla"
      "fortaleza" → "Castillo"
      …
    """
    result = {}
    skipped = 0
    for entry in alias_ontology:
        label = (entry.get('label') or '').strip()
        if not label:
            continue
        for kw in (entry.get('keywords') or []):
            kw_norm = kw.strip().lower()
            if not kw_norm:
                continue
            if kw_norm not in result:
                result[kw_norm] = label
            else:
                skipped += 1   # ya existe con mayor precedencia
    print(f"[INFO] Aliases cargados desde JSON: {len(result)} "
          f"({skipped} keywords duplicadas ignoradas por precedencia)")
    return result


ALIASES = _build_aliases(ALIASES_LIST)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6 — DEFAULTS: categoría de origen → label cuando no hay señales
# ════════════════════════════════════════════════════════════════════════════
# Solo se usan como último recurso cuando el scoring no produce ningún
# candidato. La confianza resultante es deliberadamente baja (0.5).

DEFAULTS = {
    'edificio religioso':        'iglesia',
    'construcción civil':        'arquitectura vernácula',
    'monumento':                 'monumento',
    'espacio natural':           'parque natural',
    'castillos y fortalezas':    'castillo',
    'punto de ruta':             'ruta',
    'museo':                     'museo',
    'casa solariega':            'arquitectura vernácula',
    'parques y jardines':        'jardín',
    'zona arqueológica':         'yacimiento arqueológico',
    'paraje pintoresco':         'mirador natural',
    'arte rupestre':             'yacimiento arqueológico',
    'jardín histórico':          'jardín',
    'conjunto histórico':        'conjunto de interés artístico',
    'vía histórica':             'ruta',
    'escudos':                   'conjunto de interés artístico',
    'construcción militar':      'castillo',
    'oficina de turismo':        'oficina de turismo',
    'zona comercial':            'centro comercial',
    'otros núcleos de población':'destino turístico',
    # Categoría 'Otros' y vacía: fallback genérico patrimonial
    # Solo llega aquí si el scoring no encontró nada en nombre+descripción.
    'otros':                     'conjunto de interés artístico',
    '':                          'conjunto de interés artístico',

    # ── Categorías directas del CSV de servicios ──────────────────────────
    # Estas categorías son prácticamente labels 1:1 con la ontología.
    # Se usan como default de alta confianza cuando el scoring no aporta señales.
    'farmacia':                  'farmacia',
    'ayuntamiento':              'ayuntamiento',
    'supermercado':              'supermercado',
    'cajero':                    'cajero automático',
    'bomberos':                  'estación de bomberos',
    'instalación deportiva':     'polideportivo',
    'policía y/o guardía civil': 'comisaría de policía',
    'centro de salud':           'centro de atención primaria',
    'oficina de información turística': 'oficina de turismo',
    'taller de automóvil':       'taller mecánico',
    'consulta médica':           'hospital público',
    'estación de servicio':      'gasolinera',
    'educación':                 'centro educativo',
    # 'Cultura' es amplia: biblioteca, teatro, museo… el scoring decide.
    # Solo si no hay señal cae aquí con un fallback genérico.
    'cultura':                   'centro cultural',
}


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7 — NORMALIZACIÓN Y EXPANSIÓN DE CLUSTERS
# ════════════════════════════════════════════════════════════════════════════

def normalize_text(s):
    """
    Normalización para comparaciones robustas:
      1. Minúscula
      2. Elimina acentos (á→a, é→e, ó→o, ü→u, ñ→n…)
      3. Elimina espacios extra

    Esto permite que 'comisaria de policia' del mapping encuentre
    'Comisaría de policía' en la ontología, y viceversa.
    """
    s = s.strip().lower()
    # NFD descompone los caracteres acentuados en base + diacrítico
    # Mn = Mark, Nonspacing (los diacríticos: tildes, diéresis…)
    s = unicodedata.normalize('NFD', s)
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    # La ñ se descompone en n + combinador; tras el paso anterior ya es 'n'.
    # No hace falta caso especial.
    return re.sub(r'\s+', ' ', s).strip()


# Índice invertido normalizado de la ontología para expand_cluster:
#   normalize_text(nivel_3) → [labels hoja]
# Se construye UNA SOLA VEZ al cargar el módulo, no en cada llamada.
_CLUSTER_INDEX_N3: dict = {}   # nivel_3 normalizado → labels
_CLUSTER_INDEX_N2: dict = {}   # nivel_2 normalizado → labels (hojas con nivel_3 pero sin nivel_4)
for _it in ONTO:
    _n2 = normalize_text(_it.get('nivel_2') or '')
    _n3 = normalize_text(_it.get('nivel_3') or '')
    _n4 = normalize_text(_it.get('nivel_4') or '')
    if _n3 and _n4:                   # hoja colgando de nivel_3
        _CLUSTER_INDEX_N3.setdefault(_n3, []).append(_it['label'])
    if _n2 and _n3 and not _n4:       # hoja colgando de nivel_2 (intermedio sin nivel_4)
        _CLUSTER_INDEX_N2.setdefault(_n2, []).append(_it['label'])

# Caché de expand_cluster: evita recalcular y repetir WARNs para la misma
# categoría origen que aparece en miles de filas.
_EXPAND_CACHE: dict = {}
# Caché de warn ya emitidos: cada WARN se imprime solo una vez.
_WARNED_CLUSTERS: set = set()


def expand_cluster(label):
    """
    Dado un nombre de categoría padre (cluster), devuelve los labels hoja
    de la ontología que cuelgan de él.

    Mejoras respecto a v3:
      · Normaliza acentos antes de comparar (normalize_text):
        'comisaria de policia' encuentra 'Comisaría de policía'.
      · Usa índices precalculados (_CLUSTER_INDEX_N3/N2) → O(1) en lugar de O(n).
      · Caché en _EXPAND_CACHE: la misma categoría origen solo se calcula una vez.
      · El WARN se emite solo la PRIMERA vez que un cluster no produce hojas
        (evita el spam de miles de mensajes idénticos en el log).
    """
    global _WARNED_CLUSTERS
    lab = normalize_text(label)

    # Consultar caché primero
    if lab in _EXPAND_CACHE:
        return _EXPAND_CACHE[lab]

    # Buscar en los índices precalculados
    out = list(_CLUSTER_INDEX_N3.get(lab, []))
    if not out:
        out = list(_CLUSTER_INDEX_N2.get(lab, []))

    if not out and lab not in _WARNED_CLUSTERS:
        _WARNED_CLUSTERS.add(lab)
        print(f"[WARN] expand_cluster: '{label}' (norm: '{lab}') "
              f"no produjo hojas en la ontología.")

    _EXPAND_CACHE[lab] = out
    return out


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8 — CANDIDATOS DEL MAPPING
# ════════════════════════════════════════════════════════════════════════════

def mapping_candidate_labels(cat_origen):
    """
    Dado el valor de 'categoria' del CSV, devuelve:
      (set_de_labels_candidatos, objeto_ruta_del_mapping)

    Estos candidatos actúan como 'pistas previas': si el scoring elige
    uno de ellos, la confianza sube; si elige otro, se marca conflicto.

    Estrategia en dos pasos:
      1. Buscar nivel_2/3/4 del mapping en el LABEL_INDEX de la ontología.
         Si hay coincidencias directas ("concretos"), devolverlos y parar.
      2. Si no hay concretos, tratar esos valores como clusters (nodos padre)
         y expandirlos para obtener sus hojas.
    """
    if not cat_origen or (isinstance(cat_origen, float) and pd.isna(cat_origen)):
        return set(), None

    cat_origen = str(cat_origen)
    ruta = MAPPING.get(cat_origen.strip().lower())
    if not ruta:
        return set(), None   # Esta categoría no está en el mapping

    concrete      = set()    # Labels que existen directamente en la ontología
    cluster_names = []       # Nombres que parecen clusters (padres)

    for key in ('nivel_2', 'nivel_3', 'nivel_4'):
        vals = ruta.get(key, []) or []
        if isinstance(vals, str):
            vals = [vals]
        for v in vals:
            if not v:
                continue
            # Los valores pueden separarse con | o ,
            for part in re.split(r'[|,]', v):
                part = part.strip()
                if not part:
                    continue
                entry = ontology_entry(part)
                if entry:
                    concrete.add(entry['label'])   # ¡Existe como label!
                else:
                    cluster_names.append(part)     # Probablemente un cluster

    # Paso 1: hay concretos → devolverlos directamente (más preciso)
    if concrete:
        return concrete, ruta

    # Paso 2: no hay concretos → expandir clusters
    out = set()
    for c in cluster_names:
        for sub in expand_cluster(c):
            out.add(sub)
    return out, ruta


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9 — FUNCIONES DE SCORING
# ════════════════════════════════════════════════════════════════════════════

def onto_keywords_match(tokens, entry):
    """
    Cuenta cuántas keywords de la entrada de la ontología están presentes
    en el set de tokens dado.

    Una keyword puede ser multipalabra (ej: "parque nacional"); todos sus
    tokens deben estar presentes para que esa keyword cuente.

    Devuelve el número de keywords que hacen match (0 si ninguna).
    Si la entrada no tiene keywords, devuelve 0 (no añade ruido).
    """
    kws = entry.get('keywords', []) or []
    if not kws:
        return 0
    count = 0
    for k in kws:
        sub = re.findall(r"[a-záéíóúüñç]+", k.lower())
        if not sub:
            continue
        # Todos los sub-tokens de esta keyword deben estar presentes
        if all(s in tokens for s in sub):
            count += 1
    return count


def has_negative(tokens, entry):
    """
    Devuelve True si ALGUNA keyword negativa de la entrada está presente
    en los tokens del texto.

    Las keywords negativas sirven para descartar falsos positivos.
    Ejemplo: la entrada "Río" tiene como negativa "plaza" para evitar
    que "Plaza del Río" se clasifique como Río.
    """
    for k in entry.get('keywords_negativas', []) or []:
        sub = re.findall(r"[a-záéíóúüñç]+", k.lower())
        if sub and all(s in tokens for s in sub):
            return True
    return False


def label_words(label):
    """
    Devuelve el set de tokens del texto del label.
    Usado para comprobar si las palabras del label aparecen en el nombre.
    """
    return set(re.findall(r"[a-záéíóúüñç]+", label.lower()))


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10 — REGLAS DURAS (HARD RULES)  [HR-1 a HR-4]
# ════════════════════════════════════════════════════════════════════════════
# Las reglas duras se evalúan ANTES del scoring y devuelven resultado
# inmediato con confianza alta. Son condiciones de negocio explícitas
# que el scoring genérico no puede manejar bien.

# Tokens que en la descripción apuntan a que un edificio religioso
# con "hospital" en el nombre es en realidad una capilla/oratorio.
_CAPILLA_TOKENS   = {'capilla', 'ermita', 'iglesia', 'oratorio', 'culto', 'altar', 'retablo'}
# Tokens que sugieren que es un edificio histórico (no sanitario moderno)
_HISTORICO_TOKENS = {'medieval', 'histórico', 'historico', 'antiguo', 'siglo',
                     'patrimonio', 'renacentista', 'gótico', 'gotico', 'barroco',
                     'románico', 'romanico', 'arquitectura', 'monumental'}
# Tokens de cueva/gruta para las reglas HR-4
_CUEVA_TOKENS     = {'cueva', 'cuevas', 'gruta', 'grutas', 'caverna', 'cavernas',
                     'espeleon', 'espeleología', 'espeleologia'}


def apply_hard_rules(name_clean, categoria_lower, desc_raw, tok_n, tok_d):
    tok_f = tok_n | tok_d  # tokens combinados de nombre + descripción para reglas que buscan en ambos
    """
    Evalúa las reglas de negocio de alta prioridad en orden.
    Devuelve (entry_ontología, texto_explicación) si alguna dispara,
    o (None, None) si ninguna aplica.

    ─────────────────────────────────────────────────────────────────────
    HR-0  'Centro de interpretación' en nombre o descripción → Centro de
          interpretación. Prioridad máxima: prevalece sobre cualquier
          otra señal o categoría de origen.
    ─────────────────────────────────────────────────────────────────────
    HR-1  URL fexme.com/senderos en descripción → Sendero
          La presencia de esta URL identifica sin ambigüedad un sendero
          del portal extremeño de rutas.
    ─────────────────────────────────────────────────────────────────────
    HR-2  Edificio religioso + "colegio" en nombre → Centro educativo
          Los colegios religiosos (Colegio Sagrado Corazón, etc.) a veces
          entran en el CSV con categoría "Edificio religioso". El nombre
          siempre lo aclara.
    ─────────────────────────────────────────────────────────────────────
    HR-3  Edificio religioso + "hospital" en nombre
          El scoring genérico suele elegir Iglesia o Capilla por el
          mapping de "edificio religioso". Aquí forzamos una búsqueda
          activa en la descripción para discriminar:
            · Tokens de capilla/culto   → Capilla
            · Tokens históricos         → Arquitectura vernácula
            · Sin señales claras        → dejamos pasar al scoring
    ─────────────────────────────────────────────────────────────────────
    HR-4  Espacio natural / Punto de ruta + cueva/gruta en nombre →
          Cueva o caverna
          Los defaults de esas dos categorías son "Parque natural" y "Ruta"
          respectivamente. Si el nombre incluye cueva/gruta, el elemento
          es claramente una cueva aunque su categoría de origen sea otra.
    ─────────────────────────────────────────────────────────────────────
    """

    # ── HR-0: "Centro de interpretación" en nombre o descripción ────────
    # Esta regla tiene la máxima prioridad (se evalúa antes que todas las
    # demás) porque "centro de interpretación" es una etiqueta muy específica
    # que no debe perder frente a ninguna otra señal.
    #
    # Detectamos dos variantes:
    #   · Tokens completos en nombre o descripción:
    #       {'centro', 'interpretación'} ⊆ tok_n  o  ⊆ tok_f
    #   · La cadena literal (con/sin tilde) en el texto crudo:
    #       "centro de interpretacion" / "centre d'interpretació" etc.
    #
    # Usamos tok_f (nombre + descripción) para que baste con que aparezca
    # en cualquiera de los dos campos.
    _ci_tokens = {'centro', 'interpretación'} | {'centro', 'interpretacion'}
    if (
        {'centro', 'interpretación'}.issubset(tok_f)
        or {'centro', 'interpretacion'}.issubset(tok_f)
    ):
        entry = ontology_entry('centro de interpretación')
        if entry:
            origin = 'nombre' if (
                {'centro', 'interpretación'}.issubset(tok_n)
                or {'centro', 'interpretacion'}.issubset(tok_n)
            ) else 'descripción'
            return entry, f"HR-0: 'centro de interpretación' detectado en {origin} → Centro de interpretación"

    # ── HR-1: URL fexme.com/senderos ────────────────────────────────────
    # Buscamos la URL en el texto crudo de la descripción (más fiable que tokens)
    if 'fexme.com' in desc_raw.lower():
        entry = ontology_entry('sendero')
        if entry:
            return entry, "HR-1: URL fexme.com en descripción → Sendero"

    # ── HR-2: Edificio religioso + "colegio" ────────────────────────────
    if categoria_lower == 'edificio religioso' and 'colegio' in tok_n:
        entry = ontology_entry('centro educativo')
        if entry:
            return entry, "HR-2: edificio religioso + 'colegio' en nombre → Centro educativo"

    # ── HR-3: Edificio religioso + "hospital" ───────────────────────────
    # Un elemento categorizado como "edificio religioso" que contiene la
    # palabra "hospital" en el nombre es un hospital histórico-religioso
    # (Hospital de la Caridad, Hospital de Santiago…), NO un hospital
    # sanitario moderno. Por tanto NUNCA debe clasificarse como
    # "Hospital público" ni "Hospital privado".
    #
    # Orden de preferencia:
    #   1. Tokens de capilla/culto en descripción → Capilla
    #   2. Tokens históricos en descripción      → Arquitectura vernácula
    #   3. Sin señales claras                    → Conjunto de interés artístico
    #      (fallback seguro: es un edificio singular con valor patrimonial)
    if categoria_lower == 'edificio religioso' and 'hospital' in tok_n:
        # ¿La descripción habla de culto / elementos religiosos?
        if tok_d & _CAPILLA_TOKENS:
            matched = tok_d & _CAPILLA_TOKENS
            entry = ontology_entry('capilla')
            if entry:
                return entry, (f"HR-3: edificio religioso + hospital + "
                               f"tokens religiosos en desc {matched} → Capilla")
        # ¿La descripción contextualiza como edificio histórico?
        if tok_d & _HISTORICO_TOKENS:
            matched = tok_d & _HISTORICO_TOKENS
            entry = ontology_entry('arquitectura vernácula')
            if entry:
                return entry, (f"HR-3: edificio religioso + hospital + "
                               f"contexto histórico en desc {matched} → Arquitectura vernácula")
        # Sin señales claras → Conjunto de interés artístico como fallback
        # patrimonial seguro (evita que el alias 'hospital' dispare
        # 'hospital público', que sería semánticamente incorrecto aquí).
        entry = ontology_entry('conjunto de interés artístico')
        if entry:
            return entry, ("HR-3: edificio religioso + hospital sin señales "
                           "adicionales → Conjunto de interés artístico")

    # ── HR-3b: Hospital en nombre, categoría vacía u 'otros' ───────────
    # 'Hospital de San Miguel', 'Hospital de Santa Elena'…
    # Son hospitales histórico-religiosos catalogados como 'Otros' o sin
    # categoría. Miramos la descripción para decidir:
    #   · Tokens de capilla/culto  → Capilla
    #   · Tokens históricos        → Conjunto de interés artístico
    #   · Sin señales              → Conjunto de interés artístico (fallback)
    if categoria_lower in ('', 'otros') and 'hospital' in tok_n:
        if tok_d & _CAPILLA_TOKENS:
            entry = ontology_entry('capilla')
            if entry:
                return entry, ("HR-3b: hospital + tokens religiosos en desc → Capilla")
        entry = ontology_entry('conjunto de interés artístico')
        if entry:
            return entry, ("HR-3b: hospital en nombre (cat vacía/otros) + "
                           f"contexto desc → Conjunto de interés artístico")

    # ── HR-4: Cueva en nombre cuando la categoría dificulta el scoring ──
    # "espacio natural" tiene default "parque natural" y el mapping no incluye cueva.
    # "punto de ruta" tiene default "ruta".  Ambos enmascaran las cuevas.
    if categoria_lower in ('espacio natural', 'punto de ruta'):
        if tok_n & _CUEVA_TOKENS:
            matched = tok_n & _CUEVA_TOKENS
            entry = ontology_entry('cueva o caverna')
            if entry:
                return entry, (f"HR-4: categoría '{categoria_lower}' + "
                               f"{matched} en nombre → Cueva o caverna")

    return None, None   # Ninguna regla disparó


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11 — FUNCIÓN PRINCIPAL DE CLASIFICACIÓN
# ════════════════════════════════════════════════════════════════════════════

def classify(nombre, categoria, descripcion):
    """
    Clasifica un punto de interés turístico y devuelve un diccionario con:
      categoria_final, Nivel 1-4, confianza,
      conflicto_semantico, ambiguedad, sin_clasificar, explicacion

    Pipeline de ejecución (en orden):
      1. Normalización de tipos
      2. Limpieza HTML + expansión de abreviaturas
      3. Eliminación de dirección postal del nombre  [FIX-4]
      4. Tokenización
      5. Obtención de candidatos del mapping
      6. Reglas duras (retorno inmediato si aplican)  [HR-1 a HR-4]
      7. Scoring: recorrido de la ontología
      8. Scoring: aliases
      9. Desambiguaciones manuales
     10. Fallback a defaults / "No clasificado"
     11. Selección del ganador y cálculo de confianza
    """

    # ── 1. Normalización de tipos ───────────────────────────────────────
    if categoria is None or (isinstance(categoria, float) and pd.isna(categoria)):
        categoria = ''
    else:
        categoria = str(categoria)
    categoria_lower = categoria.strip().lower()

    # ── 2. Limpieza HTML + expansión de abreviaturas ────────────────────
    name_raw = expand_abbrev(strip_html(nombre if isinstance(nombre, str) else ''))
    desc_raw = expand_abbrev(strip_html(descripcion if isinstance(descripcion, str) else ''))

    # ── 3. Eliminación de dirección postal del nombre  [FIX-4] ──────────
    # "CASA SOLARIEGA CALLE SANTA ANA 1" → "CASA SOLARIEGA"
    # "ERMITA CALLE DEL PINO" → "ERMITA"
    # Excepción: si la categoría es "punto de ruta", no tocamos el nombre.
    name_clean = strip_address_suffix(name_raw, categoria_lower)

    # ── 4. Tokenización ─────────────────────────────────────────────────
    tok_n = tokenize(name_clean)    # tokens del nombre limpio
    tok_d = tokenize(desc_raw)      # tokens de la descripción
    tok_f = tok_n | tok_d           # unión: para checks de negativos y desc

    # ── 5. Candidatos del mapping ────────────────────────────────────────
    mapping_labels, ruta = mapping_candidate_labels(categoria)

    # ── 6. Reglas duras ──────────────────────────────────────────────────
    hard_entry, hard_reason = apply_hard_rules(
        name_clean, categoria_lower, desc_raw, tok_n, tok_d
    )
    if hard_entry:
        # Confianza alta (0.92) porque la regla es determinista
        return build_result(hard_entry, 0.92, False, False, hard_reason)

    # ── 7. Scoring: recorrido de la ontología ────────────────────────────
    # Para cada label de la ontología calculamos una puntuación según
    # cuántas señales del texto coinciden con él.
    #
    # Jerarquía de señales (de más a menos peso):
    #   · Las palabras del label están en el nombre    → 7-10 pts
    #   · Múltiples keywords en el nombre             → 5-8  pts
    #   · Keyword + label monopalabra en nombre       → 6-9  pts
    #   · Señales solo en descripción (muy débil)     → 1-2  pts
    #
    # El modificador "+in_map" añade puntos extra si el label coincide
    # con los candidatos del mapping (refuerzo por coherencia semántica).

    scores = {}

    def add(label, pts, reason):
        """Acumula puntuación y razonamiento para un label candidato."""
        if label not in scores:
            scores[label] = [pts, [reason]]
        else:
            scores[label][0] += pts
            scores[label][1].append(reason)

    for it in ONTO:
        lab    = it['label']
        lw     = label_words(lab)
        in_map = lab in mapping_labels

        # Descarte por keyword negativa: si el texto contiene una señal
        # que contradice este label, lo ignoramos completamente.
        if has_negative(tok_f, it):
            continue

        kw_n = onto_keywords_match(tok_n, it)   # keywords en NOMBRE
        kw_d = onto_keywords_match(tok_d, it)   # keywords en DESCRIPCIÓN

        # SEÑAL 1 — Palabras del label ⊆ tokens del nombre
        # La señal más fiable: "Castillo de X" → label "Castillo" matchea.
        if lw and lw.issubset(tok_n):
            pts = 10 if in_map else 7
            add(lab, pts, f"'{lab}' en nombre" + (" (mapping)" if in_map else ""))
            continue   # No seguimos evaluando señales para este label

        # SEÑAL 2 — Múltiples keywords de la ontología en el nombre
        # Ej: keywords ["alcázar","castillo"] → ambas en el nombre.
        if kw_n >= 2:
            pts = 8 if in_map else 5
            add(lab, pts, f"keywords en nombre ({kw_n})" + (" (mapping)" if in_map else ""))
            continue

        # SEÑAL 3 — Una keyword de la ontología en el nombre
        # Cubre dos variantes:
        #   3a. Label monopalabra → la única palabra del label actúa de keyword.
        #       Alta confianza (9/6) porque hay coincidencia total.
        #   3b. Label multipalabra con 1 keyword → señal válida pero más débil
        #       porque no sabemos si el label completo encaja.
        #       [FIX] En v3 este caso no se puntuaba → labels como
        #       'Centro de interpretación' se perdían con kw_n=1.
        if kw_n == 1 and lw:
            if len(lw) == 1:
                # 3a: label monopalabra
                pts = 9 if in_map else 6
                add(lab, pts, f"keyword '{list(lw)[0]}' en nombre" + (" (mapping)" if in_map else ""))
            else:
                # 3b: label multipalabra, solo 1 keyword matcheada → puntuación reducida
                pts = 5 if in_map else 3
                add(lab, pts, f"1 keyword en nombre (label multipalabra)" + (" (mapping)" if in_map else ""))
            continue

        # SEÑAL 4 — Solo descripción (puntuación muy baja)
        # Deliberadamente débil para que el NOMBRE siempre domine.
        # Únicamente labels multipalabra para evitar que palabras comunes
        # como "plaza", "río", "torre" contaminen el scoring.
        if lw.issubset(tok_f) and not lw.issubset(tok_n):
            if in_map and len(lw) >= 2:
                add(lab, 2, f"'{lab}' en descripción (mapping, multipalabra)")
            elif len(lw) >= 3:
                add(lab, 1, f"'{lab}' en descripción (multipalabra)")
        # Descripción sin mapping: cuando la categoría es 'otros' o vacía
        # no existe mapping que refuerce señales. Permitimos puntuar
        # keywords de la descripción aunque no haya in_map.
        no_mapping = not mapping_labels  # True cuando cat='' o 'otros'
        if kw_d >= 2 and kw_n == 0:
            if in_map:
                add(lab, 2, f"keywords en desc ({kw_d}) (mapping)")
            elif no_mapping and kw_d >= 3:
                # Sin mapping: umbral más alto (>=3) para evitar ruido
                add(lab, 2, f"keywords en desc ({kw_d}) (sin mapping, cat vacía/otros)")
        # Señal extra: label multipalabra ⊆ descripción sin mapping
        if no_mapping and lw.issubset(tok_d) and not lw.issubset(tok_n) and len(lw) >= 2:
            add(lab, 2, f"'{lab}' en descripción (sin mapping)")

    # ── 8. Scoring: aliases ──────────────────────────────────────────────
    # Si el alias (palabra o frase) aparece en los tokens del NOMBRE,
    # añadimos puntuación al label canónico.
    #
    # [FIX] Aliases multipalabra (ej: 'aula de naturaleza'):
    #   tok_n es un SET de tokens individuales, por lo que
    #   'aula de naturaleza' in tok_n → siempre False.
    #   Solución: tokenizamos el alias y comprobamos que TODOS sus
    #   tokens estén en tok_n (misma lógica que onto_keywords_match).
    import re as _re
    for alias, target in ALIASES.items():
        alias_tokens = set(_re.findall(r'[a-záéíóúüñç]+', alias.lower()))
        if not alias_tokens:
            continue
        # Alias monopalabra: el token debe estar en el nombre
        # Alias multipalabra: TODOS sus tokens deben estar en el nombre
        if alias_tokens.issubset(tok_n):
            entry = ontology_entry(target)
            if entry and not has_negative(tok_f, entry):
                in_map = entry['label'] in mapping_labels
                # Los multipalabra reciben un punto extra por ser más específicos
                bonus = 1 if len(alias_tokens) > 1 else 0
                pts = (9 if in_map else 7) + bonus
                add(entry['label'], pts, f"alias '{alias}'→{entry['label']}")

    # ── 9. Desambiguaciones manuales ─────────────────────────────────────
    # Resuelven pares de labels que compiten frecuentemente.

    # "Plaza de toros" vs "Plaza" cuando el nombre dice PLAZA + TOROS
    if 'Plaza de toros' in scores and 'Plaza' in scores \
            and {'plaza', 'toros'}.issubset(tok_n):
        scores['Plaza'][0] = 0

    # "Bar de vinos" vs "Bar" cuando el nombre incluye "vinos"
    if 'Bar de vinos' in scores and 'Bar' in scores and 'vinos' in tok_n:
        scores['Bar'][0] = 0

    # "Parque natural" vs "Parque infantil": la palabra específica decide
    if 'Parque natural' in scores and 'Parque infantil' in scores:
        if 'natural'   in tok_n: scores['Parque infantil'][0] = 0
        elif 'infantil' in tok_n: scores['Parque natural'][0]  = 0

    # "Mirador natural" vs "Mirador urbano": si dice "urbano" → urbano
    if 'Mirador natural' in scores and 'Mirador urbano' in scores:
        if 'urbano' in tok_n: scores['Mirador natural'][0] = 0
        else:                 scores['Mirador urbano'][0]  = 0

    # "Cine" solo si la palabra cine aparece literalmente en el nombre
    # (las keywords de Cine son comunes y generan falsos positivos)
    if 'Cine' in scores and 'cine' not in tok_n:
        scores['Cine'][0] = 0

    # Eliminamos los labels que quedaron a 0 tras las desambiguaciones
    scores = {k: v for k, v in scores.items() if v[0] > 0}

    # ── 10. Fallback ─────────────────────────────────────────────────────
    if not scores:
        # Intentamos el default para esta categoría de origen
        default = DEFAULTS.get(categoria_lower)
        if default:
            entry = ontology_entry(default)
            if entry:
                return build_result(
                    entry, 0.5, False, False,
                    f"Sin señales específicas; default de '{categoria}' → {entry['label']}"
                )
        # Sin default posible: no clasificado
        return build_unclassified()

    # ── 11. Selección del ganador y confianza ────────────────────────────
    ranked       = sorted(scores.items(), key=lambda kv: -kv[1][0])
    best_label,  (best_score,  best_reasons) = ranked[0]
    second_label = ranked[1][0]    if len(ranked) > 1 else None
    second_score = ranked[1][1][0] if len(ranked) > 1 else 0

    # Ambigüedad: los dos primeros candidatos están muy cerca
    ambig = bool(
        second_label and
        (best_score - second_score) < 2 and
        best_score < 10
    )

    # Conflicto: el ganador NO estaba entre los candidatos del mapping
    in_map_final = best_label in mapping_labels
    conflicto    = bool(ruta and not in_map_final)

    # Confianza: escala lineal simple, penalizada por ambigüedad y conflicto
    # Rango efectivo: ~0.45 (score 1, con penalizaciones) a 0.98 (score máximo)
    conf = min(0.98, 0.4 + best_score * 0.05)
    if ambig:     conf = max(0.35, conf - 0.10)
    if conflicto: conf = max(0.45, conf - 0.05)

    entry       = ontology_entry(best_label)
    explicacion = '; '.join(best_reasons[:3])
    if ambig and second_label:
        explicacion += f" | alt: {second_label}"
    if conflicto:
        explicacion = f"[Conflicto con '{categoria}'] " + explicacion

    return build_result(entry, conf, conflicto, ambig, explicacion)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12 — CONSTRUCTORES DE RESULTADO
# ════════════════════════════════════════════════════════════════════════════

def build_result(entry, conf, conflicto, ambig, explicacion):
    """
    Construye el diccionario de salida a partir de una entrada de la ontología.
    Nota: los niveles del CSV se desplazan un nivel respecto a la ontología
    (nivel_2 de la ontología → "Nivel 1" de la salida, etc.).
    """
    return {
        'categoria_final':     entry.get('label', ''),
        'Nivel 1':             entry.get('nivel_2') or '',
        'Nivel 2':             entry.get('nivel_3') or '',
        'Nivel 3':             entry.get('nivel_4') or '',
        'Nivel 4':             entry.get('nivel_5') or '',
        'confianza':           round(conf, 2),
        'conflicto_semantico': bool(conflicto),
        'ambiguedad':          bool(ambig),
        'sin_clasificar':      False,
        'explicacion':         explicacion[:500],
    }


def build_unclassified():
    """Resultado para puntos que no pudieron clasificarse de ninguna forma."""
    return {
        'categoria_final':     'No clasificado',
        'Nivel 1': '', 'Nivel 2': '', 'Nivel 3': '', 'Nivel 4': '',
        'confianza':           0.2,
        'conflicto_semantico': False,
        'ambiguedad':          False,
        'sin_clasificar':      True,
        'explicacion':         'Sin información suficiente para clasificar.',
    }


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13 — MAIN
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 14 — DETECCIÓN DINÁMICA DE COLUMNAS
# ════════════════════════════════════════════════════════════════════════════
# Permite que el clasificador funcione con cualquier CSV sin hardcodear
# nombres de columna. Se define qué rol semántico busca cada grupo de
# candidatos, ordenados por prioridad (el primero que exista gana).

COLUMN_ROLES = {
    # Nombre / título del punto de interés
    'nombre':      ['titulo', 'nombre', 'name', 'denominacion', 'title'],
    # Categoría de origen del sistema fuente
    'categoria':   ['categoria', 'category', 'tipo_categoria', 'tipo', 'type'],
    # Descripción larga
    'descripcion': ['descripcion', 'descripción', 'long_description',
                    'description', 'texto', 'text', 'resumen'],
}


def detect_columns(df_columns):
    """
    Dada la lista de columnas de un DataFrame (ya normalizadas a minúscula
    sin espacios extra), devuelve un dict:
        {'nombre': 'col_real', 'categoria': 'col_real', 'descripcion': 'col_real'}

    Si un rol no se encuentra, su valor es None y se usará '' en los accesos.
    Imprime un aviso para cada rol no encontrado.
    """
    col_lower = {c.strip().lower(): c for c in df_columns}
    result = {}
    for role, candidates in COLUMN_ROLES.items():
        found = None
        for cand in candidates:
            if cand.lower() in col_lower:
                found = col_lower[cand.lower()]
                break
        if found:
            result[role] = found
        else:
            result[role] = None
            print(f"[WARN] No se encontró columna para el rol '{role}'. "
                  f"Candidatos buscados: {candidates}. "
                  f"Columnas disponibles: {list(col_lower.keys())}")
    return result


def get_col_value(row, col_name):
    """Accede a una columna de forma segura; devuelve '' si es None o NaN."""
    if col_name is None:
        return ''
    val = row.get(col_name, '')
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ''
    return str(val)


def main():
    # ── Lectura del CSV con fallback de codificación  [FIX-2] ───────────
    try:
        df = pd.read_csv(CSV_PATH, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        print("[WARN] UTF-8 falló al leer el CSV, reintentando con latin-1…")
        df = pd.read_csv(CSV_PATH, sep=';', encoding='latin-1')

    # ── Normalización de nombres de columna ─────────────────────────────
    df.columns = [c.strip() for c in df.columns]
    print(f"[INFO] Columnas detectadas en el CSV: {list(df.columns)}")

    # ── Detección dinámica de columnas semánticas ────────────────────────
    cols = detect_columns(df.columns)
    print(f"[INFO] Mapeo semántico: nombre='{cols['nombre']}' | "
          f"categoria='{cols['categoria']}' | descripcion='{cols['descripcion']}'")

    # ── Clasificación fila a fila ────────────────────────────────────────
    rows = [
        classify(
            get_col_value(r, cols['nombre']),
            get_col_value(r, cols['categoria']),
            get_col_value(r, cols['descripcion']),
        )
        for _, r in df.iterrows()
    ]

    out   = pd.DataFrame(rows)
    final = pd.concat([df.reset_index(drop=True), out.reset_index(drop=True)], axis=1)

    # ── Selección de columnas de salida ─────────────────────────────────
    # Se incluyen TODAS las columnas del CSV original más las de clasificación.
    # Así funciona igual con ambos formatos (las columnas extra simplemente
    # se conservan tal cual: latitud, longitud, municipio, etc.)
    output_classifier_cols = [
        'categoria_final', 'Nivel 1', 'Nivel 2', 'Nivel 3', 'Nivel 4',
        'confianza', 'conflicto_semantico', 'ambiguedad',
        'sin_clasificar', 'explicacion'
    ]
    # Columnas del CSV original que sí existen en el resultado
    existing_orig = [c for c in df.columns if c in final.columns]
    # Columnas de clasificación que existen en el resultado
    existing_classif = [c for c in output_classifier_cols if c in final.columns]
    final = final[existing_orig + existing_classif]

    # ── Estadísticas de salida ───────────────────────────────────────────
    print(f"\n[INFO] Total filas clasificadas: {len(final)}")
    print("=== Distribución categoria_final (top 35) ===")
    print(final['categoria_final'].value_counts().head(35))
    print(f"\nconflicto_semantico: {final['conflicto_semantico'].sum()}")
    print(f"ambiguedad:          {final['ambiguedad'].sum()}")
    print(f"sin_clasificar:      {final['sin_clasificar'].sum()}")
    print(f"confianza media:     {final['confianza'].mean():.3f}")
    return final


if __name__ == '__main__':
    final = main()
    final.to_excel('clasificacion_output.xlsx', index=False)
    print("Guardado en clasificacion_output.xlsx")