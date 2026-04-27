"""
debug_by_name.py — Busca un POI por nombre en el CSV y genera su traza.

USO:
  python debug_by_name.py "Agencia de Lectura Manuel Durán"           # Modo interactivo (logs + archivo)
  python debug_by_name.py "Agencia de Lectura Manuel Durán" --json    # JSON puro a stdout (para piping)
  python debug_by_name.py "Agencia de Lectura Manuel Durán" --quiet   # Solo genera archivo, sin logs

Genera el JSON de traza y lo guarda en traza_<nombre>.json
"""

import sys
import json
import pandas as pd
from classify_debug import classify_debug

CSV_PATH = 'Servicios_Limpio.csv'

# Parse arguments
if len(sys.argv) < 2:
    print("Uso: python debug_by_name.py 'Nombre del POI' [--json|--quiet]", file=sys.stderr)
    sys.exit(1)

search_name = sys.argv[1].strip().lower()
mode = 'interactive'  # interactive | json | quiet

if len(sys.argv) > 2:
    if sys.argv[2] == '--json':
        mode = 'json'
    elif sys.argv[2] == '--quiet':
        mode = 'quiet'

def log(msg):
    """Solo imprime en modo interactivo, a stderr en modo json"""
    if mode == 'interactive':
        print(msg)
    elif mode == 'json':
        print(msg, file=sys.stderr)
    # En modo quiet no imprime nada

# Cargar CSV
try:
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, sep=";", encoding="latin-1")

df.columns = [c.strip() for c in df.columns]

# Detectar columnas (copia de classify_debug)
COLUMN_ROLES = {
    "nombre":      ["titulo", "nombre", "name", "denominacion", "title"],
    "categoria":   ["categoria", "category", "tipo_categoria", "tipo", "type"],
    "descripcion": ["descripcion", "descripción", "long_description", "description"],
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
    return result

cols = detect_columns(df.columns)
log(f"[INFO] Columnas detectadas: {cols}")

# Buscar el POI
col_nombre = cols['nombre']
if not col_nombre or col_nombre not in df.columns:
    log(f"[ERROR] No se encontró la columna de nombre. Disponibles: {list(df.columns)}")
    sys.exit(1)

# Búsqueda case-insensitive
matches = df[df[col_nombre].astype(str).str.lower().str.contains(search_name, na=False)]

if len(matches) == 0:
    log(f"[ERROR] No se encontró ningún POI con nombre que contenga: '{search_name}'")
    log("\nSugerencia: verifica el nombre exacto en el CSV.")
    sys.exit(1)

if len(matches) > 1:
    log(f"[WARN] Se encontraron {len(matches)} POIs con ese nombre:")
    for idx, row in matches.head(10).iterrows():
        log(f"  - {row[col_nombre]} (cat: {row.get(cols['categoria'], '?')})")
    log("\nUsando el primero. Si quieres otro, ajusta el nombre de búsqueda.")

# Tomar el primero
row = matches.iloc[0]
nombre      = str(row[col_nombre]) if col_nombre else ""
categoria   = str(row[cols['categoria']]) if cols['categoria'] and cols['categoria'] in df.columns else ""
descripcion = str(row[cols['descripcion']]) if cols['descripcion'] and cols['descripcion'] in df.columns else ""

log(f"\n[OK] POI encontrado:")
log(f"  Nombre:      {nombre}")
log(f"  Categoría:   {categoria}")
log(f"  Descripción: {descripcion[:100]}{'...' if len(descripcion) > 100 else ''}")

# Generar traza
log("\n[INFO] Generando traza...")
trace = classify_debug(nombre, categoria, descripcion)

# Guardar JSON (siempre, incluso en modo json)
safe_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in nombre)[:50]
out_file = f"traza_{safe_name}.json"

# CRÍTICO: Escribir con encoding UTF-8 sin BOM
with open(out_file, "w", encoding="utf-8", newline='\n') as f:
    json.dump(trace, f, ensure_ascii=False, indent=2)

# Modo JSON: imprimir JSON puro a stdout (para piping o captura)
if mode == 'json':
    print(json.dumps(trace, ensure_ascii=False, indent=2))
else:
    log(f"\n✓ Traza guardada en: {out_file}")
    log(f"\nResultado de clasificación:")
    log(f"  → {trace['result']['categoria_final']}")
    log(f"  Confianza: {trace['result']['confianza']}")
    log(f"  Conflicto: {trace['result']['conflicto_semantico']}")
    log(f"  Ambiguo:   {trace['result']['ambiguedad']}")
    log(f"\n[INFO] Ahora abre scorer_visualizer.html y carga el fichero {out_file}")