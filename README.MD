# Clasificador Turístico SEGITTUR — v3

Clasificador automático de puntos de interés turístico basado en la **ontología SEGITTUR**. Toma un CSV con nombre, categoría y descripción de cada punto y devuelve un Excel con la categoría final normalizada, sus niveles jerárquicos y métricas de confianza.

---

## Requisitos

```
Python >= 3.9
pandas
openpyxl
```

```bash
pip install pandas openpyxl
```

---

## Ficheros necesarios

| Fichero | Descripción |
|---|---|
| `clasificador_v3.py` | Script principal |
| `Ontology_SEGITTUR_2.json` | Ontología de categorías turísticas (labels, keywords, jerarquía) |
| `Mapping.json` | Correspondencia entre categorías de origen del CSV y la ontología |
| `Servicios_Limpio.csv` | CSV de entrada (separador `;`, codificación UTF-8 o latin-1) |

---

## Uso rápido

```bash
python clasificador_v3.py
```

Genera `clasificacion_output.xlsx` en el directorio de trabajo.

---

## Columnas de salida

| Columna | Descripción |
|---|---|
| `categoria_final` | Label canónico de la ontología SEGITTUR |
| `Nivel 1–4` | Jerarquía del label dentro de la ontología |
| `confianza` | Puntuación 0–1 (≥ 0.85 = alta, 0.5–0.84 = media, < 0.5 = baja) |
| `conflicto_semantico` | `True` si el resultado contradice el mapping esperado |
| `ambiguedad` | `True` si dos candidatos tienen puntuaciones muy similares |
| `sin_clasificar` | `True` si no se encontró ninguna categoría válida |
| `explicacion` | Razón textual de la clasificación (útil para debug) |

---

## Configuración rápida

Todas las rutas de ficheros están centralizadas al principio del script:

```python
ONTO_PATH    = 'Ontology_SEGITTUR_2.json'
ALIASES_PATH = 'Ontology_SEGITTUR_2.json'
MAPPING_PATH = 'Mapping.json'
CSV_PATH     = 'Servicios_Limpio.csv'
```

---

## Documentación completa

Ver [`Docu.md`](Docu.md) para detalles sobre la arquitectura interna, las reglas de negocio, cómo extender el sistema y cómo integrarlo con APIs externas.S

---

## Actualización 24/04/2026

Se agrega lo siguiente:

1. classify_debug.py — módulo Python que instrumenta el pipeline completo y genera un JSON de traza detallado
2. scorer_visualizer.html — visualizador interactivo que renderiza esa traza como un debugger paso a paso


Cómo funciona el sistema
classify_debug.py instrumenta el mismo pipeline de classify() sin alterarlo: captura la salida de cada etapa como un dict JSON y la devuelve junto con el resultado. Tres modos de uso:
bash# Modo individual (genera JSON para pegar en el visualizador)
python classify_debug.py \
  --nombre "Iglesia de San Pedro" \
  --categoria "edificio religioso" \
  --pretty

# Modo batch (genera un .jsonl con una traza por fila)
python classify_debug.py --csv Servicios_Limpio.csv --out trazas.jsonl --max 200

# Desde Python
from classify_debug import classify_debug
trace = classify_debug("Fuente del Pino", "espacio natural", "...")
scorer_visualizer.html — abre en cualquier navegador, sin servidor. Muestra:
PanelQué vesNormalizaciónname_raw → name_clean, si se eliminó direcciónTokenizacióntokens coloreados: head (azul), complemento genitivo (tachado), primera palabra (violeta)Genitivosi se dividió el nombre, qué conector activó el split, si hubo excepción multipalabraPrimera palabrasi el token es semántico, si el bonus +4 se activaMappingcandidatos resueltos, nivel del mapping (n4/n3/n2), si fue deterministaHard Rulestodas las reglas con cuál disparó (si aplica)Scoringleyenda de señales con puntuaciones, stats de ejecuciónRanking finaltabla completa con barras de score, badges por tipo de señal, filtros MAP / Top10 / búsqueda
El banner de cortocircuito (violeta) aparece automáticamente si el pipeline se detuvo antes del scoring por hard rule o mapping determinista.


# Pendiente de subir new version