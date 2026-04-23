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