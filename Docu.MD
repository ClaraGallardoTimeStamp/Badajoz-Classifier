# Documentación Técnica — Clasificador SEGITTUR v3

## Índice

1. [Visión general](#1-visión-general)
2. [Arquitectura del sistema](#2-arquitectura-del-sistema)
3. [Ficheros de configuración](#3-ficheros-de-configuración)
4. [Pipeline de clasificación paso a paso](#4-pipeline-de-clasificación-paso-a-paso)
5. [Sistema de puntuación (scoring)](#5-sistema-de-puntuación-scoring)
6. [Reglas de negocio (Hard Rules)](#6-reglas-de-negocio-hard-rules)
7. [Sistema de aliases](#7-sistema-de-aliases)
8. [Interpretación de la salida](#8-interpretación-de-la-salida)
9. [Cómo extender el clasificador](#9-cómo-extender-el-clasificador)
10. [Integración con APIs externas](#10-integración-con-apis-externas)
11. [Referencia de funciones](#11-referencia-de-funciones)
12. [Preguntas frecuentes y troubleshooting](#12-preguntas-frecuentes-y-troubleshooting)

---

## 1. Visión general

El clasificador toma cada punto de interés del CSV de origen y le asigna una categoría canónica de la **ontología SEGITTUR**, junto con su posición jerárquica (hasta 4 niveles) y varias métricas de calidad.

El sistema es **completamente basado en reglas y vocabulario** — no usa modelos de lenguaje ni ML. Su lógica se puede auditar, depurar y ajustar editando los ficheros JSON.

**Diagrama de flujo de alto nivel:**

```
CSV de entrada
      │
      ▼
  Limpieza (HTML, abreviaturas, dirección postal)
      │
      ▼
  Tokenización (nombre + descripción)
      │
      ▼
  Candidatos del Mapping.json
      │
      ▼
  Reglas duras (HR-0 … HR-4)  ──► resultado inmediato si aplican
      │
      ▼
  Scoring contra Ontology_SEGITTUR_2.json
      │
      ▼
  Scoring de aliases
      │
      ▼
  Desambiguaciones manuales
      │
      ▼
  Fallback / "No clasificado"
      │
      ▼
  Excel de salida
```

---

## 2. Arquitectura del sistema

### Módulos funcionales

| Sección en el código | Responsabilidad |
|---|---|
| Sección 1 — Rutas | Centraliza las rutas de los ficheros. Único lugar a tocar para cambiar ubicaciones. |
| Sección 2 — Carga segura | Lee los JSON con mensajes de error claros si falta algún fichero. |
| Sección 3 — Índices | Construye diccionarios O(1) al arrancar para búsquedas rápidas. |
| Sección 4 — Limpieza | `strip_html`, `expand_abbrev`, `strip_address_suffix`, `tokenize`. |
| Sección 5 — Aliases | Construye el vocabulario de sinónimos dinámicamente desde el JSON. |
| Sección 6 — Defaults | Categorías de último recurso cuando el scoring no produce señales. |
| Sección 7 — Normalización | `normalize_text` (elimina acentos) y `expand_cluster` (expande categorías padre). |
| Sección 8 — Mapping | Traduce la categoría de origen del CSV a candidatos de la ontología. |
| Sección 9 — Scoring | Funciones de comparación: `onto_keywords_match`, `has_negative`, `label_words`. |
| Sección 10 — Hard Rules | Reglas de negocio deterministas que se evalúan antes del scoring. |
| Sección 11 — `classify()` | Función principal: orquesta todos los pasos y devuelve el resultado. |
| Sección 12 — Constructores | `build_result` y `build_unclassified` generan el diccionario de salida. |
| Sección 13-14 — Main | Lectura del CSV, detección de columnas, clasificación y escritura del Excel. |

---

## 3. Ficheros de configuración

### `Ontology_SEGITTUR_2.json`

Array de objetos. Cada objeto define una categoría turística con su posición jerárquica y el vocabulario que la identifica.

```json
{
  "label":              "Castillo",
  "nivel_2":            "Patrimonio Cultural",
  "nivel_3":            "Arquitectura Defensiva",
  "nivel_4":            "Castillo y Fortaleza",
  "nivel_5":            "",
  "keywords":           ["castillo", "fortaleza", "alcázar", "torreón"],
  "keywords_negativas": ["parque"]
}
```

| Campo | Descripción |
|---|---|
| `label` | Nombre canónico de la categoría. Es la salida en `categoria_final`. |
| `nivel_2` … `nivel_5` | Jerarquía. Se vuelcan en las columnas `Nivel 1`–`Nivel 4` del Excel. |
| `keywords` | Palabras o frases que identifican esta categoría en nombre/descripción. |
| `keywords_negativas` | Si alguna de estas palabras aparece en el texto, se descarta este label. |

**Para añadir una nueva categoría:** agregar un objeto nuevo al array. El orden importa: ante keywords duplicadas, gana el label que aparece antes (mayor precedencia).

### `Mapping.json`

Array de objetos que traduce las categorías del sistema de origen a las de la ontología.

```json
{
  "categoria_origen": "Edificio religioso",
  "ruta": {
    "nivel_2": "Patrimonio Cultural",
    "nivel_3": "Arquitectura Religiosa",
    "nivel_4": ["Iglesia", "Capilla", "Monasterio"]
  }
}
```

Los valores de `nivel_2/3/4` pueden ser:
- Un string exacto que coincide con un `label` de la ontología.
- Un nombre de cluster (nodo padre) que se expande automáticamente a sus hojas.
- Un array de varios valores.
- Valores separados por `|` o `,` dentro de un string.

**Para añadir una nueva categoría de origen:** añadir un objeto al array con su `categoria_origen` y la `ruta` correspondiente.

---

## 4. Pipeline de clasificación paso a paso

La función `classify(nombre, categoria, descripcion)` ejecuta estos pasos en orden:

### Paso 1 — Normalización de tipos
Convierte `None` y `NaN` a string vacío para evitar errores en los pasos siguientes.

### Paso 2 — Limpieza de texto
- `strip_html()`: elimina etiquetas HTML y entidades (`&amp;`, `&nbsp;`…).
- `expand_abbrev()`: expande abreviaturas de vías (`C/` → `calle`, `Avda.` → `avenida`).

### Paso 3 — Eliminación de dirección postal del nombre
`strip_address_suffix()` elimina la parte de dirección que a veces se concatena al nombre en el CSV:

```
"CASA SOLARIEGA CALLE SANTA ANA 1"  →  "CASA SOLARIEGA"
"ERMITA CALLE DEL PINO S/N"         →  "ERMITA"
```

**Excepción:** si la categoría de origen es `punto de ruta`, el nombre no se modifica (las rutas son vías y el nombre describe precisamente esa vía).

### Paso 4 — Tokenización
Genera conjuntos de tokens (palabras sin puntuación ni mayúsculas):
- `tok_n` — tokens del nombre limpio.
- `tok_d` — tokens de la descripción.
- `tok_f` — unión de ambos.

### Paso 5 — Candidatos del mapping
`mapping_candidate_labels()` consulta `Mapping.json` para obtener los labels esperados según la categoría de origen. Estos candidatos actúan como "pistas previas":
- Si el ganador del scoring es uno de ellos → `conflicto_semantico = False`.
- Si el ganador es otro → `conflicto_semantico = True`.

### Paso 6 — Reglas duras
Se evalúan antes del scoring. Si alguna dispara, se devuelve resultado inmediato con confianza 0.92. Ver [sección 6](#6-reglas-de-negocio-hard-rules).

### Pasos 7-8 — Scoring
Ver [sección 5](#5-sistema-de-puntuación-scoring).

### Paso 9 — Desambiguaciones manuales
Resuelve pares de labels que compiten habitualmente:

| Situación | Resolución |
|---|---|
| `Plaza de toros` vs `Plaza` con "toros" en nombre | Puntúa `Plaza` a 0 |
| `Bar de vinos` vs `Bar` con "vinos" en nombre | Puntúa `Bar` a 0 |
| `Parque natural` vs `Parque infantil` | La palabra específica decide |
| `Mirador natural` vs `Mirador urbano` | "urbano" en nombre → mirador urbano; si no → natural |
| `Cine` sin "cine" en el nombre | Puntúa `Cine` a 0 |

### Paso 10 — Fallback
Si el scoring no produce ningún candidato, se consulta el diccionario `DEFAULTS`. Si tampoco hay default, se devuelve `build_unclassified()` con `sin_clasificar = True`.

### Paso 11 — Selección del ganador
Se ordena por puntuación descendente. Se calcula confianza, ambigüedad y conflicto semántico.

---

## 5. Sistema de puntuación (scoring)

El scoring recorre todos los labels de la ontología y asigna puntos según las señales encontradas. Las señales del **nombre siempre dominan** sobre las de la descripción.

### Señales y puntos

| Señal | Puntos (sin mapping) | Puntos (con mapping) |
|---|---|---|
| Palabras del label ⊆ tokens del nombre | 7 | 10 |
| ≥ 2 keywords de ontología en nombre | 5 | 8 |
| 1 keyword + label monopalabra en nombre | 6 | 9 |
| 1 keyword + label multipalabra en nombre | 3 | 5 |
| Label (≥ 2 palabras) ⊆ descripción (con mapping) | 2 | 2 |
| Label (≥ 3 palabras) ⊆ descripción (sin mapping) | 1 | — |
| ≥ 2 keywords en descripción (con mapping) | — | 2 |
| ≥ 3 keywords en descripción (sin mapping, cat vacía) | 2 | — |
| Alias en nombre | 7 (+1 si multipalabra) | 9 (+1 si multipalabra) |

**"Con mapping"** significa que el label está entre los candidatos que devolvió `Mapping.json` para la categoría de origen de esa fila.

### Cálculo de confianza

```
confianza = min(0.98, 0.4 + puntuación_ganador × 0.05)
si ambigüedad:      confianza -= 0.10  (mínimo 0.35)
si conflicto:       confianza -= 0.05  (mínimo 0.45)
si regla dura:      confianza = 0.92   (fijo)
si default:         confianza = 0.50   (fijo)
si sin_clasificar:  confianza = 0.20   (fijo)
```

### Ambigüedad
Se marca cuando la diferencia entre el primero y segundo candidato es menor de 2 puntos y el ganador no alcanza 10 puntos.

---

## 6. Reglas de negocio (Hard Rules)

Las reglas duras se evalúan **antes** del scoring. La primera que dispara devuelve resultado inmediato. Se definen en `apply_hard_rules()`.

| ID | Condición | Resultado |
|---|---|---|
| HR-0 | "centro de interpretación" en nombre O descripción | Centro de interpretación |
| HR-1 | `fexme.com` en la descripción (URL de senderos extremeños) | Sendero |
| HR-2 | Categoría `edificio religioso` + "colegio" en nombre | Centro educativo |
| HR-3 | Categoría `edificio religioso` + "hospital" en nombre | Capilla / Arquitectura vernácula / Conjunto artístico (según descripción) |
| HR-3b | Categoría vacía u "otros" + "hospital" en nombre | Capilla / Conjunto artístico (según descripción) |
| HR-4 | Categoría `espacio natural` o `punto de ruta` + cueva/gruta en nombre | Cueva o caverna |

**Para añadir una nueva regla dura:**

1. Identificar la condición determinista (categoría + palabra clave + señal en descripción).
2. Añadir un bloque `if` numerado (`HR-5`, `HR-6`…) en `apply_hard_rules()`, antes del `return None, None` final.
3. Documentar la regla en el docstring de `apply_hard_rules()`.

Ejemplo:

```python
# HR-5: Categoría "espacio natural" + "laguna" en nombre → Laguna
if categoria_lower == 'espacio natural' and 'laguna' in tok_n:
    entry = ontology_entry('laguna')
    if entry:
        return entry, "HR-5: espacio natural + laguna en nombre → Laguna"
```

---

## 7. Sistema de aliases

Los aliases son sinónimos que mapean una palabra del texto al label canónico de la ontología.

**Se generan automáticamente** desde las `keywords` del JSON. No hay aliases hardcodeados en el código.

```
"ermita"    → "Capilla"         (porque "ermita" es keyword de la entrada "Capilla")
"fortaleza" → "Castillo"        (porque "fortaleza" es keyword de la entrada "Castillo")
"balneario" → "Spa"             (porque "balneario" es keyword de la entrada "Spa")
```

**Para añadir o modificar aliases:** editar el campo `keywords` de la entrada correspondiente en `Ontology_SEGITTUR_2.json`. No tocar el código.

**Precedencia:** si la misma keyword aparece en varios labels, gana el que aparece primero en el JSON. El script informa en consola cuántas keywords se ignoraron por duplicación al arrancar.

---

## 8. Interpretación de la salida

### Columnas de clasificación

| Columna | Tipo | Descripción |
|---|---|---|
| `categoria_final` | string | Label canónico de la ontología. `"No clasificado"` si no se pudo asignar ninguno. |
| `Nivel 1` | string | Equivale a `nivel_2` de la ontología (el nivel raíz es interno). |
| `Nivel 2` | string | Equivale a `nivel_3` de la ontología. |
| `Nivel 3` | string | Equivale a `nivel_4` de la ontología. |
| `Nivel 4` | string | Equivale a `nivel_5` de la ontología. Puede estar vacío. |
| `confianza` | float 0–1 | Fiabilidad estimada de la clasificación. |
| `conflicto_semantico` | bool | El label asignado no coincide con lo que el mapping esperaba para esa categoría de origen. Merece revisión manual. |
| `ambiguedad` | bool | Dos candidatos estaban muy igualados. El resultado puede ser incorrecto. |
| `sin_clasificar` | bool | No se encontró ningún label válido. |
| `explicacion` | string | Cadena de debug: señales que determinaron el resultado. Muy útil para entender por qué se asignó una categoría. |

### Umbrales de confianza recomendados

| Rango | Interpretación recomendada |
|---|---|
| ≥ 0.85 | Alta confianza. Apto para importación directa. |
| 0.65 – 0.84 | Confianza media. Revisar si `conflicto_semantico = True`. |
| 0.50 – 0.64 | Baja confianza. Revisión manual recomendada. |
| < 0.50 | Muy baja confianza o fallback. Revisar siempre. |

### Lectura de la columna `explicacion`

```
"keywords en nombre (2) (mapping) | alt: Monasterio"
 └── señal principal               └── segundo candidato (ambigüedad)

"[Conflicto con 'Edificio religioso'] 'Cine' en nombre"
 └── el mapping esperaba otra cosa   └── señal que ganó
```

---

## 9. Cómo extender el clasificador

### Añadir una categoría nueva a la ontología

Editar `Ontology_SEGITTUR_2.json` e insertar un nuevo objeto. Colocarlo **antes** de cualquier otro label que comparta keywords para darle mayor precedencia.

```json
{
  "label": "Destilería",
  "nivel_2": "Turismo Enológico e Industrial",
  "nivel_3": "Industria Agroalimentaria",
  "nivel_4": "Destilería y Bodega",
  "nivel_5": "",
  "keywords": ["destilería", "destileria", "aguardiente", "alambique"],
  "keywords_negativas": []
}
```

### Añadir una categoría de origen nueva

Editar `Mapping.json`:

```json
{
  "categoria_origen": "Instalación industrial",
  "ruta": {
    "nivel_2": "Turismo Enológico e Industrial",
    "nivel_3": ["Destilería", "Bodega"]
  }
}
```

### Cambiar el CSV de entrada

El clasificador detecta automáticamente las columnas relevantes (ver `COLUMN_ROLES` en Sección 14 del código). Si tu CSV tiene nombres de columna distintos, añade el nombre a la lista de candidatos correspondiente:

```python
COLUMN_ROLES = {
    'nombre':      ['titulo', 'nombre', 'name', 'mi_columna_titulo', ...],
    'categoria':   ['categoria', 'category', 'mi_columna_tipo', ...],
    'descripcion': ['descripcion', 'descripción', 'mi_columna_texto', ...],
}
```

### Usar el clasificador como librería (sin CSV)

`classify()` es una función pura que no depende del CSV. Se puede importar y llamar directamente:

```python
from clasificador_v3 import classify

resultado = classify(
    nombre      = "Castillo de Belmonte",
    categoria   = "Castillos y fortalezas",
    descripcion = "Fortaleza medieval del siglo XV, declarada Bien de Interés Cultural."
)

print(resultado['categoria_final'])   # "Castillo"
print(resultado['confianza'])         # 0.98
print(resultado['Nivel 1'])           # "Patrimonio Cultural"
```

---

## 10. Integración con APIs externas

Esta sección explica cómo conectar el clasificador a flujos de datos que llegan o salen por API en lugar de por ficheros CSV.

### 10.1 Exponer el clasificador como API REST (Flask)

Instalar Flask:

```bash
pip install flask
```

Crear `api_server.py`:

```python
from flask import Flask, request, jsonify
from clasificador_v3 import classify

app = Flask(__name__)

@app.route('/clasificar', methods=['POST'])
def clasificar():
    """
    Endpoint de clasificación individual.

    Cuerpo JSON esperado:
    {
        "nombre":      "Castillo de Belmonte",
        "categoria":   "Castillos y fortalezas",
        "descripcion": "Fortaleza medieval..."
    }

    Respuesta JSON:
    {
        "categoria_final": "Castillo",
        "Nivel 1": "Patrimonio Cultural",
        ...
        "confianza": 0.98,
        "conflicto_semantico": false,
        "ambiguedad": false,
        "sin_clasificar": false,
        "explicacion": "..."
    }
    """
    data = request.get_json(force=True)

    nombre      = data.get('nombre', '')
    categoria   = data.get('categoria', '')
    descripcion = data.get('descripcion', '')

    resultado = classify(nombre, categoria, descripcion)
    return jsonify(resultado)


@app.route('/clasificar/lote', methods=['POST'])
def clasificar_lote():
    """
    Endpoint de clasificación en lote.

    Cuerpo JSON esperado:
    {
        "registros": [
            {"nombre": "...", "categoria": "...", "descripcion": "..."},
            ...
        ]
    }
    """
    data      = request.get_json(force=True)
    registros = data.get('registros', [])

    resultados = [
        classify(r.get('nombre', ''), r.get('categoria', ''), r.get('descripcion', ''))
        for r in registros
    ]
    return jsonify(resultados)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

Arrancar:

```bash
python api_server.py
```

Llamada de prueba:

```bash
curl -X POST http://localhost:5000/clasificar \
  -H "Content-Type: application/json" \
  -d '{"nombre":"Ermita de San Blas","categoria":"Edificio religioso","descripcion":"Pequeña capilla rural"}'
```

### 10.2 Consumir registros desde una API externa

Si los datos llegan paginados desde una API REST en lugar de un CSV:

```python
import requests
from clasificador_v3 import classify
import pandas as pd

API_BASE    = "https://mi-plataforma.com/api/puntos"
API_TOKEN   = "mi_token_secreto"
PAGE_SIZE   = 100

def fetch_page(page: int) -> list:
    r = requests.get(
        API_BASE,
        params={"page": page, "size": PAGE_SIZE},
        headers={"Authorization": f"Bearer {API_TOKEN}"},
        timeout=30
    )
    r.raise_for_status()
    return r.json().get("items", [])

def clasificar_desde_api() -> pd.DataFrame:
    resultados = []
    page = 1

    while True:
        items = fetch_page(page)
        if not items:
            break

        for item in items:
            resultado = classify(
                nombre      = item.get("titulo") or item.get("nombre", ""),
                categoria   = item.get("categoria", ""),
                descripcion = item.get("descripcion") or item.get("description", ""),
            )
            # Conservar campos originales junto con la clasificación
            resultados.append({**item, **resultado})

        print(f"[INFO] Página {page} procesada ({len(items)} registros)")
        page += 1

    return pd.DataFrame(resultados)


if __name__ == "__main__":
    df = clasificar_desde_api()
    df.to_excel("clasificacion_output.xlsx", index=False)
    print(f"Guardado: {len(df)} registros clasificados")
```

### 10.3 Subir resultados a una API externa

Una vez clasificados, para hacer POST de los resultados de vuelta a una plataforma:

```python
import requests
import pandas as pd

UPLOAD_ENDPOINT = "https://mi-plataforma.com/api/puntos/actualizar"
API_TOKEN       = "mi_token_secreto"
BATCH_SIZE      = 50   # ajustar según los límites de la API destino

def subir_clasificaciones(df: pd.DataFrame):
    """
    Envía los resultados de clasificación a la API destino en lotes.
    """
    registros = df.to_dict(orient='records')
    total     = len(registros)
    enviados  = 0
    errores   = 0

    for i in range(0, total, BATCH_SIZE):
        lote = registros[i : i + BATCH_SIZE]

        # Construir el payload según el esquema de la API destino.
        # Ajustar los nombres de campos a lo que espere tu API.
        payload = {
            "actualizaciones": [
                {
                    "id":                  r.get("id"),
                    "categoria_segittur":  r.get("categoria_final"),
                    "nivel_1":             r.get("Nivel 1"),
                    "nivel_2":             r.get("Nivel 2"),
                    "nivel_3":             r.get("Nivel 3"),
                    "confianza":           r.get("confianza"),
                    "requiere_revision":   r.get("conflicto_semantico") or r.get("ambiguedad"),
                }
                for r in lote
            ]
        }

        try:
            r = requests.post(
                UPLOAD_ENDPOINT,
                json=payload,
                headers={"Authorization": f"Bearer {API_TOKEN}"},
                timeout=60
            )
            r.raise_for_status()
            enviados += len(lote)
            print(f"[OK] Lote {i // BATCH_SIZE + 1}: {len(lote)} registros enviados")
        except requests.RequestException as e:
            errores += len(lote)
            print(f"[ERROR] Lote {i // BATCH_SIZE + 1} falló: {e}")

    print(f"\nResumen: {enviados} enviados correctamente, {errores} con error sobre {total} totales")
```

### 10.4 Pipeline completo: API de entrada → clasificación → API de salida

```python
# pipeline.py

from clasificar_desde_api import clasificar_desde_api
from subir_clasificaciones import subir_clasificaciones

def pipeline_completo():
    print("[1/3] Descargando registros desde la API de origen...")
    df = clasificar_desde_api()

    print(f"[2/3] {len(df)} registros clasificados.")
    print(f"      sin_clasificar: {df['sin_clasificar'].sum()}")
    print(f"      confianza media: {df['confianza'].mean():.3f}")

    # Opcional: guardar copia local antes de subir
    df.to_excel("clasificacion_output.xlsx", index=False)

    print("[3/3] Subiendo resultados a la API de destino...")
    subir_clasificaciones(df)

if __name__ == "__main__":
    pipeline_completo()
```

### 10.5 Autenticación con OAuth 2.0

Si la API usa OAuth 2.0 (client credentials), obtener el token antes de las llamadas:

```python
import requests

def obtener_token(token_url: str, client_id: str, client_secret: str) -> str:
    r = requests.post(
        token_url,
        data={
            "grant_type":    "client_credentials",
            "client_id":     client_id,
            "client_secret": client_secret,
            "scope":         "write:clasificaciones"  # ajustar al scope real
        },
        timeout=15
    )
    r.raise_for_status()
    return r.json()["access_token"]

TOKEN = obtener_token(
    "https://auth.mi-plataforma.com/oauth/token",
    "mi_client_id",
    "mi_client_secret"
)
# Usar TOKEN en los headers de las llamadas posteriores
```

---

## 11. Referencia de funciones

### Funciones públicas principales

#### `classify(nombre, categoria, descripcion) → dict`
Función principal. Acepta strings (o `None`/`NaN`). Devuelve el diccionario de resultado descrito en la [sección 8](#8-interpretación-de-la-salida).

#### `ontology_entry(label) → dict | None`
Busca una entrada en la ontología por su label. Primero búsqueda exacta (sin importar mayúsculas), luego búsqueda normalizada sin acentos.

#### `mapping_candidate_labels(cat_origen) → (set, dict | None)`
Dado el valor de la columna `categoria` del CSV, devuelve los labels candidatos del mapping y el objeto ruta completo.

#### `expand_cluster(label) → list[str]`
Expande un nombre de nodo padre de la ontología a sus labels hoja. Usa caché interna.

#### `apply_hard_rules(name_clean, categoria_lower, desc_raw, tok_n, tok_d) → (entry | None, str | None)`
Evalúa las reglas duras. Devuelve `(None, None)` si ninguna aplica.

### Funciones de limpieza

| Función | Entrada | Salida |
|---|---|---|
| `strip_html(s)` | HTML string | Texto limpio |
| `expand_abbrev(s)` | Texto con abreviaturas | Texto expandido |
| `strip_address_suffix(name, categoria)` | Nombre + categoría | Nombre sin dirección postal |
| `tokenize(s)` | String | `set` de tokens en minúscula |
| `normalize_text(s)` | String | Minúsculas sin acentos |

---

## 12. Preguntas frecuentes y troubleshooting

**P: El script arranca y luego imprime muchos `[WARN] expand_cluster`.**
R: Hay valores en `Mapping.json` que no coinciden con ningún nodo de la ontología. Revisar que los nombres del mapping correspondan exactamente (sin acentos o con acentos) a `nivel_2/3` del JSON de ontología. La función `normalize_text` debería absorber diferencias de tildes, pero puede haber diferencias tipográficas mayores.

**P: Muchas filas salen con `sin_clasificar = True`.**
R: Las causas más frecuentes son: (1) la columna de nombre o descripción está vacía en el CSV, (2) la categoría de origen no está en `Mapping.json` y tampoco hay señales en el texto, (3) faltan keywords en la ontología para ese tipo de elemento. Revisar la columna `explicacion` para ver qué ocurrió en cada fila.

**P: Un elemento se clasifica mal a pesar de tener la palabra clave exacta en el nombre.**
R: Puede que una keyword negativa esté descartando el label correcto, o que otro label esté acumulando más puntos. Añadir `print(scores)` antes del paso 11 en `classify()` para ver la tabla de puntuaciones completa durante el debug.

**P: ¿Cómo proceso un CSV con separador `,` en lugar de `;`?**
R: Cambiar el argumento `sep` en la llamada a `pd.read_csv` dentro de `main()`:
```python
df = pd.read_csv(CSV_PATH, sep=',', encoding='utf-8')
```

**P: ¿Se puede procesar el CSV en paralelo para acelerar la clasificación?**
R: Sí. `classify()` no tiene estado global mutable durante la clasificación (los índices se construyen al cargar el módulo). Con `pandas` se puede paralelizar con `swifter` o `multiprocessing`:

```python
import swifter  # pip install swifter

rows = df.swifter.apply(
    lambda r: classify(
        get_col_value(r, cols['nombre']),
        get_col_value(r, cols['categoria']),
        get_col_value(r, cols['descripcion']),
    ),
    axis=1,
    result_type='expand'
)
```