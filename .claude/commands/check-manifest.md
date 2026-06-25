# /check-manifest — Audita un manifest CSV antes de entrenar

Verifica integridad de datos: patient leakage, balance de clases, y columnas requeridas. Implementa los checks de `docs/DATA_PREPARATION.md` y `docs/EXPERIMENT_AUDIT.md`.

## Uso
```
/check-manifest <ruta_manifest> [--node-dir <dir_manifests>]
```

**Ejemplos:**
- `/check-manifest manifests/node0_manifest.csv`
- `/check-manifest manifests/ddsm-split.csv`
- `/check-manifest --node-dir manifests/` → audita todos los `nodeN_manifest.csv` y verifica que no hay pacientes compartidos entre nodos

## Instrucciones

### 1. Cargar el manifest

Lee el CSV con pandas. Si falla, reporta el error y para.

### 2. Verificar columnas requeridas

Busca las columnas clave (acepta variantes de nombre):
- `image_path` o `filepath` o `path` → rutas de imagen
- `label` o `pathology` o `_label` → etiqueta de clase
- `patient_id` o `pid` → identificador de paciente (**crítico**)
- `split` → columna de partición (opcional pero recomendada)

Reporta cuáles están presentes y cuáles faltan.

### 3. Verificar patient_id

```python
missing_pid = df["patient_id"].isna().sum()
```
Si > 0: **WARN** — "N filas sin patient_id. El particionamiento federado usará sample-level (sin garantía anti-leakage)."

### 4. Verificar patient leakage entre splits

Si la columna `split` existe:
```python
per_patient = df.groupby("patient_id")["split"].nunique()
leaky = per_patient[per_patient > 1]
```
Si `len(leaky) > 0`: **FAIL** — "Pacientes en múltiples splits: {lista}. Esto invalida las métricas de evaluación."

Si no hay columna `split`: informa que el split será generado automáticamente al cargar.

### 5. Balance de clases por split

Para cada split (o para el total si no hay columna `split`):
- Cuenta benign vs malignant (mapea etiquetas conocidas: MALIGNANT/malignant/1/Malignant → malignant, etc.)
- Calcula ratio y porcentaje
- Reporta clase minoritaria

```
Split: train
  benign:    3420  (62.1%)
  malignant: 2088  (37.9%)
  ratio:     1.64:1
  → Moderadamente desbalanceado. Recomendado: auto_class_weights=true
```

Umbrales:
- ratio < 2:1 → OK
- ratio 2:1–5:1 → WARN "Desbalanceado. Usar `loss.auto_class_weights: true`"
- ratio > 5:1 → WARN fuerte "Muy desbalanceado. Considerar oversampling o pos_weight manual."

### 6. Verificación de rutas de imagen (muestra aleatoria)

Toma 5 filas aleatorias y verifica que los archivos existen en disco.
- Intenta resolver las rutas relativas (primero relativa al CWD, luego relativa a `data/`).
- Si ≥ 1 no existe: WARN con las rutas problemáticas.
- Si ninguna existe: sugiere verificar `image_root` en el YAML.

### 7. Modo `--node-dir`: audit multi-nodo

Si se pasa `--node-dir`:
- Carga todos los `node<N>_manifest.csv` del directorio
- Verifica que **ningún patient_id aparece en más de un nodo**
- Muestra distribución por nodo: N imágenes, N pacientes, ratio

```
Distribución entre nodos:
  node1: 5202 imgs, 1108 pacientes — benign:malignant = 1108:4094
  node2:  410 imgs,  205 pacientes — benign:malignant =  310:100
  ...
  → Pacientes compartidos entre nodos: 0 ✓
```

### 8. Resumen final

```
AUDITORÍA: manifests/node0_manifest.csv
═══════════════════════════════════════════════
  Filas        : 5202
  Pacientes    : 1108
  Splits       : train (4162), val (520), test (520)
  Columnas OK  : image_path ✓  patient_id ✓  label ✓  split ✓

  LEAKAGE:       Sin leakage entre splits ✓
  BALANCE:       WARN — ratio malignant:benign = 3.69:1 (usar auto_class_weights)
  RUTAS:         5/5 imágenes de muestra encontradas ✓
═══════════════════════════════════════════════
  Estado: LISTO CON ADVERTENCIAS
```

Estado final: `LISTO`, `LISTO CON ADVERTENCIAS`, o `PROBLEMAS CRÍTICOS`.
