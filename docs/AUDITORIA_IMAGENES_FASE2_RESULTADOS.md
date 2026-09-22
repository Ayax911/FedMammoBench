# Resultados de la Auditoría de Carga y Procesamiento de Imágenes — Fase 2

**Fecha de ejecución:** 2026-09-22  
**Entorno de ejecución:** Workstation (`labmirp-Precision-5860-Tower`, `/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/...`)  
**Python / Entorno:** `.venv` (Python 3.12.8, PyTorch 2.13.0+cu130, Torchvision 0.23.0+cu130, Pillow, Pandas, NumPy)  
**Rutas de datos validadas (`image_root`):**
- **Workstation (Almacenamiento central):**  
  `/media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench/preproccesed_julian`
- **Configuraciones `_local` (`labmirp`):**  
  `/home/labmirp/Escritorio/FL-JULIAN/FedMammoBench/data/preproccesed_julian`

---

## Resumen Ejecutivo

La **Fase 2** de la auditoría de imágenes cerró los dos vacíos restantes identificados tras la Fase 1:
1. **Verificación de manifests `_local.csv` (T6 y T7):** Se comprobó que los manifests `*_local.csv` (usados en los experimentos de ablación `exp58`, `exp59`, `exp60`) son **100% idénticos fila a fila en sus 20 columnas de metadatos** respecto a sus contrapartes estándar del repositorio. La única diferencia existente es el prefijo de ruta absoluta `/home/labmirp/Escritorio/FL-JULIAN/FedMammoBench/data/preproccesed_julian/` en la columna `preprocessed_image_path`. Las 8,341 imágenes correspondientes fueron abiertas y decodificadas completamente en disco, reportando **0 faltantes y 0 corruptas**.
2. **`DataLoader` real end-to-end con multiprocesamiento (T8):** Se ejecutó el pipeline completo de carga (`Manifest` + `Split` + `TransformBuilder` + `builder_dataloader`) con **`num_workers=4`** (fork de procesos) sobre las imágenes reales en disco. Se validaron decenas de batches de entrenamiento y la totalidad de los conjuntos de validación y test:
   - **0 excepciones** de multiprocesamiento (`RuntimeError`, `BrokenPipeError`).
   - **100% de batches** con shape `(16, 3, 224, 224)` y `dtype == torch.float32`.
   - **0 valores no finitos:** Todos los tensores resultantes tras augmentaciones (flip horizontal, rotación aleatoria, flip vertical, desenfoque gaussiano) son estrictamente finitos (sin `NaN` ni `Inf`).
3. **Throughput y escalabilidad de I/O (T9):** Se midió el rendimiento del `DataLoader` en la workstation con distintos niveles de paralelismo, demostrando un escalamiento eficiente desde **38.08 batches/s (609.2 img/s)** con `num_workers=0` hasta **94.09 batches/s (1505.5 img/s)** con `num_workers=4` y **116.72 batches/s (1867.5 img/s)** con `num_workers=8`.

---

## T6 — Integridad de Archivos de Manifests `_local.csv`

> **Pregunta:** ¿Existen y decodifican correctamente las 8,341 imágenes referenciadas en los manifests `*_local.csv`?

### Resultado: ✅ Confirmado

Se validaron las 8,341 filas de `manifests/fedmammobench_norm_0_1_local.csv` y sus variantes `by_database/*_local.csv` ejecutando `PIL.Image.open(path).load()` para forzar la decodificación completa de cada imagen en disco.

| Manifest `_local` | Total Filas | Válidas | Faltantes | Corruptas | Estado |
|---|---|---|---|---|---|
| `manifests/fedmammobench_norm_0_1_local.csv` | 8,341 | 8,341 (100%) | 0 | 0 | ✅ Íntegro |
| `manifests/fedmammobench_norm_neg1_1_local.csv` | 8,341 | 8,341 (100%) | 0 | 0 | ✅ Íntegro |
| `manifests/by_database/cdd-cesm_norm_0_1_local.csv` | 1,003 | 1,003 (100%) | 0 | 0 | ✅ Íntegro |
| `manifests/by_database/cmmd_norm_0_1_local.csv` | 4,722 | 4,722 (100%) | 0 | 0 | ✅ Íntegro |
| `manifests/by_database/inbreast_norm_0_1_local.csv` | 410 | 410 (100%) | 0 | 0 | ✅ Íntegro |
| `manifests/by_database/kau-bcmd_norm_0_1_local.csv` | 2,206 | 2,206 (100%) | 0 | 0 | ✅ Íntegro |

**Detalle del entorno:**
- En los configs `exp58`, `exp59` y `exp60`, el parámetro `image_root` está configurado como `/home/labmirp/Escritorio/FL-JULIAN/FedMammoBench/data/preproccesed_julian`.
- Debido al comportamiento de `pathlib.Path.__truediv__` en `Manifest.resolve_image_paths()` (`src/datasets/manifest.py:112-114`), cuando `preprocessed_image_path` contiene una ruta absoluta, `self.image_root / abs_path` preserva la ruta absoluta directamente.
- Todas las 8,341 imágenes del árbol `norm_0_1` y `norm_neg1_1` se encuentran completas y libres de corrupción.

---

## T7 — Consistencia de Datos: Manifests `_local.csv` vs. Workstation

> **Pregunta:** ¿Los manifests `*_local.csv` contienen exactamente la misma información que sus contrapartes estándar, variando únicamente en `preprocessed_image_path`?

### Resultado: ✅ Confirmado

Se realizó una comparación exhaustiva columna por columna y fila por fila entre todos los pares de manifests:

| Par de Manifests Comparados | Filas Orig | Filas Local | Columnas Totales | Columnas Idénticas | Columnas con Diferencias |
|---|---|---|---|---|---|
| `fedmammobench_norm_0_1.csv` vs `_local.csv` | 8,341 | 8,341 | 21 | **20** | 1 (`preprocessed_image_path`) |
| `fedmammobench_norm_neg1_1.csv` vs `_local.csv` | 8,341 | 8,341 | 21 | **20** | 1 (`preprocessed_image_path`) |
| `by_database/cdd-cesm_norm_0_1.csv` vs `_local.csv` | 1,003 | 1,003 | 21 | **20** | 1 (`preprocessed_image_path`) |
| `by_database/cmmd_norm_0_1.csv` vs `_local.csv` | 4,722 | 4,722 | 21 | **20** | 1 (`preprocessed_image_path`) |
| `by_database/inbreast_norm_0_1.csv` vs `_local.csv` | 410 | 410 | 21 | **20** | 1 (`preprocessed_image_path`) |
| `by_database/kau-bcmd_norm_0_1.csv` vs `_local.csv` | 2,206 | 2,206 | 21 | **20** | 1 (`preprocessed_image_path`) |

### Hallazgos de T7:
1. **Identidad estricta de metadatos:** Las 20 columnas restantes (`ID_image`, `source_dataset`, `laterality`, `view`, `classification`, `density`, `BIRADS`, `abnormality`, `molecular_subtype`, `raw_image_path`, `mask_path`, `ROI_path`, `x`, `y`, `radius`, `subject_age`, `source_subjectID`, `original_source_path`, `patient_id`, `split`) son **100% idénticas fila a fila**.
2. **Cero discrepancias en etiquetas o splits:** No existe ningún desajuste en `classification`, `patient_id` o `split` entre ninguna versión local y estándar.
3. **Estructura de rutas:** En el 100% de las filas de los manifests `_local.csv`, `preprocessed_image_path` equivale exactamente a:
   ```text
   /home/labmirp/Escritorio/FL-JULIAN/FedMammoBench/data/preproccesed_julian/ + <ruta_relativa_original>
   ```

---

## T8 — `DataLoader` Real de Punta a Punta (`num_workers=4`, Fork)

> **Pregunta:** ¿El pipeline real de `DataLoader` con procesos hijos forkeados (`num_workers=4`) y augmentaciones completas procesa imágenes reales de forma estable y sin errores?

### Resultado: ✅ Confirmado

Se probó la instanciación y ejecución del `DataLoader` real utilizando `src.datasets.build.builder_dataloader` sobre las 3 configuraciones reales del proyecto:

### 1. Configuración Real Float TIFF `norm_neg1_1` (como `exp05`, `exp10`, serie federada)
- **Manifest:** `manifests/fedmammobench_norm_neg1_1.csv` (8,341 muestras).
- **Parámetros:** `batch_size=16`, `num_workers=4`, augmentaciones activas (`horizontal_flip`, `rotation=15°`, `vertical_flip`, `gaussian_blur`), `normalize_mean=None`, `normalize_std=None`.
- **Resultados:**
  - **Train:** 30 batches evaluados exitosamente a una tasa de **25.19 batches/s**.
  - **Val:** Pasada completa (53 batches, 848 muestras) verificada al 100% a **76.41 batches/s**.
  - **Test:** Pasada completa (53 batches, 848 muestras) verificada al 100% a **100.81 batches/s**.
  - **Dtypes y Shapes:** Todos los tensores de imagen tienen shape `(16, 3, 224, 224)` y `dtype == torch.float32`. Todos los tensores de etiquetas tienen `dtype == torch.int64`.
  - **Finitud numérica:** 0 tensores con valores `NaN` o `Inf`.
  - **Excepciones de multiprocesamiento:** 0 excepciones de workers (sin cuelgues ni `BrokenPipeError`).

### 2. Configuración Real Float TIFF `norm_0_1` con normalización 1-canal (como `exp58`, `exp59`, `exp60`)
- **Manifest:** `manifests/fedmammobench_norm_0_1.csv`.
- **Parámetros:** `batch_size=16`, `num_workers=4`, augmentaciones completas, `normalize_mean=(0.449,)`, `normalize_std=(0.226,)`.
- **Resultados:** 30 batches evaluados exitosamente a **57.28 batches/s**, todos con shape `(16, 3, 224, 224)`, `dtype == torch.float32` y 100% finitos.

### 3. Configuración Real JPG Grayscale / RGB (como `exp01`)
- **Manifest:** `manifests/fedmammobench.csv`.
- **Parámetros:** `batch_size=16`, `num_workers=4`, `image_size=(224, 224)`, `normalize_mean=(0.5, 0.5, 0.5)`, `normalize_std=(0.5, 0.5, 0.5)`.
- **Resultados:** 30 batches evaluados exitosamente a **11.68 batches/s**, shapes y dtypes correctos sin errores numéricos.

### ⚠️ Hallazgo de Interfaz (Documentado para Buenas Prácticas)
Si se instancia `TransformBuilder()` con sus valores por defecto (`normalize_mean=(0.5, 0.5, 0.5)`) directamente sobre imágenes TIFF float (modo `"F"`), `transforms.Normalize` fallará con `RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]`, debido a que los TIFFs float entran a `self.transform` como tensores de 1 canal antes de ser replicados a 3 canales por `MammoBenchDataset`.
- Los scripts oficiales (`src/cli.py` y `src/federated/assembly.py`) gestionan esto correctamente leyendo `config.data.normalize_mean` del YAML (que para TIFFs es `None` o una tupla de 1 elemento `(0.449,)`).

---

## T9 — Diagnóstico de Throughput (`num_workers=0` vs. `4` vs. `8`)

> **Diagnóstico:** ¿Cómo escala el I/O del almacenamiento en la workstation al variar el número de workers en el `DataLoader`?

### Resultado: Diagnóstico Completado

Se evaluó el throughput iterando 100 batches reales (`batch_size=16`, 1,600 imágenes por corrida) sobre `manifests/fedmammobench_norm_neg1_1.csv` con todo el pipeline de transformaciones y augmentaciones activado:

| `num_workers` | Tiempo para 100 Batches (s) | Batches por Segundo (b/s) | Imágenes por Segundo (img/s) | Speedup Relativo |
|---|---|---|---|---|
| `num_workers = 0` (proceso principal) | 2.626 s | **38.08 b/s** | 609.2 img/s | 1.00× (base) |
| `num_workers = 2` | 1.976 s | **50.61 b/s** | 809.8 img/s | 1.33× |
| `num_workers = 4` (default configs) | 1.063 s | **94.09 b/s** | **1,505.5 img/s** | **2.47×** |
| `num_workers = 8` | 0.857 s | **116.72 b/s** | **1,867.5 img/s** | **3.07×** |

### Conclusiones del Diagnóstico:
1. **Excelente escalamiento:** El mount de almacenamiento `/media/imagenesmedicas/DATA1/...` no presenta cuellos de botella de contención de I/O.
2. **Eficiencia con `num_workers = 4`:** Con `num_workers=4` (el valor configurado en los YAMLs de entrenamiento), el pipeline alcanza más de **1,500 imágenes por segundo**, superando ampliamente el consumo requerido por un paso de forward/backward en GPU para batch sizes habituales (16 a 64).

---

## Tabla Resumen de Conformidad (Fase 1 + Fase 2)

| Ítem | Descripción | Fase | Estado |
|---|---|---|---|
| **T1** | Integridad de archivos en manifests principales (15 manifests, 50k+ lecturas) | Fase 1 | ✅ Confirmado (0 faltantes, 0 corruptas) |
| **T2** | Conformidad de modos PIL (`"F"` para TIFF, `"L"`/`"RGB"` para JPG) | Fase 1 | ✅ Confirmado (100% conforme) |
| **T3** | Consistencia de rangos numéricos (`[-1, 1]`, `[0, 1]`, `[0, 255]`) | Fase 1 | ✅ Confirmado (100% conforme) |
| **T4** | Análisis de aspect ratio y distorsión geométrica | Fase 1 | ✅ Completado |
| **T5** | Verificación de no-uso de `mask_path`/`ROI_path` en código fuente | Fase 1 | ✅ Confirmado (0 referencias indebidas) |
| **T6** | Integridad de archivos para manifests `*_local.csv` (8,341 TIFFs) | Fase 2 | ✅ Confirmado (0 faltantes, 0 corruptas) |
| **T7** | Identidad de metadatos `*_local.csv` vs manifests estándar | Fase 2 | ✅ Confirmado (20/20 columnas idénticas) |
| **T8** | `DataLoader` real con multiprocesamiento (`num_workers=4`, fork) | Fase 2 | ✅ Confirmado (0 excepciones, dtypes/shapes válidos) |
| **T9** | Throughput y escalamiento de I/O (`num_workers=0, 2, 4, 8`) | Fase 2 | ✅ Completado (>1,500 img/s) |
