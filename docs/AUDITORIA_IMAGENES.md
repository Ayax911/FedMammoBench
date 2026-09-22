# Auditoría de carga y procesamiento de imágenes (con datos reales)

## Resumen ejecutivo

Esta sesión (una laptop sin el mount de datos) no puede correr nada de esto con imágenes reales — ya
se verificó (ver §Restricción de entorno) que no hay ningún `image_root` accesible aquí. Quien ejecute
este brief lo hace en la workstation/`labmirp`, donde sí hay acceso real a las ~lo que sea que tenga
`Mammo-Bench/preproccesed_julian` (o el path equivalente en esa máquina). El pipeline de carga
(`Manifest` → `Split` → `TransformBuilder` → `MammoBenchDataset` → `DataLoader`) ya fue auditado una
vez con datos **sintéticos** (`docs/AUDITORIA_SOBREAJUSTE.md`/`_RESULTADOS.md`, H1–H7, y los fixes de
`docs/FIXES_AUDITORIA_SOBREAJUSTE.md`). Esta vuelta es específicamente para encontrar lo que solo se
ve con imágenes de verdad: archivos rotos/faltantes, modos PIL inesperados, rangos de normalización
que no son los que el nombre del manifest promete, y variabilidad de resolución/aspecto real entre
bases de datos que un TIFF sintético `"F"` de 4×4 no puede exponer.

## Restricción de entorno

Verificado en esta sesión (2026-09-22): no existe ningún directorio de imágenes en esta laptop.
`grep image_root configs/*.yaml` da rutas absolutas de otras máquinas (workstation de laboratorio,
`labmirp`) — ver `.claude/context/code/CONFIG.md` §rutas por máquina. `/media/akira` existe pero sin
permiso de lectura (`drwxr-x---+`, sin sudo) desde esta sesión. El usuario confirmó: las imágenes NO
están en este equipo — este brief se sube a GitHub y se ejecuta en el computador donde sí hay acceso
real a los datos. **Quien lo ejecute debe confirmar primero, con un comando real, cuál de las rutas de
`CONFIG.md` (o una nueva) es el `image_root` correcto en esa máquina**, y decirlo explícitamente en el
informe — no asumir cuál config usar.

## Lo que YA está cerrado — no reabrir

- **`drop_last=True` solo en train, nunca val/test`** (`src/datasets/build.py:76`) — deliberado, BatchNorm1d
  revienta con batch de 1. Documentado con su propio comentario largo. No es un bug.
- **Modo PIL `"F"` (float TIFF, pre-normalizado) se salta `.convert()` y expande a 3 canales *después*
  de `transform`** (`src/datasets/dataset.py:89-106`) — decisión deliberada y documentada
  (`.convert()` en modo `"F"` trunca en vez de reescalar). No proponer volver a usar `.convert("RGB")`
  para TIFFs float.
- **`.iloc`, nunca `.loc`, en accesos por índice posicional** — convención de la casa
  (`.claude/context/code/DATASETS.md`). Si el informe encuentra un `.loc` en código de carga de
  imágenes, es candidato a bug real, no algo a "corregir hacia `.loc`".
- **`ColorJitter` destruye TIFFs float** — ya diagnosticado en una vuelta anterior
  (memoria de sesión `diagnostico-sobreajuste-2026-09`); confirmado que `TransformBuilder`
  (`src/datasets/transform.py`) actualmente **no** ofrece `ColorJitter` en absoluto. Si el informe lo
  encuentra reintroducido en algún branch de config, es un hallazgo válido; si no aparece, no hace
  falta mencionarlo.
- **Fuga de pacientes entre splits ya se previene por `patient_id` no vacío**
  (`Manifest.check_patients_id()`, `src/datasets/manifest.py:63-78`, con el fix H6.d ya aplicado que
  detecta también strings vacíos/whitespace, no solo NaN) — el chequeo de que `Split` realmente separa
  por `patient_id` sin overlap ya se hizo con datos sintéticos en la auditoría anterior. Esta vuelta
  puede re-confirmarlo a escala real (todos los ~miles de pacientes reales, no un puñado sintético),
  pero no es el foco — el foco es la carga/decodificación de la imagen en sí, no el split.

## El trabajo pedido

Cada ítem requiere iterar el manifest real completo (todas las filas de `manifests/*.csv` con split
`train`/`val`/`test`) contra los archivos de imagen reales en el `image_root` de esa máquina.

### T1 — Integridad de archivos: ¿existe y abre cada imagen que el manifest promete?

`Manifest.resolve_image_paths()` (`src/datasets/manifest.py:98-114`) solo valida que el **directorio**
`image_root` exista — nunca que cada `abs_image_path` individual exista o sea un archivo válido.
`MammoBenchDataset.__getitem__` (`src/datasets/dataset.py:87`, `Image.open(row["abs_image_path"])`) es
perezoso: un archivo roto o faltante solo se descubre cuando el `DataLoader` llega a esa fila
específica, potencialmente minutos u horas dentro de un entrenamiento largo.

- **Verificación**: recorrer TODAS las filas de cada manifest en `manifests/` (los que tengan
  configs activos referenciándolos, ver `grep manifest_path configs/*.yaml`) y, por cada
  `abs_image_path` resuelto, intentar `Image.open(path).load()` (`.load()` fuerza la decodificación
  completa, no solo el header). Reportar: cuántos archivos faltan, cuántos existen pero fallan al
  decodificar, y — si hay algún patrón — a qué `source_dataset`/`split` pertenecen los rotos.
- **Criterio de aceptación**: "confirmado" si el conteo de faltantes/rotos es 0 en todos los splits
  activos; si no es 0, listar cada ruta rota con su `experiment_id`/manifest de origen — esto es lo
  más urgente de este brief, porque afecta directamente si se puede confiar en cualquier corrida
  próxima.

### T2 — ¿El modo PIL real de cada archivo coincide con lo que el código asume?

El código (`dataset.py:89`) bifurca en exactamente dos casos: modo `"F"` (float TIFF pre-normalizado)
vs. todo lo demás (convertido a RGB). Con datos sintéticos nunca se probó un modo PIL inesperado
(ej. `"I"`, `"1"`, `"P"`, o un TIFF float pero con más de 1 canal).

- **Verificación**: por cada manifest, tabular `Image.open(path).mode` real contra
  `preprocessed_image_path` — ¿hay algún archivo que no sea ni `"F"` ni algo convertible limpiamente a
  `"RGB"`? ¿Algún `"F"` con más de 1 canal (rompería el chequeo `image_tensor.shape[0] == 1` de
  `dataset.py:104`, que asume float TIFF = 1 canal siempre)?
- **Criterio de aceptación**: confirmado si el modo observado en 100% de los archivos reales cae
  limpiamente en uno de los dos casos que el código maneja; si aparece un tercer caso, describirlo con
  path de ejemplo y decir qué hace hoy el código con él (probablemente crashea o produce un tensor
  corrupto silenciosamente — decir cuál de las dos cosas pasa).

### T3 — ¿El rango de valores real coincide con lo que el nombre del manifest promete?

Los manifests se llaman `fedmammobench_norm_neg1_1.csv` / `_norm_0_1.csv` — nombres que prometen un
rango de pixel específico *ya aplicado en disco* (por el script de preprocesamiento, fuera de este
repo). `TransformBuilder`/`Normalize` (`src/datasets/transform.py:117-119`) asume que esa promesa es
cierta y no la reverifica.

- **Verificación**: sobre una muestra real (ej. 200 imágenes por manifest, estratificada por
  `source_dataset`), cargar el array crudo (antes de cualquier `transform`) y reportar min/max/media
  por archivo. Para `norm_neg1_1`: ¿el rango real está efectivamente en `~[-1, 1]`? Para `norm_0_1`:
  ¿en `~[0, 1]`?
- **Criterio de aceptación**: confirmado si el 95%+ de la muestra cae dentro de una tolerancia
  razonable (a definir por el propio informe, ej. ±5% del rango nominal); si hay una base de datos
  específica (`source_dataset`) sistemáticamente fuera de rango, reportarlo — es evidencia de que el
  preprocesamiento de esa base específica no coincide con lo que su nombre de archivo promete, y
  explicaría cualquier degradación de métricas específica de esa base ya vista en
  `.claude/context/experiments/CENTRALIZED.md` (recalibración por base de datos).

### T4 — Variabilidad de resolución/aspecto real entre bases de datos, contra `Resize` fijo

Todos los configs activos usan `image_size=(224, 224)` (ver `TransformBuilder` default y
`configs/*.yaml`). Un TIFF sintético de la auditoría anterior es cuadrado por construcción — nunca
expuso qué tan distinto es el aspecto real de una mamografía CC vs. MLO, o entre bases de datos con
equipos de captura distintos (CMMD, CDD-CESM, INBreast, KAU-BCMD, etc., ver `source_dataset` en el
manifest).

- **Verificación**: sobre la misma muestra estratificada de T3, reportar `(width, height)` real de
  cada imagen (antes de `Resize`) agrupado por `source_dataset` y `view` (CC/MLO). Calcular cuánto
  distorsiona el aspect ratio un `Resize((224,224))` no-preservante (¿hay bases con aspect ratio muy
  distinto a 1:1 que sufren más distorsión que otras?).
- **Criterio de aceptación**: esto es diagnóstico, no pasa/falla — el entregable es la tabla de
  aspect ratios por base + una recomendación (ej. "considerar padding en vez de resize directo para la
  base X") si algo salta a la vista como desproporcionado. No implementar el cambio en esta vuelta —
  solo señalarlo si aparece.

### T5 — `mask_path`/`ROI_path`: confirmar que nunca se usan por accidente en vez de `preprocessed_image_path`

El manifest trae columnas `mask_path` y `ROI_path` (visibles en el header real, ver ejemplo de fila en
`manifests/fedmammobench_norm_neg1_1.csv`) que el pipeline de carga actual (`Manifest`,
`MammoBenchDataset`) nunca lee — solo usa `preprocessed_image_path`. Confirmar que sigue siendo así en
el código actual (grep de `mask_path`/`ROI_path` fuera de `manifest.py`'s parsing/pass-through) y que
ningún config o script post-hoc los usa como fuente de imagen por error.

- **Verificación**: `grep -rn "mask_path\|ROI_path" src/ scripts/` y confirmar que ningún resultado
  los usa para abrir una imagen (leer/mostrar el contexto de cada match).
- **Criterio de aceptación**: confirmado si cero usos como fuente de imagen; si aparece alguno,
  reportar archivo:línea exacto — sería un bug serio (entrenar sobre máscaras/ROIs en vez de la imagen
  real).

## Qué NO hacer

- No tocar `Manifest.resolve_image_paths()` para agregar validación de existencia por archivo en esta
  vuelta — el brief pide **medir** cuántos archivos faltan/rompen, no arreglarlo todavía. Si T1
  encuentra faltantes reales, el fix (agregar el chequeo, decidir si debe ser `raise` o solo warning)
  es la siguiente vuelta del ciclo (`docs/FIXES_AUDITORIA_IMAGENES.md`), no esta.
- No reabrir el split por paciente (`Split`) ni la auditoría de sobreajuste (H1–H7) — temas ya
  cerrados, ciclos distintos.
- No proponer cambiar `image_size=(224,224)` ni ninguna augmentación de `TransformBuilder` a partir de
  lo que encuentre T4 — solo diagnosticar y recomendar, no aplicar.
- No commitear ningún artefacto de imagen real ni ruta absoluta de la workstation a este repo — el
  informe reporta números agregados (conteos, tablas, estadísticas), nunca copia archivos de datos ni
  hardcodea la ruta de `image_root` de esa máquina en código versionado (ya existe la convención de
  rutas-por-máquina en `configs/*.yaml`, no crear una tercera).

## Entregable esperado

`docs/AUDITORIA_IMAGENES_RESULTADOS.md`, con una sección por cada T1–T5, siguiendo el mismo formato de
"Resultado: ✅ Confirmado / ❌ Refutado / ⚠️ hallazgo nuevo" que
`docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md`. Cada hallazgo con evidencia citable (conteos exactos,
paths de ejemplo, `archivo:línea` del código relevante) — no solo "se revisó y está bien". Si T1
encuentra archivos rotos/faltantes, el informe debe listarlos (o, si son demasiados, un resumen
agregado por `source_dataset`/`split` con el conteo exacto) para que la siguiente vuelta del ciclo
pueda decidir el fix.

## Verificación

No aplica el criterio habitual de "verificable sin datos reales" — el punto de esta vuelta es
justamente que corre donde sí hay datos. La única verificación de bajo costo que Antigravity debe
correr antes de reportar nada:

```bash
.venv/bin/python -c "import src.datasets.manifest; import src.datasets.dataset; import src.datasets.build; import src.datasets.transform"
```

y confirmar que el `image_root` que usó en cada verificación existe y es el que la máquina real tiene
configurado (no asumir un path de memoria — resolverlo con `Path(...).is_dir()` antes de arrancar).
