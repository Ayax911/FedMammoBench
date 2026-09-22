# Auditoría de imágenes — Fase 2: labmirp + DataLoader real end-to-end

## Resumen ejecutivo

`docs/AUDITORIA_IMAGENES.md` → `docs/AUDITORIA_IMAGENES_RESULTADOS.md` (ya revisado, aguanta la
verificación) cerró T1–T5 sobre la workstation (`image_root=/media/imagenesmedicas/...`), 15 de los 21
manifests del repo. Quedan dos huecos reales, sin declarar en ese informe: (1) los 3 manifests
`*_local.csv` que usan `exp58`/`exp59`/`exp60` apuntan a otra máquina (`labmirp`) y nunca se tocaron
con datos reales; (2) el `DataLoader` real (con `num_workers>0`, fork de procesos, y el pipeline
completo de `TransformBuilder`) nunca se ejecutó de punta a punta sobre imágenes reales — la vuelta
anterior verificó archivos y arrays crudos por separado, nunca el camino real de entrenamiento. Este
brief cierra ambos huecos.

## Restricción de entorno

Esta sesión (laptop) confirmó por SSH que hay acceso a ambas máquinas: la workstation (usada en la
Fase 1, `/media/imagenesmedicas/...`) y `labmirp` (accesible vía `ssh torre` desde la laptop, o en
sesión directa ahí). T6–T7 de este brief corren en `labmirp`; T8–T9 corren en la workstation, sobre el
mismo `image_root` ya confirmado en la Fase 1.

## Lo que YA está cerrado — no reabrir

- **T1–T5 de `docs/AUDITORIA_IMAGENES_RESULTADOS.md`** — íntegros, re-verificados independientemente
  (conteos de fila exactos contra los CSV reales del repo, grep propio de `mask_path`/`ROI_path` sin
  resultados). No repetir ninguno de esos cinco puntos sobre los 15 manifests de la workstation.
- **Fuga de pacientes entre splits** (`Split.verify_patient_consistency()`,
  `src/datasets/split.py:47`) — se dispara automáticamente en cada `Manifest`+`Split` real, y por
  construcción ya quedó probada en cada corrida real completada hasta hoy (`exp01`–`exp37`, la serie
  federada): si hubiera fuga, esas corridas habrían lanzado `ValueError` al arrancar y nunca habrían
  llegado a generar `runs/expNN/.../metrics.json`. No es un ítem de este brief.
- **Carga de imágenes en federado** (`src/federated/assembly.py:build_node_assembly()`) reusa
  exactamente `Manifest`/`Split`/`TransformBuilder`/`builder_dataloader` sobre los mismos
  `manifests/by_database/*_norm_neg1_1.csv`/`*_norm_0_1.csv` y el mismo `image_root` de la workstation
  ya auditados — confirmado que ningún `configs/federated/*/*.yaml` referencia `labmirp` ni
  `_local.csv`. T8 (mismo código, mismas clases) ya lo cubre; no hace falta un ítem separado para
  federado.
- **`ColorJitter` sobre TIFFs float, modo `"F"` sin `.convert()`, `drop_last=True` solo en train,
  `.iloc` vs. `.loc`** — decisiones ya documentadas y confirmadas en la Fase 1, no reabrir.

## El trabajo pedido

### T6 — Integridad de archivos en `labmirp` (equivalente a T1, máquina distinta)

Antes de nada, confirmar el set exacto de manifests con:
```bash
grep -n "manifest_path\|image_root" configs/exp58_pretrain_ablation_imagenet_2048_1024_1024_512_512.yaml \
    configs/exp59_pretrain_ablation_imagenet_512.yaml configs/exp60_pretrain_ablation_imagenet_256.yaml
```
(la Fase 1 en esta sesión ya confirmó que los tres usan `manifests/fedmammobench_norm_0_1_local.csv` y
`image_root=/home/labmirp/Escritorio/FL-JULIAN/FedMammoBench/data/preproccesed_julian` — reverificarlo
en el momento, no asumirlo, porque los configs pueden haber cambiado desde entonces).

Mismo método que T1: para cada fila de `fedmammobench_norm_0_1_local.csv` (y, si el alcance de `_local`
incluye variantes `by_database/*_local.csv` referenciadas por algún otro config activo, agregarlas),
resolver `abs_image_path` contra el `image_root` de `labmirp` y ejecutar
`PIL.Image.open(path).load()` para forzar la decodificación completa.

- **Criterio de aceptación**: 0 faltantes y 0 corruptos, igual que T1; si hay alguno, listar la ruta
  exacta y a qué `source_dataset`/`split` pertenece.

### T7 — ¿Los manifests `_local.csv` son la misma data que su contraparte de workstation?

`.claude/context/code/CONFIG.md` afirma que `*_local.csv` son "las mismas filas, distinto
`preprocessed_image_path`" que su par sin `_local`. Nunca se verificó con datos.

- **Verificación**: comparar `manifests/fedmammobench_norm_0_1.csv` vs.
  `manifests/fedmammobench_norm_0_1_local.csv` (ambos con 8,341 filas, ya confirmado en la Fase 1):
  mismo conjunto de `ID_image`/`patient_id`, mismo `classification`/`label_norm` fila a fila, y que la
  ÚNICA columna que difiere entre ambos es `preprocessed_image_path`. Si el tiempo alcanza, repetir
  para los `by_database/*_local.csv` que tengan config activo referenciándolos.
- **Criterio de aceptación**: confirmado si son idénticos salvo esa columna. Si hay una fila con
  `classification`/`patient_id` distinto entre el par, es un hallazgo serio — los dos manifests se
  desincronizaron entre máquinas y hay que decir exactamente qué fila(s).

### T8 — `DataLoader` real (fork + `num_workers>0`) de punta a punta, en la workstation

Sobre la workstation (mismo `image_root` que T1–T5), construir el objeto real que usa cualquier
entrenamiento — no una imitación ni un harness sintético:

```python
from src.datasets.manifest import Manifest
from src.datasets.split import Split
from src.datasets.transform import TransformBuilder
from src.datasets.build import builder_dataloader
import torch

manifest = Manifest(manifest_path="manifests/fedmammobench_norm_neg1_1.csv", image_root="<el real de la workstation>")
split = Split(manifest=manifest)
loaders = builder_dataloader(
    split=split,
    train_transform_builder=TransformBuilder(use_horizontal_flip=True, use_rotation=True, rotation_degrees=15),
    eval_transform_builder=TransformBuilder(),
    batch_size=16,
    num_workers=4,   # el valor real de exp05/exp10/etc., no 0 ni 1
)
```

Iterar las primeras ~20 iteraciones de `loaders["train"]`, y una pasada completa de `loaders["val"]`
si el tiempo alcanza. Por cada batch, confirmar:
- shape `(16, 3, 224, 224)`, `dtype == torch.float32`.
- `torch.isfinite(batch).all()` — sin NaN/Inf tras Resize + flip + rotation + `ToTensor` + `Normalize`
  actuando sobre floats mode `"F"` **reales**, no sintéticos de 4×4.
- Cero excepciones de los workers forkeados (`RuntimeError`/`BrokenPipeError`, típicos de decoders
  nativos de PIL bajo `fork`) — esto es exactamente lo que un harness de un solo proceso no expone.

- **Criterio de aceptación**: confirmado si 0 excepciones y 100% de los batches con shape/dtype
  correctos y sin NaN/Inf. Si aparece un error de multiprocessing, reportar en qué iteración/fila del
  manifest ocurrió (imprimir el índice del batch al fallar, no solo el traceback).

### T9 (opcional, solo si el tiempo alcanza) — Throughput real `num_workers=0` vs. `4`

Diagnóstico, no pasa/falla: medir `batches/segundo` de `loaders["train"]` con `num_workers=0` y con
`4`, para saber si el mount de `/media/imagenesmedicas` es cuello de botella de I/O en la workstation.
Reportar ambos números, sin recomendar ningún cambio de config en esta vuelta.

## Qué NO hacer

- No editar ni re-ejecutar nada de `docs/AUDITORIA_IMAGENES.md`/`_RESULTADOS.md` — este es un archivo
  nuevo, ese ciclo ya cerró.
- No lanzar ningún entrenamiento real (`-m src.cli`) completo — T8 es unas pocas iteraciones del
  `DataLoader`, nunca una corrida de `Trainer.fit()`.
- No tocar `TransformBuilder`/`Manifest`/`build.py`/`split.py` — si T8 o T6/T7 encuentran un bug real,
  el fix es la siguiente vuelta del ciclo (`docs/FIXES_AUDITORIA_IMAGENES_FASE2.md`), no esta.
- No reabrir la auditoría de sobreajuste (`AUDITORIA_SOBREAJUSTE*.md`) ni la de W&B
  (`FIXES_WANDB_METRICAS*.md`) — ciclos distintos, ya cerrados.
- No commitear ningún artefacto de imagen real ni hardcodear la ruta de `image_root` de `labmirp`/la
  workstation en código versionado — los configs `exp58`–`exp60` ya la tienen donde debe estar.

## Entregable esperado

`docs/AUDITORIA_IMAGENES_FASE2_RESULTADOS.md`, mismo formato que
`docs/AUDITORIA_IMAGENES_RESULTADOS.md` (✅ Confirmado / ❌ Refutado / ⚠️ hallazgo nuevo por sección),
dejando explícito en cada sección (T6–T9) en qué máquina corrió — la mezcla de dos entornos en un solo
informe es la razón de ser de este brief, no debe quedar ambiguo cuál resultado viene de dónde.

## Verificación

Antes de reportar nada, en cada máquina donde corra su parte:
```bash
.venv/bin/python -c "import src.datasets.manifest; import src.datasets.split; import src.datasets.build; import src.datasets.transform"
```
y confirmar que el `image_root` usado en cada verificación existe y es el que esa máquina tiene
configurado — resolverlo con `Path(...).is_dir()` antes de arrancar cualquier T, no asumirlo de
memoria ni copiarlo del brief sin comprobarlo en el momento.
