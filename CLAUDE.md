# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FedMammoBench: clasificación binaria de mamografía (benigno / maligno) con un ResNet50 pre-entrenado en
RadImageNet, en PyTorch. El entrenamiento centralizado (`src/`) es la línea base validada. Encima hay
una capa federada (`src/federated/`, despliegue gRPC real sobre Flower — sin simulación, sin Ray).

**Este archivo es un enrutador, no el contexto completo.** Carga la hoja que corresponda a la tarea
antes de trabajar; no las cargues todas.

---

## Hacia dónde ir

### Rama de código — `.claude/context/code/`

| si vas a… | lee |
|---|---|
| tocar cualquier cosa bajo `src/` (empieza siempre aquí) | [ARCHITECTURE.md](.claude/context/code/ARCHITECTURE.md) |
| escribir o editar un `configs/*.yaml`, o lanzar un sweep | [CONFIG.md](.claude/context/code/CONFIG.md) |
| manifests, splits, `Dataset`, transforms, dataloaders | [DATASETS.md](.claude/context/code/DATASETS.md) |
| backbones, heads, congelamiento, carga de pesos | [MODELS.md](.claude/context/code/MODELS.md) |
| el loop, optimizadores, pérdidas, early stopping, `Trainer` | [TRAIN.md](.claude/context/code/TRAIN.md) |
| servidor/cliente/estrategias de Flower | [FEDERATED.md](.claude/context/code/FEDERATED.md) |

Los contratos por método (args, raises, returns, ejemplos) están en los `DOCS.md` que viven **dentro**
de `src/`: `src/DOCS.md`, `src/datasets/DOCS.md`, `src/models/DOCS.md`, `src/train/DOCS.md`,
`src/federated/DOCS.md`. Las hojas de arriba no los duplican — aportan el porqué y los invariantes.
Actualiza ambos junto al código.

### Rama experimental — `.claude/context/experiments/`

| si vas a… | lee |
|---|---|
| citar, comparar o extender resultados centralizados (exp01–39, 57–60) | [CENTRALIZED.md](.claude/context/experiments/CENTRALIZED.md) |
| citar, comparar o extender resultados federados (exp40–56) | [FEDERATED.md](.claude/context/experiments/FEDERATED.md) |

Esas dos hojas guardan los **números medidos**. La *receta* de cada experimento vive en el encabezado
de su `configs/expNN_*.yaml`, que es el log experimental real.

---

## El comando

```bash
.venv/bin/python -m src.cli --config configs/exp05_fedmammobench_full_weighted.yaml
```

Esa es toda la interfaz de entrenamiento: semilla → manifest → split → transforms → dataloaders →
backbone + head → `Trainer.fit()` → re-evaluar el **mejor** checkpoint sobre val y test → plots,
JSONs, `predictions.csv`, W&B. No hay resume.

Segundo entrypoint, para evaluar un checkpoint ya entrenado sin reentrenar:

```bash
.venv/bin/python -m src.evaluate --config configs/expNN.yaml --checkpoint runs/expNN/weights/best_epoch123.pt
```

Federado (un proceso servidor + uno por nodo), o todo junto con Docker:

```bash
.venv/bin/python -m src.federated.server --config configs/federated/exp40_fedavg_full/server.yaml
.venv/bin/python -m src.federated.client --config configs/federated/exp40_fedavg_full/node_cmmd.yaml
EXPERIMENT=exp40_fedavg_full docker compose -f docker-compose.federated.yaml up
```

Cada corrida centralizada regenera `experiment_registry.xlsx` (raíz del repo, un consolidado
config+resultados por experimento) al terminar; un hook `PostToolUse` en `.claude/settings.json` lo
vuelve a disparar tras cualquier `Bash` que matchee `-m src.(cli|federated.server|evaluate)` — cubre
federado (que no lo llama desde código) y la re-evaluación pooled post-federada. Regenerarlo a mano:
`.venv/bin/python -m scripts.build_experiment_registry` (lógica en
[`src/registry.py`](src/registry.py)); hace falta si editaste un YAML sin reentrenar, o si entrenaste
fuera de una sesión de Claude Code.

Comprobación barata de que el árbol importa:

```bash
.venv/bin/python -c "import src.cli; import src.evaluate"
.venv/bin/python -c "import src.federated.server; import src.federated.client; import src.federated.evaluate_node"
```

---

## Las trampas que matan una corrida en silencio

Resumen; el detalle y el porqué están en las hojas.

- **`metric_name: f1` vs `f1_macro`** — `f1` se clava en `0.0` sobre `fedmammobench.csv` y devuelve el
  checkpoint sin entrenar. → [CONFIG.md](.claude/context/code/CONFIG.md)
- **Evaluar el mejor checkpoint, nunca el último.** Nunca agregues un flag "qué checkpoint".
  → [TRAIN.md](.claude/context/code/TRAIN.md)
- **Los writers de `MetricsLogger` se abren perezosamente**, nunca en `__init__`: abrirlos con ansia
  trunca el `metrics.csv` commiteado al re-evaluar. → [ARCHITECTURE.md](.claude/context/code/ARCHITECTURE.md)
- **`load_weights()` lanza cuando `matched == 0`** — sin eso, la corrida entrena desde random y solo
  parece mediocre. → [MODELS.md](.claude/context/code/MODELS.md)
- **Deriva de BatchNorm bajo congelamiento** (`_set_frozen_bn_eval`), y `freeze_bn_stats: false` como
  eje experimental deliberado. → [TRAIN.md](.claude/context/code/TRAIN.md)
- **`drop_last=True` solo en train**; **`.iloc`, nunca `.loc`**; **nunca crear `src/data/`**.
  → [DATASETS.md](.claude/context/code/DATASETS.md)
- **Defaults de `fedadam`/`fedyogi` = corrida muerta** (NaN, o AUC 0,5000 exacto).
  → [FEDERATED.md](.claude/context/code/FEDERATED.md)
- **El AUC agregado del servidor no se compara contra un AUC centralizado** — re-evalúa pooled.
  → [experiments/FEDERATED.md](.claude/context/experiments/FEDERATED.md)
- **Recalibra el umbral antes de reportar por base de datos** — con 0,5 fijo, kau-bcmd queda en ~0 de
  sensibilidad pese a AUC ~0,9. → [experiments/CENTRALIZED.md](.claude/context/experiments/CENTRALIZED.md)

---

## Entorno

`.venv/` (Python **3.12.8**) es el intérprete y ya tiene todo lo de `requirements.txt`: torch 2.13 +
CUDA, torchvision 0.28, torchmetrics, pandas 3.0, pydantic 2.13, PyYAML, tensorboard, matplotlib,
wandb, scikit-learn, y `flwr==1.31.0` pinneado exacto. **Invócalo siempre explícitamente**
(`.venv/bin/python`) — no hay paquete instalado ni paso de activación.

`src/__init__.py` hace que `src` sea un paquete, así que `from src.datasets import ...` funciona desde
la raíz sin tocar `PYTHONPATH`. **No** uses `PYTHONPATH=src` + `from datasets import ...`: ese nombre
choca con el `datasets` de HuggingFace.

**No hay suite de tests ni test runner.** `pytest` no está instalado; `tests/` solo tiene
`__init__.py` y `test_wandb_writer.py`, que importa el paquete borrado y no puede correr. `pyright`
está configurado (`pyrightconfig.json`, `strict`) pero tampoco está instalado — mantén consistentes
las anotaciones y los `# pyright: ignore` aunque nada los verifique aquí. La verificación se hace como
documenta `PHASES.md`: manejar el código real desde un script desechable con tensores sintéticos y
hacer asserts sobre los artefactos.

**Rutas absolutas por máquina.** Los configs cargan rutas absolutas de la workstation del laboratorio
o de `labmirp`, con familias de manifest distintas (`*.csv` vs `*_local.csv`). Detalle en
[CONFIG.md](.claude/context/code/CONFIG.md).

W&B: `train.wandb_project` (`null` lo desactiva). Comprueba credenciales con
`grep -q "api.wandb.ai" ~/.netrc` — **nunca hagas `cat` de ese archivo ni pegues una key**.
`WANDB_API_KEY` en el entorno le gana a `~/.netrc` (ver `run_exp58_60.sh`).

`Dockerfile` construye una imagen **solo-entorno** (sin código; el repo se monta en `/workspace`), así
que un cambio de código nunca necesita rebuild. El **único CI** es
`.github/workflows/docker-publish.yml`: publica `ayax911/federal-learning:<tag>` en Docker Hub con un
tag `v*.*.*` o `workflow_dispatch`. Nada en CI lintea, tipa ni ejecuta el código.

---

## Estado del repo — qué es real y qué está obsoleto

`main` es la rama viva y tiene el paquete `src/` reescrito. Mucha documentación commiteada es anterior
a eso y describe cosas que ya no existen en ninguna rama:

- **El paquete legacy `src/fedmammobench/` no existe en ninguna rama** (71 archivos: registries,
  estrategias, cargadores de pesos, su propio `Trainer`, herencia vía `configs/base.yaml`, los scripts
  de consola `fedmammobench-*`). **No hay `pyproject.toml` en este árbol**, así que `pip install -e .`
  y todo comando `fedmammobench-*` están muertos por construcción.
- **La serie de notebooks exp01–exp32 tampoco existe**, borrada en `0e934ec`, junto con
  `scripts/gen_*.py` y los `scripts/run-expNN-*.sh`/`eval-expNN-*.sh`. `configs/` es solo YAML y
  `scripts/` tiene los cuatro post-hoc (`calibrate_threshold.py`, `ensemble_eval.py`,
  `split_manifest_by_database.py`, `build_experiment_registry.py`), el shim de sweep
  (`sweep_train.py`) y `DOCS.md`.
- **`ec55408` es el último commit que tiene ambos** — el paquete legacy completo *y* la serie de
  notebooks con sus generadores:
  ```bash
  git show ec55408:src/fedmammobench/training/trainer.py
  git show ec55408:configs/exp28/exp28.ipynb          # el notebook que src/ fue escrito para reproducir
  git worktree add ../fedmammobench-legacy ec55408    # para navegarlo como checkout completo
  ```
- El convenio de ramas `develop` / `phaseN-*` que describe `PHASES.md` es histórico; esas ramas están
  fusionadas y borradas.

### Mapa de documentación, de mayor a menor confiabilidad

- **`.claude/context/`** — las hojas de arriba. Actual; mantenlas al día con el código.
- `PHASES.md` — qué se portó del proyecto INC (un repo hermano no commiteado aquí) y por qué cada
  default es el que es. Actual. Léelo antes de tocar métricas F1/precision, `drop_last`, determinismo
  de cuDNN, `FocalLoss`, `EarlyStopping`, `ConfigurableMLPHead` o los checkpoints periódicos.
- `docs/FEDERATED_DESIGN.md` — el diseño de la capa federada. Actual.
- `docs/AUDITORIA_SOBREAJUSTE.md` — brief de auditoría del sobreajuste (2026-09-20): qué ya se
  descartó como causa con `archivo:línea`, el diagnóstico medido (el sobreajuste es del régimen de
  entrenamiento, no un bug) y siete hipótesis priorizadas. Actual.
- `docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md` — resultados de esa auditoría (Antigravity, 2026-09-20),
  re-verificados independientemente: H1–H7 todas confirmadas. La más importante para leer resultados
  de `CENTRALIZED.md`: exp37 vs exp28 **no es estadísticamente significativo** a nivel paciente
  (p=0,933) — el ranking del sweep está inflado por medir AUC por imagen en vez de por paciente.
  Actual, con dos números a verificar antes de citarlos en otro lado: el conteo de "505 mamas" de
  test en H1 da 405 al recalcularlo con pandas, y la cita de línea de H6.b (`build.py:106`) señala el
  registro del scheduler, no una instanciación con `mode="min"` hardcodeado.
- `docs/FIXES_AUDITORIA_SOBREAJUSTE.md` — prompt de las cuatro correcciones de bajo riesgo derivadas
  de esa auditoría (H5, H6.a, H6.b, H6.d), con la dirección de fix corregida donde el informe de
  Antigravity se equivocó. Aplicar y luego archivar o borrar — es un prompt de encargo, no diseño
  permanente.
- `docs/FIXES_WANDB_METRICAS.md` — prompt para arreglar dos síntomas de W&B: métricas que no
  coinciden con lo local, y el eje "Step" desalineado de época/ronda. Tres mecanismos distintos
  diagnosticados con datos reales (`wandb-summary.json` vs `metrics.csv`): `log_image()`/`log_table()`
  sin `step=` inflan el step (`+12` a `+13` steps fantasma por corrida, medido), el summary del mejor
  checkpoint colisiona de nombre con la serie por-época (`val_auc` de summary y de la última época son
  números distintos y correctos, no un bug de cómputo), y en federado "epoch"/step significa ronda en
  el servidor y época local en el nodo. Aplicar y luego archivar o borrar, igual que el anterior.
- `docs/AUDITORIA_IMAGENES.md` — brief para Antigravity, para correr en la máquina con acceso real a
  imágenes (esta sesión no lo tiene): integridad de archivos, modo PIL real vs. asumido, rango de
  valores real vs. lo que promete el nombre del manifest, aspecto/resolución real por base de datos, y
  que `mask_path`/`ROI_path` nunca se usen como fuente de imagen. Aplicar y luego archivar o borrar,
  igual que los anteriores.
- `docs/AUDITORIA_IMAGENES_RESULTADOS.md` — resultados (Antigravity, 2026-09-22, workstation con
  `image_root=/media/imagenesmedicas/...`), re-verificados independientemente: conteos de fila de T1
  coinciden exacto con los CSV reales del repo, T5 (`mask_path`/`ROI_path` sin uso) confirmado con
  grep propio. T2–T4 dependen de abrir archivos de imagen reales que esta sesión no tiene — se les
  cree por coherencia interna, no re-verificados a mano. **Caveat de alcance no declarado por el
  informe**: auditó 15 de los 21 manifests del repo — los 6 `*_local.csv` (`manifests/**/*_local.csv`)
  quedaron fuera sin decirlo; tres de ellos son los que usan `exp58`/`exp59`/`exp60`
  (`image_root=/home/labmirp/...`, otra máquina) — esos tres siguen sin auditar con datos reales.
- `docs/AUDITORIA_IMAGENES_FASE2.md` — brief de seguimiento: cierra los dos huecos de la vuelta
  anterior — integridad de los 3 manifests `_local.csv` de `exp58`/`59`/`60` en `labmirp` (T6),
  confirmar que `_local.csv` es la misma data que su contraparte de workstation salvo la ruta (T7), y
  correr el `DataLoader` real (`num_workers=4`, fork, `TransformBuilder` completo) de punta a punta
  sobre imágenes reales en la workstation (T8) — nunca ejercitado antes, la Fase 1 solo verificó
  archivos y arrays crudos por separado. Aplicar y luego archivar o borrar, igual que los anteriores.
- `src/**/DOCS.md` — contratos por método. Actual.
- `configs/*.yaml` (encabezados) — el log experimental real. `exp04_inc_strict_replica.yaml` documenta
  las cuatro divergencias con el INC que corrige y la que deliberadamente no.
- `REFACTOR.md` — **rationale** actual (§4 decisiones de arquitectura, §6 pipeline de transforms y la
  trampa de índices, §7 bugs legacy a reproducir, §10 estadísticas del manifest); sus **secciones de
  estado están mal** (afirma que `src/` son 155 líneas en 4 archivos). Usa el rationale, ignora los
  checklists.
- `docs/DATA_PREPARATION.md`, `docs/METHODOLOGY.md` — formato de manifest y diseño experimental; no
  son específicos del paquete, siguen valiendo.
- `graphify-out/` — grafo de conocimiento generado del árbol (salida de herramienta, gitignorada).
  `GRAPH_REPORT.md` y `graph.json` responden rápido "qué toca qué"; regenéralo tras un refactor grande.
- `docs/EXPERIMENTOS_CENTRALIZADOS.md`, `docs/INFORME_EXP01_22.md` — resultados de la serie de
  notebooks borrada, incluida la falla de propagación de etiquetas a nivel paciente en CMMD que pone un
  piso de ~0,44 de val-loss bajo todas. Históricos.
- **Describen el paquete borrado — no los uses para decidir qué correr:** `README.md` hasta
  "Documentation Architecture", `scripts/DOCS.md`. Describen el paquete federado *legacy* y su
  despliegue de 6 nodos, superados por `src/federated/`.
- **Borrados de `docs/` (2026-09-20)** por describir exclusivamente ese paquete legacy y su despliegue
  de 6 nodos, sin nada rescatable para el árbol actual: `SRC_STRUCTURE.md`, `EXTENDING.md`,
  `CHECKPOINT_COMPATIBILITY.md`, `EXPERIMENT_AUDIT.md`, `RADIMAGENET_IMPLEMENTATION.md`,
  `TRANSFER_LEARNING_GUIDE.md`, `FEDERATED_DEPLOYMENT_GUIDE.md`, `SETUP_6NODES.md`,
  `QUICK_START_6NODES.md`, `NODE_CONFIGURATION_MATRIX.md`, `DOCKER.md`, `audit-plan.md`, `audit/`
  (4 archivos). Cada uno se verificó antes de borrar — importan de `fedmammobench.*` o de un
  `@register_*`/`.samples` que ya no existen, o describen `docker-compose.yml` (singular, con 6
  nodos), reemplazado por `docker-compose.federated.yaml` (4 nodos). Recuperables con
  `git log --all --oneline -- docs/<nombre>.md` si hiciera falta consultar algo puntual.
- **`.claude/commands/`** (`/docker-run`, `/docker-queue`, `/new-exp`, `/eval-experiments`, `/plot`,
  `/compare`, `/check-manifest`, `/validate-configs`) son anteriores a la reescritura y asumen el
  paquete legacy o la imagen Docker vieja. Verifica que uno aplique antes de usarlo;
  `docker-compose.federated.yaml` supera a `/docker-run` y `/docker-queue` para lo federado.
- **`/antigravity`** (`.claude/commands/antigravity.md`, actual) — formaliza el ciclo
  tarea→ejecución externa→revisión con Antigravity: `/antigravity tarea "<tema>"` escribe un brief
  autocontenido en `docs/`, `/antigravity revisar <informe>` audita lo que Antigravity entregó
  re-derivando sus números en vez de confiar en ellos. Nace de
  `docs/AUDITORIA_SOBREAJUSTE.md`/`_RESULTADOS.md`/`FIXES_*.md` — léelos como ejemplo del formato
  antes de usarlo por primera vez.

---

## Convenciones

- **El idioma es por archivo y está mezclado a propósito.** Español: `config.py`, `cli.py`, `train/`,
  `datasets/build.py`, todos los `DOCS.md`, todo `src/federated/`, `.claude/context/`. Inglés:
  `datasets/dataset.py`, `datasets/manifest.py`, `models/`. Acompaña al archivo que edites.
- **Los comentarios cargan el *porqué*, con extensión.** Los docstrings y comentarios registran qué bug
  previene cada línea y qué hace distinto el INC. Esa densidad es el estilo de la casa: cuando cambies
  comportamiento, **extiende** ese registro en vez de recortarlo.
- Asuntos de commit: `<Verb>: descripción` (`<Feat>:`, `<Fix>:`, `<Docs>:`, `<add>:`, `<exp>:`), con
  `<exp>:` reservado para commitear los resultados de una corrida.
