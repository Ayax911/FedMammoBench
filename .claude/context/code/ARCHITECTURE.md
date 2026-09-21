# Arquitectura y funcionamiento general de `src/`

Punto de entrada de la rama de código. Lee esto antes de tocar cualquier cosa bajo `src/`; para el
detalle de una carpeta, salta a su hoja. Las firmas por método viven en los `DOCS.md` que ya están
*dentro* de `src/` — estas hojas no los duplican, los complementan con el porqué y los invariantes.

| tarea | hoja | contratos |
|---|---|---|
| escribir/editar un YAML de experimento | [CONFIG.md](CONFIG.md) | `src/config.py` |
| manifests, splits, transforms, dataloaders | [DATASETS.md](DATASETS.md) | `src/datasets/DOCS.md` |
| backbones, heads, congelamiento, pesos | [MODELS.md](MODELS.md) | `src/models/DOCS.md` |
| loop, optimizadores, pérdidas, early stopping | [TRAIN.md](TRAIN.md) | `src/train/DOCS.md` |
| servidor/cliente/estrategias Flower | [FEDERATED.md](FEDERATED.md) | `src/federated/DOCS.md` |

---

## Una sola dirección de dependencia, sin excepciones

```
config.py → seed / metrics / checkpoint / tracking / reporting / datasets / models → train/
    → eval_pipeline.py → cli.py / evaluate.py / federated/
```

Reglas que se hacen cumplir a mano (no hay linter que las verifique aquí):

- **Nada importa `cli.py`.** Ni `evaluate.py`, ni `federated/`, ni `scripts/`.
  `scripts/sweep_train.py` llama a `python -m src.cli` **como subproceso** justamente por esto.
- `evaluate.py` está al mismo nivel que `cli.py` e importa `eval_pipeline.py` directo. Esa es la razón
  de que `evaluate_split()`/`evaluate_by_database()` vivan en `eval_pipeline.py` y no como helpers
  privados dentro de `cli.py`.
- `src/federated/` está en la misma capa que `cli.py`/`evaluate.py`: importa todo lo que está encima
  de `eval_pipeline.py`, nunca `cli.py`, y **nada fuera de `federated/` importa desde `federated/`**.
  Todo contacto con la API de `flwr` está confinado a `federated/server.py` y `federated/client.py`.
- `datasets/` nunca importa `models/` ni `train/`. `models/weights.py` nunca importa `models/build.py`.
- **Un import tardío dentro de una función para esquivar un ciclo es un olor de diseño aquí, no un
  workaround aceptado.** Si hace falta, la capa está mal puesta.

Comprobación barata de que el árbol importa:

```bash
.venv/bin/python -c "import src.cli; import src.evaluate"
.venv/bin/python -c "import src.federated.server; import src.federated.client; import src.federated.evaluate_node"
```

---

## Desviaciones deliberadas del paquete borrado, todas vigentes

| decisión | elección | por qué |
|---|---|---|
| Config | Pydantic v2 + YAML, `extra="forbid"` | un typo falla la validación en vez de ignorarse |
| Herencia de config | **Ninguna** — sin `defaults:`/`base.yaml` | cada YAML de experimento se lee de principio a fin |
| Registries por decorador | **Descartados** — dicts planos de módulo | `_ARCHITECTURES`, `_HEAD_STRATEGIES`, `_OPTIMIZERS`, `_SCHEDULERS`, `_LOSSES`, `_STRATEGIES` son grepeables en un solo lugar |
| Un `Dataset` por fuente | **No** — un único `MammoBenchDataset` | ya está todo consolidado en un manifest CSV |
| Matemática de agregación federada | **Ninguna propia** — clases stock de `flwr` | ver [FEDERATED.md](FEDERATED.md) |

Si vas a revertir alguna de estas, el paquete legacy está en `ec55408` para consultar qué problema
causaba (ver el mapa de documentación en `CLAUDE.md`).

---

## Los dos entrypoints centralizados

**`src/cli.py` — entrenar.** `run()` hace todo de punta a punta: semilla → manifest → split →
transforms → dataloaders → backbone + head → `Trainer.fit()` → re-evaluar el **mejor** checkpoint
sobre val *y* test (+ desglose por base si está activado) → plots, JSONs, `predictions.csv`, W&B.
No hay resume.

```bash
.venv/bin/python -m src.cli --config configs/exp05_fedmammobench_full_weighted.yaml
```

`cli.py` es dueño del **ensamblaje de entrenamiento**, a propósito: `nn.Sequential(backbone, head.build())`,
la apertura de la corrida de W&B (antes del `Trainer`, para que entrenamiento y test caigan en la
misma corrida) y la llamada a `eval_pipeline` después de `fit()`.

**`src/evaluate.py` — re-evaluar sin reentrenar.** Mismo YAML más `--checkpoint <path>.pt`; reconstruye
manifest/split/transforms/modelo idénticos y llama a `eval_pipeline` directo. **Nunca toca
`config.yaml`/`metrics.csv`/`plots/` del `run_dir`, solo `run_dir/val/` y `run_dir/test/`.**

```bash
.venv/bin/python -m src.evaluate --config configs/expNN.yaml --checkpoint runs/expNN/weights/best_epoch123.pt
```

Sirve para backfillear `test/*_by_database.*` en corridas viejas, y para sacar el **AUC pooled** de un
checkpoint global federado (ver [../experiments/FEDERATED.md](../experiments/FEDERATED.md)).

Dos runners de shell encadenan configs y tee-ean a `runs/<exp_id>.log`: `run.sh` (editar el array
`configs=()`) y `run_exp58_60.sh`. El segundo además hace `source ../wandb.env` con `set -a`, porque
**`WANDB_API_KEY` en el entorno le gana a `~/.netrc`** — así exp57–60 suben a la cuenta de W&B
correcta. Copia ese patrón en vez de editar `~/.netrc`.

---

## Módulos de la raíz

- **`config.py`** — `ExperimentConfig` (Pydantic v2, `extra="forbid"`). Ver [CONFIG.md](CONFIG.md).
- **`seed.py`** — `set_global_seed(seed, cudnn_deterministic=True)`, más `seed_worker()` y
  `make_generator()` para que los workers del DataLoader sean deterministas.
- **`metrics.py`** — `build_metric_collection(device)` arma la `MetricCollection` de torchmetrics, e
  incluye `BinaryMacroF1Score`, una `Metric` propia (torchmetrics no trae macro-F1 binario directo).
- **`checkpoint.py`** — `save_checkpoint()`/`load_checkpoint()`. Además del `.pt` completo escribe
  `_backbone.pt` y `_head.pt` por separado, que es lo que consume
  `aggregation_scope: backbone` en federado.
- **`tracking.py`** — `MetricsLogger`: CSV + TensorBoard + W&B detrás de una sola interfaz
  (`log()`, `log_summary()`, `log_image()`, `log_table()`). Importa `wandb` de forma perezosa y
  degrada a no-op sin credenciales.
- **`reporting.py`** — escritores puros de artefactos: `save_metrics_json`, `save_predictions_csv`,
  `plot_confusion_matrix`, `plot_roc_curve`, `plot_loss_curve`, `plot_metric_curve`,
  `compute_confusion_matrix_metrics()`, y el trío por base de datos
  (`save_metrics_by_database_json`, `plot_confusion_matrix_by_database`, `plot_metrics_by_database`).
  Usa **torchmetrics, no scikit-learn**, para matriz de confusión y ROC — sklearn está instalado pero
  deliberadamente sin usar aquí.
- **`eval_pipeline.py`** — `evaluate_split()` (val/test, idéntica para ambos) y `evaluate_by_database()`
  (desglose opt-in). Ambas reciben una ruta de checkpoint y un `MetricsLogger` ya abierto.

### El invariante de `MetricsLogger` que hay que respetar sí o sí

**Los writers de CSV y TensorBoard se abren de forma perezosa, en el primer `log()` — nunca en
`__init__`.** Abrirlos con ansia truncaría el `metrics.csv` real de un `run_dir` (el historial de
entrenamiento, que está commiteado) en el instante en que `evaluate.py:run_evaluation()` instancia un
logger para reusar `log_summary()`/`log_image()`/`log_table()`, aunque nunca llame a `log()`. No
"simplifiques" esto a inicialización ansiosa: destruye historial en silencio cada vez que alguien
re-evalúa un checkpoint viejo.

---

## Artefactos de una corrida

`run_dir` (`runs/<experiment_id>/`): `config.yaml` (snapshot de lo que corrió exactamente),
`metrics.csv`, eventos de TensorBoard, `plots/` (loss + una curva train-vs-val por métrica clínica), y
una carpeta por split evaluado — `val/` y `test/`, cada una con `metrics.json`,
`confusion_matrix_metrics.json`, `predictions.csv`, `confusion_matrix.png`, `roc_curve.png`.

Con `data.by_database_manifests`, `test/` suma `metrics_by_database.json`,
`confusion_matrix_by_database.png` y `metrics_by_database.png`.

Pesos en `checkpoint_dir` (`runs/<experiment_id>/weights/`): `best_epoch<N>.pt` +
`best_epoch<N>_backbone.pt`/`_head.pt`, y `epoch<N>.pt` cada `save_every` épocas.

`ls runs/` es la forma más rápida de ver qué se ejecutó de verdad. Están gitignorados los `*.pt`/`*.pth`,
`events.out.tfevents.*` y `*.log`; **sí** se commitean `metrics.csv`, `metrics.json`,
`predictions.csv` y `plots/*.png` — son el registro de resultados.

---

## Verificación: no hay suite de tests

`pytest` no está instalado. `tests/` solo tiene `__init__.py` y `test_wandb_writer.py`, que importa el
paquete borrado y no puede correr. **Nada bajo `src/` tiene test.**

La verificación en este repo se hace como documenta `PHASES.md`: manejar el código real desde un
script desechable con tensores sintéticos y hacer asserts sobre los artefactos que produce. Pon esos
scripts en el scratchpad de la sesión, no en el repo.

`pyright` está configurado (`pyrightconfig.json`, modo `strict`, `include: ["src"]`) pero **tampoco
está instalado** en el venv. Las anotaciones de tipo y los `# pyright: ignore` en `src/` existen para
satisfacerlo: mantenlos consistentes aunque nada los compruebe aquí.

---

## Convenciones de estilo

- **El idioma es por archivo y está mezclado a propósito.** Español: `config.py`, `cli.py`, `train/`,
  `datasets/build.py`, todos los `DOCS.md`, todo `src/federated/`. Inglés: `datasets/dataset.py`,
  `datasets/manifest.py`, `models/`. Acompaña al archivo que estés editando en vez de imponer uno.
- **Los comentarios cargan el *porqué*, con extensión.** Los docstrings y comentarios existentes
  registran qué bug previene cada línea y qué hace distinto el proyecto INC. Esa densidad es el estilo
  de la casa: cuando cambies comportamiento, **extiende** ese registro en vez de recortarlo.
- Sin registries por decorador: `_ARCHITECTURES`, `_HEAD_STRATEGIES`, `_OPTIMIZERS`, `_SCHEDULERS`,
  `_LOSSES`, `_STRATEGIES` son dicts planos a nivel de módulo, grepeables en un solo lugar.
- Asuntos de commit: `<Verb>: descripción` (`<Feat>:`, `<Fix>:`, `<Docs>:`, `<add>:`, `<exp>:`), con
  `<exp>:` reservado para commitear los resultados de una corrida.
