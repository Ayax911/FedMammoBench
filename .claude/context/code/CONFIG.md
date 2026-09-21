# El contrato YAML (`src/config.py`)

Lee esto antes de escribir o editar cualquier `configs/*.yaml`. `ExperimentConfig` es el esquema,
Pydantic v2 con **`extra="forbid"` en todos los modelos**: una clave desconocida o mal escrita es un
`ValidationError`, no un no-op silencioso.

Secciones: `experiment_id`, `architecture`, `head`, `optimizer`, `scheduler` (opcional), `loss`,
`data`, `train`.

**Sin herencia de configs.** No hay `defaults:` ni `base.yaml` — cada YAML de experimento se lee de
principio a fin. Es deliberado (era una fuente de sorpresas en el paquete borrado); no lo reintroduzcas.

`head`/`optimizer`/`scheduler`/`loss` son todos `NamedComponentConfig`: `name` + `hparams` libre que
se splatea al constructor. Por eso **agregar un hiperparámetro casi siempre significa tocar solo la
factory, no los modelos de config.**

---

## Campos y defaults

### `architecture`
| campo | default | nota |
|---|---|---|
| `name` | — | clave en `_ARCHITECTURES` |
| `weights_path` | `None` | obligatorio para `resnet50_radimagenet`; prohibido para las de torchvision |
| `unfreeze_from` | `"none"` | ver [MODELS.md](MODELS.md) |

### `data`
| campo | default | nota |
|---|---|---|
| `manifest_path` | — | CSV manifest |
| `image_root` | — | raíz para resolver rutas del manifest |
| `batch_size` | `16` | los configs reales usan 64 |
| `num_workers` | `1` | los configs reales usan 4 |
| `seed` | `42` | |
| `image_size` | `(224, 224)` | **`null`** para los TIFF ya redimensionados |
| `normalize_mean` | `(0.5, 0.5, 0.5)` | **no** es identidad |
| `normalize_std` | `(0.5, 0.5, 0.5)` | |
| `augmentation` | ver abajo | |
| `by_database_manifests` | `None` | opt-in; mapa `base -> CSV` |

`augmentation`: `horizontal_flip=True/p=0.5`, `rotation_degrees=15`, `vertical_flip=False/p=0.5`,
`blur=False/p=0.3`. Los configs que replican al INC activan vertical-flip (p=0.2) y blur (p=0.3).

### `train`
| campo | default | nota |
|---|---|---|
| `epochs` | — | tope; early stopping puede cortar antes |
| `metric_name` | `"auc"` | **ver la trampa `f1` vs `f1_macro` abajo** |
| `metric_mode` | `"max"` | `"min"` para loss |
| `patience` | `None` | `None` desactiva early stopping |
| `min_delta` | `0.0` | mejora mínima para contar como mejora |
| `save_every` | `None` | checkpoint periódico cada N épocas |
| `checkpoint_dir`, `run_dir` | — | |
| `device` | `"cpu"` | los configs reales usan `"cuda"` |
| `freeze_bn_stats` | `True` | ver la trampa de BN abajo |
| `wandb_project` | `None` | `None` desactiva W&B |
| `wandb_group` | `None` | agrupa corridas en la UI |

---

## El pipeline de TIFF float pre-procesado: dos ajustes que se mueven juntos

`Preproccesed/preprocess_images.py` escribe TIFF float de 32 bits, un solo canal (modo PIL `"F"`), **ya
redimensionados a 224×224** y **ya normalizados** a `[0,1]` (`norm_0_1/`) o `[-1,1]` (`norm_neg1_1/`).
`manifests/fedmammobench_norm_{0_1,neg1_1}.csv` apuntan a ellos.

Un config que consume esos TIFF pone `image_size: null` **y** `normalize_mean: null` /
`normalize_std: null`. Redimensionar y normalizar otra vez sería incorrecto.

`normalize_mean: [0,0,0]` / `normalize_std: [1,1,1]` es el *otro* valor con sentido — identidad, o sea
píxeles `[0,1]` crudos, que es lo que hace el INC. **El default `0.5/0.5` no es equivalente.**

### Media/std de 1 elemento, nunca la tripleta de ImageNet

`MammoBenchDataset` **siempre produce 3 canales** y ya no tiene flag `grayscale` (se quitó en
`ffdb5b2`). Para los TIFF modo `"F"` la replicación ocurre sobre el **tensor, después** de que corre
`transform` — así que `transforms.Normalize` ve un tensor de **1 canal**: `normalize_mean`/`_std`
tienen que ser tuplas de un elemento ahí, y una tripleta de ImageNet revienta por shape mismatch.

Por eso los configs con backbone de ImageNet usan `[0.449]` / `[0.226]` — el promedio de las tres
estadísticas por canal de ImageNet — en vez de la tripleta usual. El encabezado de
`configs/exp58_pretrain_ablation_imagenet.yaml` desarrolla el razonamiento completo.

**Emparejar normalización con backbone importa:** RadImageNet espera `[-1,1]` (`norm_neg1_1`);
los pesos de ImageNet de torchvision esperan `[0,1]` normalizado con estadísticas de ImageNet
(`norm_0_1` + `[0.449]`/`[0.226]`). Cruzarlos cuesta ~0,02 de AUC de forma medida — es lo que miden
exp19 y exp33, ver [../experiments/CENTRALIZED.md](../experiments/CENTRALIZED.md).

---

## Trampas que arruinan una corrida en silencio

- **`metric_name: f1` vs `f1_macro`.** `f1` es `BinaryF1Score` — solo clase positiva — y se queda
  clavado en exactamente `0.0` mientras el modelo no predice ningún maligno, que es el estado normal
  de las primeras épocas sobre el 66/34 de `fedmammobench.csv`. `EarlyStopping` exige mejora estricta,
  así que esa racha de ceros nunca resetea el contador de paciencia y `fit()` devuelve el checkpoint
  de la época 0 (sin entrenar). **Usa `f1_macro` en ese manifest.** `f1` solo es correcto para las
  réplicas del INC, cuyo split de train es 84 % maligno.
- **`freeze_bn_stats: false`** apaga a propósito el re-`eval()` de las BatchNorm congeladas. Con
  backbone totalmente congelado, es la diferencia entre un backbone que aún se adapta y uno clavado a
  las estadísticas de RadImageNet. Está en `false` en los configs que replican al INC. Ver
  [TRAIN.md](TRAIN.md).
- **Evaluar el mejor checkpoint, nunca el último.** `Trainer.fit()` devuelve la ruta del mejor y
  `cli.run()` alimenta exactamente esa a `eval_pipeline`. **Nunca agregues un flag de config del tipo
  "qué checkpoint"** — se desincroniza de lo que realmente fue mejor.

---

## Rutas absolutas: dos máquinas, dos familias de manifest

Cada config carga **rutas absolutas específicas de máquina** en `weights_path` e `image_root`; no
resuelven en otro lado.

- **Workstation original**, `/media/imagenesmedicas/DATA1/.../FedMammoBench/`:
  `manifests/fedmammobench_norm_{0_1,neg1_1}.csv` y `manifests/by_database/<base>_norm_*.csv`.
- **`labmirp`**, `/home/labmirp/Escritorio/FL-JULIAN/FedMammoBench/data/preproccesed_julian`:
  las variantes paralelas **`*_local.csv`** — mismas filas, distinto `preprocessed_image_path`
  absoluto. exp57–60 corren ahí.

Al agregar un config, elige la familia que case con su `image_root`. Mezclarlas da un
`FileNotFoundError` en el primer batch, **no** en la validación del config.

`configs/exp02`, `exp03_*` y `exp04_*` apuntan a `manifests/dataset_split_formatted.csv`, el manifest
del INC, que **no está en este repo y nunca estuvo** — esos tres no se pueden re-correr tal cual en
ninguna máquina.

Los manifests derivados por base viven en `manifests/by_database/`, generados desde los manifests
fuente por `scripts/split_manifest_by_database.py`; recorre ese script si cambian los manifests fuente
(nunca toca datos de imagen).

---

## Barridos de hiperparámetros

`sweeps/hpsearch_v1.yaml` es el espacio de búsqueda, commiteado como registro metodológico.
`scripts/sweep_train.py` es un **shim, no un tercer entrypoint**: el agente de W&B le pasa
hiperparámetros muestreados como flags, los aplica sobre un YAML base, materializa el config del
trial y corre `python -m src.cli --config <ese>` **como subproceso** — nunca `import src.cli`, lo que
además regala un proceso limpio por trial (sin memoria CUDA, semilla global ni estado de W&B
arrastrados entre 100 corridas).

```bash
wandb sweep sweeps/hpsearch_v1.yaml
wandb agent <entity>/fedmammobench2.0/<sweep_id>
```

**Dos campos de ese YAML son estructurales y es fácil "limpiarlos" hasta romper el sweep:**

- `project: fedmammobench2.0` explícito. Sin él, `wandb sweep` infiere un proyecto a partir del repo
  más la carpeta de `program` (`FedMammoBench-scripts`), así que el sweep cae en un proyecto distinto
  al de los trials, que loguean por el `train.wandb_project` de cada config. El agente se ve ocupado
  mientras el sampler bayesiano no ve ni una métrica.
- `command:` fija la **ruta absoluta a `.venv/bin/python`**, no `${env}`/`${interpreter}`: ese combo
  se expande a `/usr/bin/env python` y resuelve contra el PATH del agente — en la workstation, un
  conda base cuyo numpy/torch son incompatibles (`AttributeError: module 'numpy' has no attribute
  'ndarray'` en el primer `import torch`). Tres trials murieron así en menos de 60 s cada uno y el
  flapping guard de W&B mató al agente.

Los artefactos de trial van a `sweeps/<sweep_id>/{configs,runs}/` — deliberadamente **no** a
`configs/` (YAML escrito a mano) ni a `runs/` (el registro commiteado de resultados); `sweeps/*/` está
gitignorado y el config ganador se promueve a mano a un `configs/expNN_*.yaml` permanente.

**Antes de lanzar, comprueba `grep -q "api.wandb.ai" ~/.netrc`**: sin credenciales `MetricsLogger` cae
en silencio a `mode="offline"`, y un agente cuyos trials corren offline nunca recibe métricas de vuelta.
Nunca hagas `cat` de ese archivo ni pegues una key en ningún lado.
