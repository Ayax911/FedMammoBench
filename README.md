# FedMammoBench

Clasificación binaria de mamografía (benigno / maligno) con un ResNet50 pre-entrenado en
[RadImageNet](https://github.com/BMEII-AI/RadImageNet), en PyTorch. El entrenamiento **centralizado**
(`src/`) es la línea base validada; encima hay una capa **federada** (`src/federated/`) con despliegue
gRPC real sobre [Flower](https://flower.ai/) — sin simulación y sin Ray.

**Python**: 3.12 · **Flower**: `flwr==1.31.0` (pinneado exacto) · **Licencia**: MIT

> Este repo se reescribió: el paquete legacy `fedmammobench` (registries, `pip install -e .`, comandos
> `fedmammobench-*`, `configs/base.yaml`, backend Ray) **ya no existe**. Todo lo de abajo describe el
> árbol actual.

---

## Inicio rápido

```bash
git clone <repo> && cd FedMammoBench
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

No hay paquete instalable ni paso de activación: se invoca siempre `.venv/bin/python` desde la raíz.

Comprobación barata de que el árbol importa:

```bash
.venv/bin/python -c "import src.cli; import src.evaluate; import src.gradcam"
.venv/bin/python -c "import src.federated.server; import src.federated.client; import src.federated.evaluate_node"
```

Los configs cargan **rutas absolutas** (`image_root`, `weights_path`, manifests) de la workstation del
laboratorio; ajústalas a tu máquina antes de entrenar. Ver [formato de manifest](docs/DATA_PREPARATION.md).

---

## Entrenamiento centralizado

```bash
.venv/bin/python -m src.cli --config configs/exp05_fedmammobench_full_weighted.yaml
```

Esa es toda la interfaz: semilla → manifest → split → transforms → dataloaders → backbone + head →
`Trainer.fit()` → re-evaluación del **mejor** checkpoint sobre val y test → plots, JSONs,
`predictions.csv` y W&B. No hay resume.

Evaluar un checkpoint ya entrenado sin reentrenar:

```bash
.venv/bin/python -m src.evaluate --config configs/expNN.yaml --checkpoint runs/centralizado/expNN/weights/best_epoch123.pt
```

Grad-CAM sobre un checkpoint (`src/gradcam.py`, núcleo en `src/interpretability.py`; brief en
[docs/GRADCAM.md](docs/GRADCAM.md)).

Cada corrida centralizada regenera `experiment_registry.xlsx` (un consolidado config + resultados por
experimento). A mano: `.venv/bin/python -m scripts.build_experiment_registry`.

### Backbones disponibles

ResNet50 con pesos RadImageNet o ImageNet, ResNet50 / ResNet18 desde cero, ResNet18 con pesos ImageNet
(congelamiento configurable por capa), EfficientNet-B0 y una CNN propia. Detalle y contratos en
[`src/models/DOCS.md`](src/models/DOCS.md).

### Configuración

Cada `configs/expNN_*.yaml` es **autocontenido** (sin herencia) y su encabezado es el log experimental
real: receta, motivación y resultado. Validado con Pydantic (`extra="forbid"`: una clave mal escrita
falla de inmediato). Para un barrido de hiperparámetros: `sweeps/` + `scripts/sweep_train.py`.

Trampa frecuente: con `fedmammobench.csv` usa `metric_name: f1_macro`, no `f1` (este último se clava en
`0.0` y devuelve el checkpoint sin entrenar).

### Salidas

Todo `run_dir` cuelga de `runs/centralizado/<expNN_...>/` (centralizado, auditorías, ensembles) o
`runs/federado/<expNN_...>/` (servidor + `nodes/`): pesos del mejor checkpoint, `metrics.csv`,
`val/` y `test/` con `metrics.json`, `predictions.csv`, gráficas y `config` efectivo.

---

## Aprendizaje federado

Un proceso servidor (que **nunca ve imágenes**) más un proceso por nodo, cada uno con su propio YAML
bajo `configs/federated/<exp>/` (`server.yaml` + `node_<base>.yaml`). Topología de FedMammoBench: 4
nodos, uno por base de datos (CMMD, KAU-BCMD, CDD-CESM, INbreast).

```bash
.venv/bin/python -m src.federated.server --config configs/federated/exp40_fedavg_full/server.yaml
.venv/bin/python -m src.federated.client --config configs/federated/exp40_fedavg_full/node_cmmd.yaml
# ... un cliente por nodo
```

O todo junto con Docker (servidor + un contenedor por nodo, host networking, solo Linux):

```bash
EXPERIMENT=exp40_fedavg_full docker compose -f docker-compose.federated.yaml up
```

La imagen es **solo-entorno** (el repo se monta en `/workspace`), así que un cambio de código nunca
necesita rebuild.

Re-evaluar un nodo contra el mejor modelo global:

```bash
.venv/bin/python -m src.federated.evaluate_node --config <node.yaml> --server-run-dir runs/federado/<exp>/server
```

### Estrategias

Registradas en `src/federated/strategies.py` (clases stock de `flwr.server.strategy` salvo las dos
últimas, implementadas en `cyclic.py` y `scaffold.py`):

| `strategy.name` | Notas |
|---|---|
| `fedavg` | Promedio ponderado por muestras |
| `fedprox` | `proximal_mu` en `strategy.hparams`; se reporta `train_prox_loss` |
| `fedadam`, `fedyogi`, `fedadagrad` | Los defaults de `fedadam`/`fedyogi` producen corridas muertas (NaN o AUC 0,5000 exacto): ajusta `eta`/`beta_*` |
| `cyclic` | Entrenamiento secuencial por nodo |
| `scaffold` | Variables de control |

`fedbn` no está registrado a propósito. `aggregation_scope: full` agrega el modelo entero.

### Fed-ISIC2019 (réplica de FLamby)

Además de mamografía hay un grid federado que replica el benchmark Fed-ISIC2019 de FLamby (6 centros,
EfficientNet-B0, 7 estrategias): `configs/federated/fedisic2019_*`, `scripts/fed_isic2019/` y
[docs/FEDISIC2019_FLAMBY_REPRO.md](docs/FEDISIC2019_FLAMBY_REPRO.md) (+ `_RESULTADOS`).

### Cómo leer un resultado federado

- El AUC que guarda `best.json` del servidor es el **promedio por nodo ponderado por muestras**, no el
  AUC del pool de predicciones. Para comparar contra un centralizado, **re-evalúa pooled** con
  `src.evaluate` sobre el checkpoint global.
- Distingue **AUC pooled** de **AUC intra-base**: el primero premia al modelo por reconocer de qué
  hospital viene la imagen. Ver [docs/DECISION_PASAR_A_FEDERADO.md](docs/DECISION_PASAR_A_FEDERADO.md).
- Con 0,5 fijo de umbral, kau-bcmd (4,3 % de malignos) queda en ~0 de sensibilidad pese a AUC ~0,9:
  recalibra con `scripts/calibrate_threshold.py --by-database` antes de reportar por base.

---

## Datos

`manifests/fedmammobench_norm_{0_1,neg1_1}.csv`: 8.341 imágenes, split **6.671 / 836 / 834**
(train / val / test), paciente-disjunto. Variantes `*_local.csv` apuntan a otra raíz de imágenes;
`manifests/by_database/` tiene un manifest por nodo.

| base | train | val | test | % maligno |
|---|---:|---:|---:|---:|
| cmmd | 3.776 | 474 | 472 | 48,9 % |
| kau-bcmd | 1.764 | 220 | 222 | 4,3 % |
| cdd-cesm | 803 | 100 | 100 | 33,3 % |
| inbreast | 328 | 42 | 40 | 24,4 % |

Preparación y columnas requeridas: [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md). Contexto de
cada dataset: [docs/DATASETS_CONTEXTO.md](docs/DATASETS_CONTEXTO.md).

---

## Experimentos

| rango | qué |
|---|---|
| exp01–39, 57–60 | Centralizados: backbone, head, congelamiento, pérdidas, anti-sobreajuste, búsqueda de hiperparámetros (exp37–39), ablaciones de pre-entrenamiento |
| exp40–56 | Federados: FedAvg / FedProx / FedAdam / FedYogi × rondas y barrido de `mu` |
| exp61–69 | ResNet18/ResNet50 desde cero, ResNet18 ImageNet con descongelamiento progresivo, CNN propia |
| `fedisic2019_*` | Grid federado Fed-ISIC2019 |

Resultado de referencia centralizado: exp37 (AUC test 0,9016, F1-macro 0,8071). Hallazgo importante: a
nivel **paciente**, exp37 vs exp28 no es estadísticamente significativo (p = 0,933) — el ranking del
sweep está inflado por medir AUC por imagen. El detalle de resultados está en
[`.claude/context/experiments/`](.claude/context/experiments/) (`CENTRALIZED.md`, `FEDERATED.md`);
auditorías y decisiones en [`docs/`](docs/).

---

## Documentación

- [`CLAUDE.md`](CLAUDE.md) — enrutador de contexto y mapa de qué documentación es vigente y cuál obsoleta.
- Contratos por método: [`src/DOCS.md`](src/DOCS.md), [`src/datasets/DOCS.md`](src/datasets/DOCS.md),
  [`src/models/DOCS.md`](src/models/DOCS.md), [`src/train/DOCS.md`](src/train/DOCS.md),
  [`src/federated/DOCS.md`](src/federated/DOCS.md).
- Diseño federado: [docs/FEDERATED_DESIGN.md](docs/FEDERATED_DESIGN.md).
- Metodología: [docs/METHODOLOGY.md](docs/METHODOLOGY.md) · Sobreajuste:
  [docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md](docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md).
- [`PHASES.md`](PHASES.md) — qué se portó del proyecto INC y por qué cada default es el que es.

## Estructura

```
src/
├── cli.py, evaluate.py, gradcam.py   entrypoints
├── config.py, eval_pipeline.py       esquema Pydantic y ensamblaje compartido
├── datasets/                         manifest, split paciente-disjunto, transforms, dataloaders
├── models/                           backbones, heads, congelamiento, carga de pesos
├── train/                            Trainer, loop, pérdidas, early stopping
└── federated/                        servidor, cliente, estrategias (Flower sobre gRPC)
configs/                              expNN_*.yaml (centralizado) y federated/<exp>/ (server + nodos)
manifests/                            CSV de datos y splits
scripts/                              post-hoc: calibración de umbral, ensembles, auditorías, registro
docs/                                 diseño, auditorías y briefs
runs/{centralizado,federado}/         salidas
```

No hay suite de tests ni CI de código; el único workflow (`.github/workflows/docker-publish.yml`)
publica la imagen de entorno en Docker Hub con un tag `v*.*.*`. La verificación se hace manejando el
código real con tensores sintéticos (ver `PHASES.md`).

## Licencia

MIT — ver [LICENSE](LICENSE).
