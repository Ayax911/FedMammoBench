# Documentación del Módulo Federado (`src/federated/`)

`src/federated/` implementa el despliegue federado real (gRPC, sin simulación) de FedMammoBench sobre Flower: un proceso servidor sin datos (`server.py`) + un proceso por nodo (`client.py`), cada uno con su propio YAML validado por `config.py`. Racional completo y decisiones frente al paquete legacy borrado: `docs/FEDERATED_DESIGN.md`. Este documento es el contrato por método/clase, en el mismo formato que los `DOCS.md` hermanos.

---

## Flujo Típico de Uso

```bash
# Servidor -- nunca ve imágenes.
.venv/bin/python -m src.federated.server --config configs/federated/exp40_fedavg_full/server.yaml

# Un proceso por nodo, cada uno con su YAML.
.venv/bin/python -m src.federated.client --config configs/federated/exp40_fedavg_full/node_cmmd.yaml
.venv/bin/python -m src.federated.client --config configs/federated/exp40_fedavg_full/node_kau-bcmd.yaml
.venv/bin/python -m src.federated.client --config configs/federated/exp40_fedavg_full/node_cdd-cesm.yaml
.venv/bin/python -m src.federated.client --config configs/federated/exp40_fedavg_full/node_inbreast.yaml

# Fallback: re-evaluar un nodo contra el mejor modelo global sin reentrenar
# (nodo que murió a mitad de corrida, o re-generar val/test después).
.venv/bin/python -m src.federated.evaluate_node \
    --config configs/federated/exp40_fedavg_full/node_cmmd.yaml \
    --server-run-dir runs/exp40_fedavg_full/server
```

Con Docker (ver `docker-compose.federated.yaml`):

```bash
EXPERIMENT=exp40_fedavg_full docker compose -f docker-compose.federated.yaml up
```

---

## Detalle por Archivo y Clase

### `config.py`

#### `FederatedServerConfig` / `load_server_config(path)`

Valida `server.yaml`: `federation` (transporte/rondas/`aggregation_scope`), `strategy` (`NamedComponentConfig`, nombre en `strategies._STRATEGIES`), `architecture` + `head` (modelo plantilla, duplicados textualmente en cada `node_*.yaml`), `tracking` (mejor-modelo + artefactos). **Sin** `data`/`loss`/`optimizer`/`scheduler` -- el servidor no procesa imágenes ni computa pérdida.

#### `FederatedNodeConfig` / `load_node_config(path)`

Valida `node_<nombre>.yaml`: mismo `architecture`/`head` que el servidor (hash-verificado en runtime, no en carga), `optimizer`/`scheduler`/`loss`/`data` LOCALES del nodo, `run: NodeRunConfig` (variante recortada de `TrainConfig` sin `epochs`/`patience`/`metric_name` -- esas son decisiones del servidor). Validación extra: `data.by_database_manifests` debe ser `None` (`ValueError` si no) y `data.num_workers > 0` emite un `UserWarning` (fork de DataLoader workers vs. canal gRPC abierto).

#### `save_federated_config(config, path)`

Snapshot YAML de un `FederatedServerConfig`/`FederatedNodeConfig`, mismo patrón que `src.config.save_config`.

##### Cómo usar `config.py`:
```python
from src.federated.config import load_server_config, load_node_config

server_cfg = load_server_config("configs/federated/exp40_fedavg_full/server.yaml")
node_cfg = load_node_config("configs/federated/exp40_fedavg_full/node_cmmd.yaml")
print(server_cfg.federation.rounds, node_cfg.node_name)
```

---

### `handshake.py`

#### `model_config_hash(experiment_id, architecture, head, aggregation_scope)`

SHA-256 (16 hex) sobre `{experiment_id, architecture.name, architecture.unfreeze_from, head.name, head.hparams, aggregation_scope}` -- lo que define la FORMA del modelo. Excluye `weights_path` (ruta local, los pesos efectivos vienen del servidor) y `loss`/`optimizer`/`data` (locales por nodo). El servidor lo publica en el config de cada ronda; el nodo lo recalcula de su propio YAML y aborta (`RuntimeError`) si no coincide -- el mecanismo que el paquete legacy escribió (`model_config_hash()`) y nunca conectó.

##### Cómo usar `handshake.py`:
```python
from src.federated.handshake import model_config_hash

h = model_config_hash(cfg.experiment_id, cfg.architecture, cfg.head, cfg.aggregation_scope)
```

---

### `param_utils.py`

#### `scope_module(model, scope)` / `get_model_ndarrays(model, scope)` / `set_model_ndarrays(model, arrays, scope)`

Conversión `state_dict()` <-> `list[np.ndarray]` consciente de `aggregation_scope`: `"full"` serializa `model` completo, `"backbone"` solo `model[0]`. `set_model_ndarrays` verifica largo y shape por clave ANTES de cargar, y castea cada array al `dtype` del tensor destino -- el fix del buffer `num_batches_tracked` (int64), que la agregación de Flower promedia como float y devolvería float64 sin este cast.

##### Cómo usar `param_utils.py`:
```python
from src.federated.param_utils import get_model_ndarrays, set_model_ndarrays

arrays = get_model_ndarrays(model, "full")       # -> list[np.ndarray], orden de state_dict()
set_model_ndarrays(model, arrays, "full")        # in-place, strict
```

---

### `strategies.py`

#### `build_strategy(name, *, initial_parameters, num_nodes, on_config, accept_failures, **hparams)`

Construye una estrategia STOCK de `flwr.server.strategy` (`fedavg`, `fedprox`, `fedadam`, `fedyogi` -- dict plano `_STRATEGIES`, sin decoradores) pre-cableada: participación total (`min_* = num_nodes`), `fit/evaluate_metrics_aggregation_fn = weighted_average`, `on_fit_config_fn = on_evaluate_config_fn = on_config`. `hparams` se splatea al constructor stock (`proximal_mu`, `eta`, `beta_1`, ...).

#### `weighted_average(metrics)`

Promedio ponderado por `num_examples`, POR CLAVE (una métrica ausente/NaN en un nodo no arrastra a las demás), filtrando NaN y reportes con `num_examples == 0`. **El resultado NO es la métrica del pool combinado** -- ver `docs/FEDERATED_DESIGN.md`.

#### `make_on_config(model_hash, local_epochs, aggregation_scope)`

Fábrica del `on_fit_config_fn`/`on_evaluate_config_fn`: el canal del handshake (`model_hash` viaja en el config de CADA ronda).

##### Cómo usar `strategies.py`:
```python
from flwr.common import ndarrays_to_parameters
from src.federated.strategies import build_strategy, make_on_config

on_config = make_on_config(model_hash, local_epochs=2, aggregation_scope="full")
strategy = build_strategy("fedprox", initial_parameters=ndarrays_to_parameters(arrays),
                          num_nodes=4, on_config=on_config, accept_failures=False,
                          proximal_mu=0.1)
```

---

### `assembly.py`

#### `NodeAssembly` / `build_node_assembly(config)`

Ensamblado del lado nodo (manifest -> split -> loaders -> `nn.Sequential(backbone, head)` -> `LossSpec`), factorizado UNA vez del bloque de reconstrucción de `evaluate.py:run_evaluation()` para que `client.py` y `evaluate_node.py` lo reusen idéntico. NO llama a `set_global_seed()` -- responsabilidad del entrypoint, una sola vez por proceso.

##### Cómo usar `assembly.py`:
```python
from src.federated.assembly import build_node_assembly

assembly = build_node_assembly(node_cfg)
assembly.model, assembly.loaders["train"], assembly.n_train
```

---

### `client.py`

#### `FedMammoBenchClient(flwr.client.NumPyClient)`

Un nodo. `fit()`: handshake -> `set_model_ndarrays` -> optimizer/scheduler FRESCOS cada ronda (vía `train.build.build_param_groups`, honra `backbone_lr`) -> término proximal de FedProx (si `config["proximal_mu"] > 0`) inyectado a `train_one_epoch(regularizer=...)` -> `local_epochs` x (`train_one_epoch` + `evaluate`), una fila de `metrics.csv` por época local -> (alcance `backbone`) guarda `weights/round<r>_head.pt` -> devuelve `(ndarrays, n_train, métricas)`. `evaluate()`: handshake -> carga -> `evaluate()` sobre val local -> fila en `rounds.csv` -> devuelve `(loss, n_val, métricas)`. Usa `train_one_epoch`/`evaluate` (funciones puras de `train/loop.py`), NO `Trainer.fit()` -- ver `docs/FEDERATED_DESIGN.md` §Cliente para el porqué.

**`metrics.csv` del nodo** -- una fila por ÉPOCA LOCAL, claves: `round`, `train_<7 métricas>`, `train_loss`, `train_prox_loss` (solo si la estrategia es FedProx con `proximal_mu > 0` -- presente TODA la corrida o NUNCA, nunca a medias), `val_<7 métricas>`, `val_loss`, `duration_seconds`.

**`rounds.csv` del nodo** -- una fila por RONDA, claves fijas: `round, num_examples, loss, accuracy, auc, sensitivity, specificity, f1, f1_macro, precision` (las métricas del modelo AGREGADO sobre el val local, lo que el nodo reportó al servidor esa ronda).

##### Cómo usar `client.py`:
```bash
.venv/bin/python -m src.federated.client --config configs/federated/exp40_fedavg_full/node_cmmd.yaml
```

---

### `round_tracking.py`

#### `TrackedStrategy(flwr.server.strategy.Strategy)`

Envuelve una estrategia stock por COMPOSICIÓN (no monkey-patching): delega los 6 métodos de `Strategy` al `inner`, y sobre `aggregate_fit`/`aggregate_evaluate` agrega logging por ronda (`metrics.csv` del servidor, `epoch` = número de ronda) y tracking de mejor ronda (`train.early_stopping.EarlyStopping(patience=None)` usado SOLO como tracker -- nunca activa `should_stop`; las rondas son fijas). En cada mejora: `set_model_ndarrays` sobre el modelo plantilla + `checkpoint.save_checkpoint` (`best_round<N>.pt`, con `_backbone.pt`/`_head.pt` extra en alcance `full`) + reescribe `best.json`.

**`best.json`** -- `{best_round, metric_name, metric_value, checkpoint, aggregation_scope, model_hash}`. Se reescribe en CADA mejora, así que al terminar la última ronda está garantizado final -- sin protocolo de señal aparte para que los nodos sepan cuándo leerlo.

##### Cómo usar `round_tracking.py`:
```python
from src.federated.round_tracking import TrackedStrategy

tracked = TrackedStrategy(inner_strategy, logger=logger, template_model=model, scope="full",
                          model_hash=h, best_metric_name="auc", best_metric_mode="max",
                          checkpoint_dir=ckpt_dir, run_dir=run_dir, save_every_rounds=None)
```

---

### `server.py`

#### `run_server(config)`

Arma el modelo plantilla (`architecture` + `head`, semilla fija `_TEMPLATE_SEED=42` solo para la inicialización de la cabeza -- el servidor no tiene `data.seed`), los parámetros iniciales en el alcance activo, la estrategia envuelta en `TrackedStrategy`, y llama `flwr.server.start_server(...)`. Al terminar (o si `start_server` levanta), genera `plots/` desde `tracked.round_history` y cierra el logger -- en un `finally`, así un fallo a mitad de corrida no pierde los plots de las rondas que sí completaron.

##### Cómo usar `server.py`:
```bash
.venv/bin/python -m src.federated.server --config configs/federated/exp40_fedavg_full/server.yaml
```

---

### `evaluate_node.py`

#### `run_final_evaluation(config, server_run_dir, logger=None, assembly=None)`

Evalúa el MEJOR modelo global (`best.json`) sobre val/test del nodo, vía `eval_pipeline.evaluate_split()` sin tocarlo. Alcance `full`: usa el checkpoint global tal cual. Alcance `backbone`: compone el mejor backbone global + la cabeza LOCAL que el nodo entrenó en esa misma ronda (`weights/round<N>_head.pt`), guarda `weights/final_round<N>.pt`. `logger`/`assembly` opcionales para la ruta en-proceso (reusa lo que `client.py` ya construyó); `None` (fallback standalone) los construye de cero. Nunca toca `metrics.csv`/`rounds.csv`/`config.yaml` del nodo -- `MetricsLogger` abre sus writers perezosamente, y acá nunca se llama `log()`.

Espera hasta 60s (poll cada 2s) a que `best.json` exista antes de levantar `FileNotFoundError` -- cubre el caso patológico de un servidor muerto antes de la primera mejora.

##### Cómo usar `evaluate_node.py`:
```bash
.venv/bin/python -m src.federated.evaluate_node \
    --config configs/federated/exp40_fedavg_full/node_cmmd.yaml \
    --server-run-dir runs/exp40_fedavg_full/server
```
