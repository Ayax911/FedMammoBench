# `src/federated/` — despliegue federado real sobre Flower

Contratos por método con ejemplos: **`src/federated/DOCS.md`**. Diseño completo, cada decisión mapeada
contra el dolor que arregla del paquete federado legacy: **`docs/FEDERATED_DESIGN.md`**. Resultados
medidos: [../experiments/FEDERATED.md](../experiments/FEDERATED.md).

Idioma: todo `src/federated/` está en **español**.

---

## Qué es y qué no es

**gRPC real, no simulación.** Un proceso servidor (que **nunca ve imágenes**) más un proceso por nodo,
cada uno con su propio YAML. Sin backend de Ray, sin `flwr` simulation. `flwr==1.31.0` está pinneado
exacto en `requirements.txt` — lee el comentario encima antes de subirlo.

### Reglas de capa

- `src/federated/` está **al mismo nivel** que `cli.py`/`evaluate.py`: importa todo lo que está encima
  de `eval_pipeline.py` en la cadena de dependencias, **nunca `cli.py`**.
- **Nada fuera de `federated/` importa desde `federated/`.**
- **Todo contacto con la API de `flwr` está confinado a `server.py` y `client.py`.** El resto de los
  módulos no conoce Flower. Mantenlo así: es lo que hace testeable/legible el resto.

---

## Cómo se lanza

```bash
.venv/bin/python -m src.federated.server --config configs/federated/exp40_fedavg_full/server.yaml
.venv/bin/python -m src.federated.client --config configs/federated/exp40_fedavg_full/node_cmmd.yaml
# ... un proceso cliente por nodo (kau-bcmd, cdd-cesm, inbreast)
```

O vía Docker, que es la forma primaria en la workstation — servidor + todos los contenedores de nodo
en un solo comando, con host networking para que `server_address: 127.0.0.1:<port>` funcione idéntico
dentro y fuera de los contenedores:

```bash
EXPERIMENT=exp40_fedavg_full docker compose -f docker-compose.federated.yaml up
```

`docker-compose.federated.yaml` **reemplaza** a `/docker-run` y `/docker-queue` para lo federado.
El `Dockerfile` construye una imagen **solo-entorno** (Python 3.12 + `requirements.txt`, sin copiar
código): el repo se monta en `/workspace`, así que **un cambio de código nunca necesita rebuild**.

Re-evaluar un nodo contra el mejor modelo global sin reincorporarse a una corrida viva:

```bash
.venv/bin/python -m src.federated.evaluate_node --config <node.yaml> --server-run-dir runs/<exp>/server
```

Comprobación de imports:
```bash
.venv/bin/python -c "import src.federated.server; import src.federated.client; import src.federated.evaluate_node"
```

---

## Los módulos

| módulo | rol |
|---|---|
| `config.py` | `FederatedServerConfig` / `FederatedNodeConfig` (Pydantic, `extra="forbid"` igual que el central), `load_server_config()`, `load_node_config()` |
| `server.py` | `run_server()`, arma el modelo plantilla y arranca Flower. Toca `flwr`. |
| `client.py` | `FedMammoBenchClient(fl.client.NumPyClient)`. Toca `flwr`. |
| `strategies.py` | `_STRATEGIES` (dict plano), `build_strategy()`, `weighted_average()`, `make_on_config()` |
| `assembly.py` | `build_node_assembly()` → `NodeAssembly`: reusa el ensamblaje centralizado en cada nodo |
| `param_utils.py` | `scope_module()`, `get_model_ndarrays()`, `set_model_ndarrays()` — el puente modelo ↔ lista de arrays de Flower |
| `handshake.py` | `model_config_hash()` — la verificación de que todos los nodos corren el mismo modelo |
| `round_tracking.py` | `TrackedStrategy`, que envuelve la estrategia y escribe `metrics.csv`, `best.json` y los checkpoints globales |
| `evaluate_node.py` | re-evaluación offline de un nodo contra el mejor global |

**Sin matemática de agregación propia.** Las cuatro estrategias son las clases stock de
`flwr.server.strategy` (`FedAvg`, `FedProx`, `FedAdam`, `FedYogi`), elegidas por nombre desde el YAML
con el mismo patrón `NamedComponentConfig` que optimizer/loss. Lo único a medida es
`weighted_average()` y `make_on_config()`.

`fedbn`/`scaffold` **no están registrados** a propósito: el legacy los registraba como stubs que
lanzaban `NotImplementedError` al construir, y registrar lo que no existe solo pospone el error.

---

## `aggregation_scope: full | backbone`

En `server.yaml`. Decide si se agrega el modelo entero o solo el backbone, reflejando la separación
`_backbone.pt`/`_head.pt` que los checkpoints centralizados ya producen (ver [MODELS.md](MODELS.md)).
Todo el grid ejecutado usa `full`.

---

## Artefactos

Cada nodo escribe su propio `runs/<exp>/nodes/<node_name>/`, **con la forma exacta de un `run_dir`
centralizado** (`config.yaml`, `metrics.csv`, `plots/`, `val/`, `test/`). El servidor escribe
`runs/<exp>/server/` (`metrics.csv` por ronda, `best.json`, checkpoints globales).

`round_tracking._save_best()` guarda el checkpoint global con el **mismo**
`src.checkpoint.save_checkpoint()` que usan las corridas centralizadas — por eso `src.evaluate` lo
carga directo, sin conversión.

---

## Invariantes

### `weighted_average()` pondera **por clave**, no por nodo entero
Cada métrica se promedia sobre los nodos que la reportaron con un valor numérico no-NaN. Así un nodo
que omite (o NaN-ea) `auc` porque su val local quedó monoclase **no arrastra a cero las demás
métricas**. Un reporte con `num_examples == 0` queda excluido de todo — ese es el mecanismo documentado
para que un nodo se auto-excluya. Los valores no numéricos (ej. `node_name`) se ignoran.

### El AUC agregado **no** es el AUC del pool
Media de métricas ≠ métrica de la unión. `best.json` guarda el promedio ponderado por nodo, que aquí
va **~0,02 inflado** porque kau-bcmd (4,3 % de malignos) regala specificity. **Nunca cites una brecha
federado-vs-centralizado desde `best.json`**: re-evalúa el checkpoint global pooled con `src.evaluate`.
Los números están en [../experiments/FEDERATED.md](../experiments/FEDERATED.md).

### Los defaults de `fedadam`/`fedyogi` no dan "peor", dan muerto
Con `tau=1e-9` (stock) fedadam da `val_loss=NaN` desde la ronda 1 en todos los nodos; fedyogi con
`eta=0.01/tau=1e-3` colapsa a `val_auc=0.5000` exacto en la ronda 6. Este es un fine-tuning con
backbone-lr ~2e-4: pseudo-gradientes chicos, régimen para el que esas estrategias no fueron diseñadas.
Valores que funcionan, ya baked en `configs/federated/exp47-52`: **fedadam `eta=0.01, tau=1e-3`;
fedyogi `eta=1e-3, tau=0.01`**. Si ves NaN o un 0,5000 exacto, mata la corrida y ajusta eta/tau en vez
de gastar el presupuesto de rondas.

### El handshake por hash
`make_on_config()` cablea el hash de config de modelo (`handshake.model_config_hash()`) en el config de
**cada** ronda. El legacy escribió esa pieza y nunca la conectó, así que nodos con arquitecturas
distintas podían agregarse entre sí en silencio. Está conectada; no la desconectes.
