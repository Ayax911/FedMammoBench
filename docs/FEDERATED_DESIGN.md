# Diseño federado: Flower sobre gRPC real (`src/federated/`)

Este documento registra el diseño de la capa federada de FedMammoBench: por qué Flower, por qué
despliegue real y no simulación, cómo se estructura `src/federated/`, y qué decisión de diseño
responde a qué problema documentado en el paquete legacy (`src/fedmammobench/federated/`, borrado
del árbol, consultable en `git show ec55408:...`). Mismo espíritu que `PHASES.md`: registra el
*porqué*, no solo el *qué*.

`main` sigue teniendo `src/` centralizado como scope primario (ver "The one command" en CLAUDE.md);
este documento cubre la extensión federada que se agrega ENCIMA, sin tocar esa capa.

---

## 0. Por qué Flower, y qué significa "sin simulación"

**Framework.** Flower fue la elección explícita del usuario, y también la que mejor encaja con lo
que ya existe en este proyecto:

- Es agnóstico a PyTorch: `train_one_epoch()`/`evaluate()` (`src/train/loop.py`) se conectan sin
  modificar su forma.
- Trae FedAvg/FedProx/FedAdam/FedYogi como clases stock (`flwr.server.strategy`) — sin escribir
  matemática de agregación propia.
- Los experimentos legacy reales (`exp70`, `caladam01`, ver `ec55408:configs/`) ya corrieron sobre
  Flower — mantenerlo preserva comparabilidad metodológica y el conocimiento acumulado en sus
  auditorías (`ec55408:docs/audit/`).
- Es el framework de FL generalista con desarrollo más activo hoy.

Alternativas consideradas y descartadas para este proyecto: **NVIDIA FLARE** es la opción más seria
para un despliegue clínico multi-hospital real (provisioning, TLS, consola de administración,
integración MONAI) pero es sobre-ingeniería para un benchmark de 4 nodos en una sola workstation —
es la opción a revisitar si este proyecto pasa a un despliegue multi-institucional real. **OpenFL**
(Intel, nacido para FL médico) tiene comunidad y ritmo de desarrollo menores. **FedML** es de
alcance más amplio pero cada vez más atado a su propia plataforma. **PySyft** resuelve un problema
distinto (cómputo remoto preservando privacidad, no orquestación de rondas de entrenamiento). El
costo real de Flower es el churn de su API (`start_server`/`start_client` deprecados en la línea
1.x en favor de SuperLink/SuperNode) — ver §1 y el riesgo correspondiente en §9.

**"Sin simulación" — qué significa exactamente.** Flower tiene dos modos de ejecución:

- **Simulación** (Simulation Engine, respaldado por Ray): un solo programa donde el servidor y
  todos los clientes son virtuales — Flower los lanza como tareas dentro de un proceso/clúster, con
  los datos particionados en memoria. Sirve para prototipar rápido o escalar a cientos de clientes
  ficticios en una sola GPU. **No se usa acá, en ningún punto.**
- **Despliegue** (el modo elegido): servidor y cada nodo son procesos de sistema operativo
  SEPARADOS, cada uno lanzado a mano con su propio YAML, comunicándose por gRPC real. Nada comparte
  memoria; un nodo solo conoce su propio manifest.

Esto se cumple incluso en la verificación de humo (§8): los procesos ahí también son reales,
separados, sobre gRPC real en `localhost` — la única diferencia con una corrida real es la escala
(2 nodos sintéticos en vez de 4 nodos con datos reales), no el mecanismo.

---

## 1. Elección de API de Flower

Se fija **`flwr==1.31.0`** (mismo pin que corrió el paquete legacy, según `ec55408:requirements.txt`
salvo por la versión exacta — ver abajo) usando `flwr.server.start_server()` +
`flwr.client.start_client()`, deprecadas mostrando un `DeprecationWarning` pero funcionales, en vez
de migrar a `flower-superlink`/`flower-supernode` (la ruta moderna).

**Por qué no SuperLink/SuperNode:** esa ruta requiere una app empaquetada con `pyproject.toml` y el
comando `flwr run`, lo que rompe tres convenciones del repo a la vez: no existe `pyproject.toml` en
este árbol (decisión deliberada, ver CLAUDE.md), todo corre como `.venv/bin/python -m src.<módulo>`,
y los YAML son archivos únicos sin herencia (los proyectos Flower empaquetados empujan config hacia
`[tool.flwr]` en `pyproject.toml`). El costo de quedarse en la API deprecada se contiene
deliberadamente: **todos** los puntos de contacto con `flwr` viven en dos archivos,
[`server.py`](../src/federated/server.py) y [`client.py`](../src/federated/client.py) — una futura
migración toca solo esos dos.

**Verificado en la Fase 0** (spike desechable, `flwr==1.18.0` inicial resultó incompatible con el
`protobuf` que fuerza `tensorboard`/`wandb` — se subió a `1.31.0`, que sí convive con
`protobuf>=6.31.1`, ver [`requirements.txt`](../requirements.txt)): 1 servidor + 2 clientes reales
sobre gRPC en `localhost`, 2 rondas, confirmando `start_server`/`start_client`/`to_client()` y que
`start_client()` **retorna limpio** (no levanta excepción) cuando el servidor cierra la conexión al
terminar las rondas — la premisa de la que depende la evaluación final en `client.py` (§6).

---

## 2. Estructura del módulo

`src/federated/` se sienta en la misma capa que `cli.py`/`evaluate.py` en la cadena de dependencias
de CLAUDE.md: importa de `config`, `seed`, `checkpoint`, `tracking`, `reporting`, `datasets/`,
`models/`, `train/`, `eval_pipeline` — nunca de `cli.py` — y nada fuera de `federated/` importa de
acá.

```
src/federated/
  config.py         # Pydantic: FederatedServerConfig / FederatedNodeConfig + load/save
  handshake.py       # model_config_hash() -- el handshake servidor<->nodos
  param_utils.py     # state_dict <-> list[ndarray], consciente del alcance de agregación
  strategies.py       # registro plano {fedavg,fedprox,fedadam,fedyogi} + weighted_average()
  round_tracking.py   # TrackedStrategy: logging + mejor-ronda POR COMPOSICIÓN
  assembly.py         # build_node_assembly() -- ensamblado del lado nodo, una sola vez
  client.py          # FedMammoBenchClient + main()  -> python -m src.federated.client
  server.py          # run_server() + main()          -> python -m src.federated.server
  evaluate_node.py    # evaluación final del nodo contra el mejor modelo global
  DOCS.md             # contratos por método (formato hermano de src/*/DOCS.md)
```

Contratos completos por método: [`src/federated/DOCS.md`](../src/federated/DOCS.md).

---

## 3. Config: server.yaml + node_\*.yaml, sin herencia

**Input requerido por el usuario**: un YAML por nodo + un YAML para el servidor — el servidor NUNCA
procesa imágenes. `FederatedServerConfig` (`src/federated/config.py`) por eso no tiene sección
`data`, ni `loss`, ni `optimizer`/`scheduler`: solo `federation` (transporte/rondas/alcance),
`strategy`, `architecture` + `head` (el modelo plantilla que genera los parámetros iniciales) y
`tracking` (mejor-modelo + artefactos). `FederatedNodeConfig` es el equivalente por-nodo de
`ExperimentConfig`, con una variante recortada de `TrainConfig` (`NodeRunConfig`, sin
`epochs`/`patience`/`metric_name`: esas son decisiones del servidor a granularidad de ronda, no del
nodo a granularidad de época).

Mismo principio que `src/config.py`: `extra="forbid"` en todo, **sin herencia entre YAMLs** — cada
archivo se lee de punta a punta. Esto significa que `architecture`/`head` quedan **duplicados
textualmente** entre `server.yaml` y cada `node_*.yaml`. La alternativa (una sola fuente que ambos
importen) fue descartada a propósito: rompería la convención "cada YAML es autocontenido" que el
resto del repo ya sigue. En cambio, la coherencia se vigila en RUNTIME — ver §4.

Ejemplo completo de las 5 YAMLs de un experimento de 4 nodos:
[`configs/federated/exp40_fedavg_full/`](../configs/federated/exp40_fedavg_full/).

---

## 4. El handshake por hash — el problema que el legacy nunca resolvió

El paquete legacy escribió `model_config_hash()` (`ec55408:src/fedmammobench/federated/server.py`)
explícitamente para esto — comparar la config de servidor y nodos antes de entrenar — y **nunca la
conectó a nada**. La única defensa real eran comentarios en el YAML ("DEBE coincidir con
server.yaml"), que no detienen absolutamente nada: un nodo con otra cabeza, otro `unfreeze_from`, u
otro `aggregation_scope` entrena una forma de modelo distinta, y el error se manifiesta tarde (un
largo de lista de ndarrays que no calza) o, peor, nunca — la agregación simplemente produce un
modelo sin sentido.

Acá el hash SÍ viaja: [`handshake.model_config_hash()`](../src/federated/handshake.py) resume
`{experiment_id, architecture.name, architecture.unfreeze_from, head.name, head.hparams,
aggregation_scope}` en un SHA-256 truncado a 16 hex. El servidor lo publica en el config de **cada**
ronda (`strategies.make_on_config`, vía `on_fit_config_fn`/`on_evaluate_config_fn`); cada nodo
recalcula el suyo en `__init__` y compara en cada `fit()`/`evaluate()`, levantando `RuntimeError`
con ambos hashes en el mensaje al primer desacuerdo. Con `accept_failures: false` (default) eso
aborta toda la corrida en la ronda 1, en vez de entrenar un Frankenstein en silencio.

Qué queda AFUERA del hash a propósito: `weights_path` (ruta local, puede diferir por máquina — los
pesos efectivos llegan del servidor en la ronda 1, salvo para depuración sin servidor) y
`loss`/`optimizer`/`scheduler`/`data` (locales por nodo: `pos_weight` se calcula sobre el train
split de CADA nodo, y kau-bcmd ~4.6% maligno vs. cmmd ~49% son legítimamente distintos).

---

## 5. Alcance de agregación: `full` vs `backbone`

Configurable por experimento (`federation.aggregation_scope` en el servidor, duplicado y
hash-verificado en cada nodo):

- **`full`**: agrega el `state_dict()` completo de `nn.Sequential(backbone, head)` — FedAvg
  estándar, comparable con las corridas centralizadas y con el `exp70` legacy.
- **`backbone`**: solo `model[0]` viaja y se agrega; cada nodo conserva una cabeza personalizada
  local. Es la intención detrás de que `checkpoint.save_checkpoint()` ya separe
  `best_epochN_backbone.pt`/`_head.pt` en el pipeline centralizado (ver CLAUDE.md, "invariantes").

[`param_utils.py`](../src/federated/param_utils.py) implementa la conversión
`state_dict()` <-> `list[np.ndarray]` en el alcance activo. Dos detalles que el legacy dejaba
implícitos:

- **El fix del buffer `num_batches_tracked`.** Es un tensor `int64` dentro del `state_dict` de cada
  BatchNorm. FedAvg promedia TODO lo que recibe, incluidos los buffers (comportamiento stock de
  Flower, conservado por comparabilidad) — y un promedio de enteros en Python/NumPy vuelve float64.
  `set_model_ndarrays()` castea siempre al `dtype` del tensor destino antes de `load_state_dict`, o
  la carga falla o degrada en silencio según la versión de PyTorch. Verificado explícitamente en la
  Fase 2 (round-trip bit-exacto de un ResNet50 real, en ambos alcances, incluidos los 53-54 buffers
  `int64` de cada corrida).
- **FedAdam/FedYogi sobre buffers.** Sus momentos adaptativos se mantienen sobre TODO lo que la
  estrategia agrega, incluidos los buffers de BN — no solo los pesos entrenables. Es el
  comportamiento stock de esas clases de Flower; se conserva y se documenta acá en vez de
  filtrarlo, para no introducir una estrategia "custom" donde el diseño pide reusar las stock.

---

## 6. Cliente: por qué `train_one_epoch`, no `Trainer.fit()`

[`client.py`](../src/federated/client.py) reutiliza las funciones puras de `train/loop.py`
directamente, NO `Trainer.fit()`. La razón: `Trainer` empaqueta checkpointing por-época,
`EarlyStopping` y "mejor época" — todas decisiones de granularidad ÉPOCA que en este diseño
pertenecen al SERVIDOR a granularidad RONDA (ver §7). Reusar `Trainer` habría exigido suprimir su
manejo de logger propio, su `RuntimeError` en corridas de 0 épocas, y sus efectos secundarios de
checkpoint — más superficie que las dos funciones puras que envuelve.

**Optimizador y scheduler frescos cada ronda.** Tras el reemplazo server-side de los pesos, los
momentos de Adam de la ronda anterior apuntan a un paisaje de pérdida que ya no existe — arrastrarlos
optimizaría contra estado obsoleto. Es la práctica estándar en Flower y también la que usaba el
legacy. Consecuencia documentada en los YAML de ejemplo: un `scheduler` tipo `cosine` con `T_max`
pensado para la corrida completa solo abarcaría `local_epochs` épocas si se reconstruye cada ronda
— por eso los ejemplos usan `scheduler: null`.

**`backbone_lr` extraído a una función compartida.** La lógica de LR discriminativo backbone/cabeza
vivía inline en `cli.py:132-141`. Se extrajo a
[`train/build.py:build_param_groups()`](../src/train/build.py) — de lo contrario, un cliente
federado con `backbone_lr` en su YAML lo habría perdido en silencio, entrenando con un solo LR
plano sin ningún error que lo señale. `cli.py` ahora llama a la misma función (comportamiento
idéntico, verificado en la Fase 2).

**FedProx: dónde vive el término proximal.** `LossSpec.compute` (`train/build.py`) no ve el modelo
— solo logits y labels — así que el término `(mu/2)·‖w - w_global‖²` no puede vivir ahí.
`train_one_epoch()` ganó un parámetro opcional `regularizer: Callable[[], Tensor] | None` (default
`None` = comportamiento byte-idéntico al de antes de este parámetro). El backward corre sobre
`task_loss + regularizer()`, pero la clave `"loss"` devuelta sigue siendo SOLO la task loss —
comparable entre estrategias — y el término penal se reporta aparte como `"prox_loss"`. Esto evita
el defecto documentado en la auditoría del legacy (N8: reportar `train_loss` con la penalización
incluida contamina cualquier comparación FedAvg-vs-FedProx). También se evita el otro defecto de esa
auditoría (N7: el término proximal calculado bajo `torch.cuda.amp.autocast()` puede hacer underflow
en FP16 y volver a FedProx indistinguible de FedAvg) simplemente porque este pipeline no usa
mixed-precision en ningún punto.

**Artefactos del nodo** (`runs/<exp>/nodes/<node_name>/`, escritos SOLO por el proceso del nodo):
`config.yaml`, `metrics.csv` (una fila por ÉPOCA LOCAL), `rounds.csv` (una fila por RONDA, la
calidad del modelo agregado sobre el val local), `train.log`, `plots/`, y — al terminar — `val/` +
`test/` con la misma forma que un `run_dir` centralizado. Ver claves exactas en
[`src/federated/DOCS.md`](../src/federated/DOCS.md).

---

## 7. Servidor: composición en vez de monkey-patching, rondas fijas en vez de excepción

El legacy le pegaba comportamiento a una estrategia stock reasignando métodos bound
(`strategy.aggregate_evaluate = wrapped`), apilando tres wrappers en un orden frágil documentado a
mano en comentarios. [`round_tracking.TrackedStrategy`](../src/federated/round_tracking.py)
implementa los 6 métodos de la interfaz `Strategy` de Flower **delegando** al `inner` (la clase
stock de `strategies.build_strategy()`) para toda la matemática de agregación, y solo envuelve
`aggregate_fit`/`aggregate_evaluate` con lo que este proyecto necesita encima: registrar la ronda,
decidir si es la mejor vista, persistir el checkpoint.

El legacy también implementaba early stopping FEDERADO levantando una excepción
(`EarlyStoppingTriggered`) desde dentro de la estrategia, que se des-envolvía DISTINTO según el
transporte (simulación vs. gRPC real) — una verruga que su propia auditoría señala. Acá las rondas
son **fijas** (`federation.rounds`) y lo único que existe es un TRACKER de mejor ronda
(`train.early_stopping.EarlyStopping(patience=None)`, que nunca activa `should_stop` — solo decide
"¿mejoró?"). El invariante del repo "evaluar el mejor checkpoint, nunca el último" se cumple sin
ninguna excepción de control de flujo: en cada mejora se guarda `best_round<N>.pt` y se reescribe
`best.json`.

**`best.json`** (`{best_round, metric_name, metric_value, checkpoint, aggregation_scope,
model_hash}`) se reescribe en CADA mejora — al terminar la corrida está garantizado final sin
necesitar un protocolo de señal aparte entre servidor y nodos (ver §8).

**Advertencia de lectura — promedio ponderado ≠ métrica del pool.** `weighted_average()`
(`strategies.py`) promedia las métricas de cada nodo ponderando por `num_examples`. Esto es
correcto y es lo que reporta `best.json`/`metrics.csv` del servidor, pero **el AUC promedio
ponderado de 4 nodos NO es el AUC que se obtendría evaluando el checkpoint global sobre el pool
combinado de las 4 bases de datos** — son operaciones matemáticamente distintas (media de métricas
≠ métrica de la unión de predicciones). Advertencia documentada también en la guía de despliegue del
legacy (`ec55408:docs/FEDERATED_DEPLOYMENT_GUIDE.md`). Para la métrica pooled real, evaluar el
checkpoint global con `src.evaluate` sobre el manifest completo, fuera de este pipeline.

---

## 8. Evaluación final por nodo

Al terminar la última ronda, cada nodo debe producir `val/`+`test/` con la MISMA forma que un
`run_dir` centralizado (requisito de diseño). [`evaluate_node.py`](../src/federated/evaluate_node.py)
implementa esto de dos formas:

- **Primaria, en proceso**: `client.main()` llama a `run_final_evaluation()` justo después de que
  `start_client()` retorna. Como el servidor terminó, `best.json` es final (se reescribió durante
  las rondas, en cada mejora — no hay ventana de carrera). Reusa el ensamblado y el logger que el
  cliente ya tenía abiertos — importa en un despliegue con varios nodos por GPU: evita cargar un
  SEGUNDO ResNet50 mientras el del cliente sigue vivo.
- **Fallback, standalone**: `python -m src.federated.evaluate_node --config <node.yaml>
  --server-run-dir <server_run_dir>` — la recuperación si un nodo murió a mitad de corrida, mismo
  patrón que `src/evaluate.py` para el pipeline centralizado.

Alcance `full`: el checkpoint global se pasa tal cual a `eval_pipeline.evaluate_split()` (mismo
formato que un checkpoint centralizado). Alcance `backbone`: se compone el mejor backbone global +
la cabeza LOCAL que el nodo entrenó en esa MISMA ronda (`weights/round<N>_head.pt`) — la pareja que
realmente se midió cuando esa ronda resultó ser la mejor — y se persiste como `final_round<N>.pt`,
un checkpoint auditable por nodo.

`eval_pipeline.py` no se toca en ningún punto de este diseño.

---

## 9. Salida en disco

```
runs/<experiment_id>/
├── server/            config.yaml · metrics.csv (1 fila/ronda) · server.log · best.json ·
│                      tfevents · plots/ · weights/{best_round<N>.pt[,_backbone,_head], round<N>.pt?}
└── nodes/<node_name>/ == forma de un run_dir centralizado:
                       config.yaml · metrics.csv (1 fila/época local, incl. `round`) · rounds.csv ·
                       train.log · plots/ · weights/{round<r>_head.pt?, final_round<N>.pt?} ·
                       val/ + test/ (metrics.json, confusion_matrix_metrics.json, predictions.csv,
                       confusion_matrix.png, roc_curve.png)
```

Cada proceso escribe SOLO su propia carpeta — a diferencia del legacy, donde todo se escribía desde
el proceso servidor para evitar contención de Ray workers. Con procesos gRPC reales y sistema de
archivos compartido, cada nodo puede ser dueño de sus propios artefactos sin ese problema.

W&B: una corrida por PROCESO (`wandb_group = experiment_id` en servidor y en todos los nodos, para
que las 1+N corridas queden bajo una sola fila expandible en la UI). Las reglas de `.gitignore`
existentes ya cubren esta forma (`*.csv`/`*.json`/`plots/*.png` committeados, `*.pt`/logs/tfevents
ignorados) sin cambios.

---

## 10. Docker

`Dockerfile` se reescribió por completo: la versión anterior apuntaba a Python 3.11 +
`PYTHONPATH=/app/src` + `pip install -e .`, las tres cosas ya falsas (no hay `pyproject.toml` en
este árbol, ver CLAUDE.md). La imagen nueva es **de solo entorno** — instala exactamente
`requirements.txt` (incluido el pin de `flwr`) y no copia código: el repo se monta en `/workspace`
vía bind mount, así que cambiar código no exige rebuild.

[`docker-compose.federated.yaml`](../docker-compose.federated.yaml) orquesta 1 servicio `server` +
1 servicio por nodo (`client-cmmd`, `client-kau-bcmd`, `client-cdd-cesm`, `client-inbreast`).
Decisiones:

- **`network_mode: host` en todos los servicios**, deliberado — igual que el despliegue legacy real
  (`ec55408:scripts/docker-deploy-federated.sh`): así `server_address: "127.0.0.1:8080"` en los YAML
  funciona idéntico dentro y fuera de Docker, sin mapeo de puertos. Esto ata el despliegue a Linux
  (aceptable: el target es la workstation Linux).
- **Healthcheck TCP puro** sobre el puerto del servidor antes de que cualquier cliente arranque
  (`depends_on: condition: service_healthy`) — los clientes nunca corren una carrera contra un
  servidor que todavía está cargando el backbone plantilla.
- **`shm_size: 2gb`** explícito en los servicios de nodo — el crash de memoria compartida del
  DataLoader que el legacy documentó ("unable to allocate shared memory" sin `--shm-size`).
  `num_workers: 0` en los YAML federados ya reduce la necesidad, pero no la elimina.
- **`EXPERIMENT` como variable de entorno** selecciona `configs/federated/${EXPERIMENT}/` — un solo
  `docker-compose.federated.yaml` sirve para cualquier experimento federado sin editarlo.

Verificado en esta sesión (máquina de desarrollo, sin GPU): build de la imagen CPU, y
`docker compose -f docker-compose.federated.yaml config` validando la topología completa de 4 nodos
para `exp40_fedavg_full`. El build/run con GPU real es verificación exclusiva de la workstation
(riesgo §11.6).

---

## 11. Verificación realizada

Cada fase se verificó de forma independiente (espíritu PHASES.md), con scripts desechables sobre
datos sintéticos (no hay mount de datos en la máquina de desarrollo — ver la nota de memoria
correspondiente):

1. **Config + handshake** (Fase 1): las 5 YAML de `exp40_fedavg_full` cargan; una clave mal escrita
   o `by_database_manifests` en un nodo levantan `ValidationError`; el hash de servidor y los 4
   nodos coincide; cambiar `head.hparams.hidden_dim` lo cambia, cambiar `weights_path` no.
2. **`param_utils` + refactors** (Fase 2): round-trip bit-exacto de un ResNet50 real en ambos
   alcances (incluidos los buffers `int64`); `build_param_groups` con y sin `backbone_lr` idéntico
   al código que reemplazó en `cli.py`; `train_one_epoch(regularizer=...)` reporta `prox_loss` sin
   alterar `loss`.
3. **`strategies.py`** (Fase 3): las 4 estrategias stock construyen con la participación/agregación
   cableada; `weighted_average` pondera, filtra NaN y excluye reportes de `num_examples=0`.
4. **`client.py`** (Fase 4): `fit()`/`evaluate()` llamados DIRECTAMENTE (sin gRPC) sobre un
   ResNet50 real + datos sintéticos — ndarrays, `metrics.csv`/`rounds.csv`, hash incorrecto
   levantando `RuntimeError`, y `round1_head.pt` en alcance `backbone`.
5. **`round_tracking.py`** (Fase 5): `TrackedStrategy` alimentada con `FitRes`/`EvaluateRes`
   fabricados a mano (3 rondas, mejora en la 1 y la 3) — `best_round == 3`, exactamente dos
   `best_round*.pt`, ambos cargan de vuelta en el modelo plantilla.
6. **`evaluate_node.py`** (Fase 6): evaluación final contra un `best.json`/checkpoint reales
   (ResNet50) — artefactos `val/`+`test/` completos, `metrics.csv` de entrenamiento intacto
   byte-a-byte, hash incorrecto y `best.json` ausente levantando error con mensaje accionable.
7. **Extremo a extremo, procesos reales** (Fase 7): 1 servidor + 2 clientes REALES, gRPC real en
   `localhost`, ResNet50 (`resnet50_imagenet_v1`, ya cacheado — sin descarga) contra datos
   sintéticos (TIFFs float sintéticos + manifests patient-disjoint). Dos corridas completas:
   `aggregation_scope: full` + FedAvg (2 nodos, 1 ronda) y `aggregation_scope: backbone` + FedProx
   μ=0.1 (2 nodos, 2 rondas) — ambas con los 3 procesos terminando en código 0 y el árbol de
   artefactos completo de §9, incluida la columna `train_prox_loss` en la corrida FedProx y
   `round<r>_head.pt`/`final_round<N>.pt` en la corrida `backbone`.
8. **Docker** (Fase 8): build de la imagen CPU + `docker compose config` validando la topología
   real de 4 nodos.

**Nota sobre el entorno de esta sesión**: la máquina donde se hizo esta verificación mostró
inestabilidad de reloj de pared severa e intermitente (un `sleep 200` tomó ~550s en una medición;
varios lanzamientos de los mismos procesos gRPC fallaron sin razón aparente y luego funcionaron sin
cambios). Esto es ruido de la infraestructura de desarrollo, no del código — cada fallo se
reprodujo, se investigó con un caso mínimo, y se descartó como defecto de `src/federated/` antes de
reintentar. La primera corrida real en la workstation (con datos reales, RadImageNet, y 4 nodos
sobre GPU) sigue siendo la verificación que importa para resultados reales — ver riesgos abajo.

---

## 12. Riesgos abiertos (verificar en la workstation)

1. Kwargs exactos del constructor de las 4 estrategias stock quedaron fijados contra `flwr==1.31.0`
   — reconfirmar si se sube la versión.
2. 4 procesos CUDA contra una sola GPU: sin verificación de VRAM posible desde esta máquina —
   ajustar `batch_size` por nodo en los YAML si aparece OOM.
3. Ruta RadImageNet (`resnet50_radimagenet`, requiere `weights_path`) nunca se ejercitó en esta
   sesión — solo `resnet50_imagenet_v1` (cacheado localmente). La primera corrida workstation con
   RadImageNet es la verificación real de esa arquitectura en el pipeline federado.
4. Imagen CUDA + `nvidia-container-toolkit` en la workstation: el despliegue legacy corrió
   contenedores GPU ahí, así que el toolkit presumiblemente existe — confirmar antes del primer
   `docker compose up` con GPU real.
