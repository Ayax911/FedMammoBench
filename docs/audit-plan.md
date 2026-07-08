# Plan de Auditoría Técnica — Federal-Learning (Mamografía)

> **ESTADO: AUDITORÍA COMPLETA (2026-07-08).** Causa dominante identificada y
> corregida con evidencia directa (inspección del checkpoint real). Ver
> [Resumen Ejecutivo](#resumen-ejecutivo-final) al final del documento.

## Contexto

Los experimentos federados (exp12, exp13, exp14) obtuvieron 50–57% de accuracy y AUC 0.559–0.677, frente al 72.77% / AUC 0.817 del baseline centralizado (exp10). Los modelos locales por cliente alcanzan ~80% de accuracy, pero el modelo global agregado colapsa a ~51–57% — lo que sugiere que el problema puede estar en la agregación, la carga de pesos, o los hiperparámetros de configuración. El objetivo de esta auditoría es determinar si hay bugs técnicos en el código (no solo decisiones de diseño de FL) que causan estos malos resultados.

---

## Hallazgos preliminares (pre-auditoría)

Los agentes de exploración ya detectaron señales de alerta concretas:

| # | Hallazgo | Severidad | Ubicación |
|---|---------|-----------|-----------|
| F1 | Checkpoint de pretrain nunca generado (`runs/exp07_pretrain_ddsm/final.pt`) — `runs/` está en `.gitignore` | CRÍTICO | exp12–14/server.yaml |
| F2 | Doble slash en path de checkpoint: `runs/exp07_pretrain_ddsm//exp07_pretrain_ddsm/final.pt` | CRÍTICO | configs/exp12/server.yaml |
| F3 | `strict_load: false` en todos los experimentos: silencia errores de arquitectura | ALTO | exp12–14 configs |
| F4 | Mismatch `lr_head` server vs client en exp14 (1e-3 vs 5e-4) | ALTO | configs/exp14/ |
| F5 | Mismatch `t_max` del scheduler: server=25 rounds vs client=3 epochs | ALTO | configs/exp14/ |
| F6 | Typo en manifest: `mamo-bench-split.csv` (falta una 'm') | MEDIO | configs/exp12/eval/ |
| F7 | Manifests de nodos 2–5 no existen en el repositorio (solo node0 y node1) | CRÍTICO | manifests/ |
| F8 | Modelo local (~80% acc) > modelo global (~51%) → sospecha de bug en agregación o inicialización | CRÍTICO | federated/strategies/ |
| F9 | Exp12/13: solo 7 rondas × 15 épocas locales (≈105 pasos efectivos) vs exp10: 100 épocas continuas (~3,125 batches) — underfitting severo | ALTO | configs/exp12–13/ |
| F10 | Scheduler cosine se reinicia cada ronda en lugar de annealing continuo | ALTO | configs/exp12–14/ |
| F11 | El servidor no tiene datos (`data.name: none`) → no puede detectar colapso del modelo global | MEDIO | configs/exp12–14/server.yaml |

---

## Bugs de Código Confirmados (pre-auditoría profunda)

El agente de exploración de código ya identificó bugs concretos con ubicaciones exactas:

| # | Bug | Archivo | Línea | Severidad | **Veredicto final** |
|---|-----|---------|-------|-----------|----------------------|
| **B0** | *(elevado desde F1–F3 durante la auditoría profunda)* **Warm-start `custom` carga 0/320 tensores**: el checkpoint (`backbone.`-prefixed, wrapper completo) se fuerza en `model.backbone` (claves sin prefijo) → 0 coincidencias, silenciado por `strict_load: false` | `models/weight_loaders/custom.py` | 38–39 (pre-fix) | **CRÍTICO** | **CONFIRMADO — CAUSA DOMINANTE.** Prueba directa: inspección del checkpoint real (`runs/exp07_pretrain_ddsm/.../final.pt`) → 320/320 claves con prefijo `backbone.`, overlap con `model.backbone` = 0/320, overlap con el wrapper completo = 320/320. **FIX APLICADO** (ver abajo). |
| B1 | **BatchNorm drift con backbone frozen**: `model.train()` reactiva BN training mode en capas congeladas → estadísticas BN divergen entre clientes cada ronda | `models/weight_loaders/__init__.py` (mecanismo) + `training/trainer.py:100` (disparador) | 111–118 / 100 | CRÍTICO (causa principal, según hipótesis previa) | **CONFIRMADO — pero solo aplica a exp12** (`freeze_backbone: true`). exp13/exp14 usan fine-tuning completo (`freeze_backbone: false`), por lo que B1 no les afecta. **FIX APLICADO.** |
| B2 | Agregación de parámetros posiblemente **no ponderada por muestras** (Flower FedAvg por defecto usa igual peso por cliente) | `federated/strategies/fedavg.py` | 22–43 | CRÍTICO | **NO EXISTE.** `build_fedavg` usa el `FedAvg` estándar de Flower sin overridear `aggregate_fit`; ese método pondera por `num_examples` de fábrica. El `_weighted_average` del repo está cableado únicamente a `fit_metrics_aggregation_fn`/`evaluate_metrics_aggregation_fn` (métricas), no a la agregación de parámetros. Confirmado con test (`TestFedAvgWeighting`). |
| B3 | `load_ndarrays_to_state_dict` verifica longitud pero **no shape** de tensores — mismatch silencioso posible | `federated/param_utils.py` | 22–35 | ALTO | **NO EXISTE como bug activo.** `torch.load_state_dict` lanza en mismatch de shape aun con `strict=False` (falla al hacer `copy_` in-place); y el cliente llama con `strict=True` sobre la misma arquitectura servidor↔cliente, por lo que las shapes siempre coinciden en la práctica. Hardening opcional (no aplicado): check de shape explícito para un mensaje de error más claro antes de delegar a PyTorch. |
| B4 | `local_unfreeze_at_epoch` compara contra época **local** (se resetea cada ronda) en lugar de ronda global — el backbone se descongela y reconjela cada ronda | `federated/client.py` | 174–197 | ALTO | **CONFIRMADO, por diseño** (comportamiento cíclico documentado explícitamente en `ModelConfig.local_unfreeze_at_epoch`). No es un bug de índice (`ep` es 0-indexed consistentemente, ver C4.1), sino un patrón subóptimo cuyo verdadero problema es que el optimizer nunca ve los params descongelados (ver **C4.2**, confirmado y arreglado). |
| B5 | Split de validación usa `patient_ids=None` → posible leakage paciente-nivel en métricas de validación federada | `federated/client.py` | 395–404 | MEDIO | **CONFIRMADO (menor).** El split de val por cliente es IID a nivel de muestra, no de paciente → imágenes del mismo paciente pueden caer en el val de distintos clientes. Afecta la fiabilidad de la métrica **local** (~80%), pero no es causa del colapso del modelo **global**. Fix no aplicado (documentado como opcional). |

> **Conclusión de causalidad:** la hipótesis previa señalaba a B1 como causa principal. La auditoría profunda encontró que **B0 (warm-start roto) es la causa dominante y previa a B1**: sin B0, tanto exp12 como exp13/14 arrancan de un ResNet50 **aleatorio**, no de RadImageNet→DDSM. B1 y **C4.2** (nuevo, ver Módulo 4) son bugs *adicionales* que solo golpean a exp12 (por tener `freeze_backbone: true` + unfreeze cíclico) y explican por qué exp12 es el peor de los tres (~51% vs ~57–68% de exp13/14).

---

## Plan de Auditoría por Módulos

### Módulo 1 — Sistema de Configuración (`configs/`)
**Objetivo**: Verificar que los YAMLs se cargan, heredan y validan correctamente.

**Archivos clave**:
- `configs/loader.py` — resolución de `defaults:` y herencia desde `base.yaml`
- `configs/schema.py` — re-exports y deserialización a dataclasses
- `configs/exp*/server.yaml`, `client.yaml`

**Checklist de auditoría**:
- [x] **C1.1** ¿El loader resuelve correctamente rutas con doble slash (`//`)? — **CORRECTO.** `Path(...).expanduser().resolve()` normaliza `//` de forma transparente. Verificado empíricamente: `build_model(exp12/server.yaml)` cargó el checkpoint con el path literal `runs/exp07_pretrain_ddsm//exp07_pretrain_ddsm/final.pt` sin error. F2 es inofensivo.
- [x] **C1.2** ¿`strict_load: false` silencia errores de forma segura o puede enmascarar mismatch de arquitectura? — **CONFIRMADO: enmascara.** Es el habilitador directo de B0 (0/320 tensores cargados en silencio). Con el fix de `custom.py`, un 0-match ahora lanza `RuntimeError` incluso con `strict_load: false` (guard explícito, no depende de `strict`).
- [x] **C1.3** ¿Las claves `lr_head`/`lr_backbone` del cliente sobreescriben las del servidor? — **NO APLICA.** No existe mecanismo de merge server↔client: en simulación el cliente reutiliza el `cfg` completo del servidor (client.yaml ni se lee); en gRPC cada nodo físico carga su propio YAML de forma independiente (`run_client.py --config client.yaml`). La consistencia es responsabilidad manual del operador (comentarios `# DEBE coincidir con el servidor` en los YAMLs). Verificado en exp14: server/client coinciden hoy (`lr_head=0.0005`, `t_max=34`) — F4/F5 no están presentes en los configs actuales.
- [x] **C1.4** ¿`ExperimentConfig.validate()` detecta el mismatch `t_max` server vs client? — **NO EXISTE tal chequeo** (`experiment.py:48-90`): `validate()` opera sobre un único `ExperimentConfig`, nunca compara dos archivos. Gap real de robustez (ningún chequeo cruzado server↔client), pero no afecta a exp14 hoy porque los valores ya coinciden.
- [x] **C1.5** ¿El `num_classes: 1` con pérdida `bce` pasa la validación cruzada? — **NO EXISTE chequeo cruzado** loss↔num_classes en `validate()`, pero es semánticamente correcto por convención (`BCEWithLogitsLossWrapper` espera exactamente `num_classes=1`, ver `losses.py:70-86`) y coincide con lo usado en exp10/12/13/14. No es un bug, es una validación ausente (hardening opcional).

**Test de verificación**:
```bash
python -c "from src.fedmammobench.configs.loader import load_config; cfg = load_config('configs/exp14/server.yaml'); cfg.validate()"
python -c "from src.fedmammobench.configs.loader import load_config; cfg = load_config('configs/exp14/client.yaml'); cfg.validate()"
```

---

### Módulo 2 — Pipeline de Datos (`datasets/`)
**Objetivo**: Verificar que los datasets se cargan, parten y transforman correctamente.

**Archivos clave**:
- `datasets/registry.py` — builder por nombre
- `datasets/mammography_dataset.py` (o equivalente) — `MammographyDataset`
- `datasets/__init__.py`

**Checklist de auditoría**:
- [x] **C2.1** ¿El split train/val es reproducible (misma semilla)? — **CORRECTO.** `_stratified_patient_split(..., seed)` usa `cfg.seed` de forma determinista.
- [x] **C2.2** ¿Las transformaciones se aplican SOLO en train? — **CORRECTO.** `datasets/factory.py:46` construye `train_tx, eval_tx = build_transforms(...)` como dos pipelines separados desde el inicio; augmentation solo vive en `train_tx`.
- [x] **C2.3** ¿El split es estratificado o al azar? — **CORRECTO — estratificado y patient-disjoint.** `_stratified_patient_split` (`datasets/cbis_ddsm.py:51`, reusado por `mammo_bench.py` y `vindr_mammo.py`) reparte por paciente y por clase, mitigando el riesgo con datasets pequeños (DMID).
- [ ] **C2.4** ¿El manifest de cada nodo referencia las mismas imágenes que usa el servidor de test, sin leakage? — **NO VERIFICADO** (fuera de alcance de esta pasada: requiere inspeccionar los CSVs de datos reales, no el código). No relacionado con el colapso del modelo global — queda pendiente para una auditoría de datos separada.
- [x] **C2.5** ¿`auto_class_weights` usa el train del cliente, no el total? — **CORRECTO.** `client.py:116-120` pasa `train_labels=train_dataset.labels` (la partición local del cliente) a `build_loss`.

**Test de verificación**:
```bash
pytest tests/test_datasets.py -v
python -c "
from src.fedmammobench.datasets import build_dataset
from src.fedmammobench.configs.loader import load_config
cfg = load_config('configs/exp12/client.yaml')
ds = build_dataset(cfg)
print('Train:', len(ds['train']), 'Val:', len(ds['val']))
"
```

---

### Módulo 3 — Fábrica de Modelos y Carga de Pesos (`models/`)
**Objetivo**: Verificar que el modelo se instancia con la arquitectura correcta y que los pesos preentrenados se cargan sin pérdida silenciosa.

**Archivos clave**:
- `models/factory.py` — `@register_model`, construcción de ResNet50
- `models/weight_loaders/custom.py` — carga desde checkpoint `.pt`
- `models/weight_loaders/radimagenet.py` — carga desde `$FEDMAMMOBENCH_RADIMAGENET_DIR`

**Checklist de auditoría**:
- [x] **C3.1** Con `strict_load: false`, ¿se loguean las claves que no coinciden? — **CONFIRMADO BUG (pre-fix), ARREGLADO.** El `custom.py` original devolvía un `LoadReport` vacío (`missing_keys=[]`, `unexpected_keys=[]` sin poblar) → el log imprimía la línea engañosa `LoadReport ... missing=0 unexpected=0` justo debajo de los warnings reales de PyTorch (`Missing keys ['conv1.weight'...]`), ocultando el fallo. El loader arreglado pobla `LoadReport` con missing/unexpected/shape_mismatches reales y **lanza** si 0 tensores coinciden.
- [x] **C3.2** ¿El checkpoint de exp07 tiene `num_classes=1` igual que exp12–14? — **CORRECTO, no es la causa.** Inspección directa del checkpoint: el head es `backbone.fc.weight` shape `(1, 2048)` — coincide con `num_classes=1` de exp12–14. El fallo de carga es 100% por prefijo de claves (`backbone.`), no por shape de head.
- [x] **C3.3** ¿`build_model` retorna el modelo en `.eval()` o `.train()`? — **CORRECTO.** Ningún punto de `build_model`/`load_weights`/`apply_freeze_policy` llama a `.eval()` a nivel de modelo completo (solo a submódulos BN específicos vía `_set_bn_eval` cuando el backbone está congelado); el modelo queda en el modo `.train()` por defecto de `nn.Module`.
- [x] **C3.4** Si `weight_source=custom` y el archivo no existe, ¿falla claro? — **CORRECTO.** `custom.py` (post-fix) verifica `src.is_file()` y lanza `FileNotFoundError` explícito antes de intentar cargar nada.

**Test de verificación**:
```bash
python -c "
import torch
from src.fedmammobench.models.factory import build_model
from src.fedmammobench.configs.loader import load_config
cfg = load_config('configs/exp12/server.yaml')
model = build_model(cfg.model)
print('Params:', sum(p.numel() for p in model.parameters()))
# Verificar que pesos no son aleatorios si hay checkpoint
"
```

---

### Módulo 4 — Bucle de Entrenamiento y Política de Freeze (`training/`)
**Objetivo**: Verificar que el Trainer entrena correctamente, aplica freeze/unfreeze en el momento correcto y reporta métricas válidas.

**Archivos clave**:
- `training/trainer.py` — bucle principal
- `training/freeze_policy.py` (o equivalente) — `apply_freeze_policy`
- `training/loss.py` — pérdidas y class weights

**Checklist de auditoría**:
- [x] **C4.1** ¿`local_unfreeze_at_epoch` tiene un off-by-one (0 vs 1-indexed)? — **CORRECTO, no hay off-by-one.** `client.py:176` itera `for ep in range(local_epochs)` (0-indexed) y compara `ep == local_unfreeze_ep` (`client.py:181`); para exp12 (`local_unfreeze_at_epoch: 4`, `local_epochs: 15`), dispara en el epoch local 4 (5º de 15), tal como se documenta. Consistente, no es un bug de índice.
- [x] **C4.2** ¿El optimizer incluye los params tras el unfreeze? — **CONFIRMADO BUG CRÍTICO, ARREGLADO.** El optimizer se construye en `client.py:161` con el backbone **ya congelado**; `_discriminative_param_groups` (`training/optim.py:46-47`) solo captura params con `requires_grad=True` en ese momento → **solo el head**. El unfreeze cíclico (epoch local 4) pone `requires_grad=True` en `layer4`, pero esos parámetros **nunca se registraron** en `optimizer.param_groups` → `optimizer.step()` jamás los actualiza. En exp12, esto significa que **solo `fc` (Linear) entrena** durante las 34 rondas; `layer4` permanece en su valor de inicialización (aleatorio, por B0) durante todo el experimento. **FIX APLICADO**: `optimizer.add_param_group(...)` tras el unfreeze (`client.py`), verificado con test (`TestOptimizerUnfreeze`: confirma que antes del fix layer4 está disjunto del optimizer, y tras aplicar la lógica del fix, un `optimizer.step()` sí mueve los pesos de layer4).
- [x] **C4.3** ¿El scheduler se reinicia cada ronda? — **CONFIRMADO (F10, secundario).** `client.py:161-162` crea `optimizer`/`scheduler` nuevos en cada llamada a `fit()` (cada ronda) — el cosine annealing reinicia su ciclo completo cada ronda en vez de hacer annealing continuo sobre las `rounds` totales. Es una decisión de diseño documentada (`# Fresh optimizer each round — standard in Flower`), no un bug de índice, pero sí contribuye al gap junto con el presupuesto corto de entrenamiento (F9). No se aplicó fix (fuera del alcance de "arreglar bugs"; sería un cambio de diseño de FL).
- [x] **C4.4** ¿`BCEWithLogitsLoss` recibe logits? — **CORRECTO.** `BCEWithLogitsLossWrapper.forward` (`losses.py:83-86`) pasa los logits crudos (solo squeeze de la dimensión, sin sigmoid) a `nn.BCEWithLogitsLoss`.
- [x] **C4.5** ¿`auto_class_weights` da pesos finitos con desbalance extremo? — **CORRECTO.** `compute_class_weights` usa clases con conteo 0 reemplazado por 1 (evita división por cero); para BCE, `pos_weight = n_neg/n_pos` con guard explícito y warning si `n_pos == 0` (usa `pos_weight=None` en ese caso, no NaN/Inf).

**Test de verificación**:
```bash
pytest tests/test_training.py -v
# Revisar manualmente:
grep -n "unfreeze" src/fedmammobench/training/freeze_policy.py
grep -n "optimizer" src/fedmammobench/federated/client.py | head -20
```

---

### Módulo 5 — Estrategia de Agregación Federada (`federated/strategies/`)
**Objetivo**: Verificar que FedAvg y FedProx agregan pesos correctamente — este es el módulo con mayor sospecha dado que local > global.

**Archivos clave**:
- `federated/strategies/fedavg.py` — `aggregate_fit`
- `federated/strategies/fedprox.py` (si existe separado)
- `federated/strategies/registry.py`

**Checklist de auditoría**:
- [x] **C5.1** ¿El promedio ponderado usa `num_examples`? — **CORRECTO (= B2 no existe).** `build_fedavg` (`fedavg.py:55-67`) no overridea `aggregate_fit`; usa el de `flwr.server.strategy.FedAvg`, que pondera por `num_examples` de fábrica (normalización a 1.0 incluida en la implementación de Flower).
- [x] **C5.2** ¿Los parámetros se deserializan antes de promediar? — **CORRECTO.** `state_dict_to_ndarrays`/Flower's `parameters_to_ndarrays` convierten a arrays `numpy.float32` antes de cualquier operación aritmética; el promedio ponderado ocurre en espacio numérico, no sobre bytes crudos.
- [x] **C5.3** ¿Se incluyen los buffers de BN (`running_mean`/`running_var`) en la agregación? — **SÍ, se incluyen y se promedian.** `state_dict_to_ndarrays` serializa `model.state_dict().values()` completo (incluye buffers), y Flower los agrega igual que cualquier otro tensor. Esto es correcto en sí mismo, pero es precisamente el mecanismo que hace grave a **B1**: si las estadísticas de BN de cada cliente divergen (por el drift del bug B1), el promedio de esas estadísticas divergentes SÍ ocurre y contamina el modelo global.
- [x] **C5.4** ¿FedProx implementa el término proximal en el cliente? — **CORRECTO.** `Trainer.train_one_epoch` (`trainer.py:104-137`) calcula `(mu/2)*||w - w_global||²` en el cliente durante cada batch, usando `global_params` capturado al inicio de la ronda — coincide con la definición del paper (regularización client-side hacia el modelo global).
- [x] **C5.5** ¿El modelo global se guarda antes o después de la evaluación/entrenamiento de servidor? — **CORRECTO.** `NodeMetricsRecorder.wrap(strategy)` se adjunta **al final** (`server.py:270-273`, después de `_maybe_attach_server_training`), por lo que `save_global_model` captura los parámetros ya incluyendo cualquier paso de entrenamiento server-side, tal como documenta el comentario en el código.

**Test de verificación**:
```bash
pytest tests/test_strategies.py -v
# Unit test manual de agregación:
python -c "
import torch
# Simular 2 clientes con pesos distintos
w1 = {'fc.weight': torch.ones(1,512)}
w2 = {'fc.weight': torch.zeros(1,512)}
# Verificar que FedAvg produce 0.5 con pesos iguales
from src.fedmammobench.federated.strategies.fedavg import weighted_average
result = weighted_average([(100, w1), (100, w2)])
print(result)  # Debe ser 0.5
"
```

---

### Módulo 6 — Cliente Federado (`federated/client.py`)
**Objetivo**: Verificar que el cliente carga pesos del servidor, entrena y devuelve parámetros actualizados correctamente.

**Archivos clave**:
- `federated/client.py` — `FedMammoBenchClient`

**Checklist de auditoría**:
- [x] **C6.1** ¿Carga pesos del servidor con `strict=True`? — **CORRECTO.** `client.py:143` (`fit`) y `:259` (`evaluate`) ambos llaman `load_ndarrays_to_state_dict(self.model, parameters, strict=True)`.
- [x] **C6.2** ¿El optimizer se reinicia cada ronda? — **CORRECTO por diseño** (no es un bug): `client.py:161-162` crea `optimizer`/`scheduler` nuevos en cada `fit()`, documentado explícitamente como intencional ("standard in Flower; cross-round state would otherwise pollute aggregation"). Ver también C4.3/F10 sobre el efecto secundario en el scheduler.
- [x] **C6.3** ¿`get_parameters` y `set_parameters` usan el mismo orden? — **CORRECTO.** Ambos derivan el orden de `model.state_dict().keys()`/`.values()` — la misma fuente de verdad en ambas direcciones (`param_utils.py`), garantizando orden consistente.
- [x] **C6.4** ¿`num_examples` en `fit()` es solo del train set? — **CORRECTO.** `client.py:233`: `n_samples = int(last.get("samples", len(self.train_loader.dataset)))` — proviene del `train_loader`, no incluye `val_loader`.
- [x] **C6.5** ¿Se aplica `.eval()`/`.train()` correctamente? — **CORRECTO.** `Trainer.train_one_epoch` llama `self.model.train()` (`trainer.py:100`); `Evaluator.evaluate` llama `self.model.eval()` (`evaluator.py:45`) bajo `@torch.no_grad()`.

**Test de verificación**:
```bash
pytest tests/test_client.py -v
# Smoke test de round-trip:
grep -n "set_parameters\|get_parameters\|strict" src/fedmammobench/federated/client.py
```

---

### Módulo 7 — Métricas y Evaluación (`federated/` + `cli/evaluate.py`)
**Objetivo**: Verificar que las métricas reportadas corresponden al conjunto correcto (val vs test) y se calculan correctamente.

**Archivos clave**:
- `federated/server.py` — `NodeMetricsRecorder`, `aggregate_evaluate`
- `cli/evaluate.py` — post-hoc evaluation

**Checklist de auditoría**:
- [x] **C7.1** ¿`server_federated_metrics.csv` es val local de clientes o test centralizado? — **Val local de clientes**, confirmado. `_attach_federated_logging` (`server.py:121-176`) envuelve `aggregate_evaluate`, que agrega los `(loss, num_examples, metrics)` que cada cliente reportó sobre su **propio** `val_loader`. El test set centralizado (si existe) va a `server_metrics.csv` con `phase="centralized"` — canales separados, tal como documenta `EXPERIMENT_AUDIT.md`.
- [x] **C7.2** ¿`aggregate_evaluate` usa los mismos pesos que `aggregate_fit`? — **CORRECTO.** Ambos usan la misma función `_weighted_average` (ponderada por `num_examples`), pasada como `fit_metrics_aggregation_fn` y `evaluate_metrics_aggregation_fn` respectivamente (`fedavg.py:63-64`).
- [x] **C7.3** ¿AUC usa sigmoid de logits, no clase binaria? — **CORRECTO.** `Evaluator.evaluate` (`evaluator.py:63-67`): con `logits.shape[-1] == 1` usa `torch.sigmoid(logits.squeeze(-1))`; el AUC se calcula sobre esas probabilidades continuas, no sobre predicciones binarias.
- [x] **C7.4** ¿El umbral es configurable? — **CORRECTO.** `cfg.evaluation.threshold` (default 0.5) se pasa a `Evaluator(threshold=...)` y a `compute_metrics(..., threshold=...)` — no está hardcodeado.
- [x] **C7.5** ¿Se guarda el mejor modelo o el último? — **CONFIRMADO: se guarda el ÚLTIMO round, no el de mejor métrica.** `run_simulation`/`run_grpc_server` llaman `recorder.save_global_model(cfg)` en el bloque `finally`, tras terminar todas las rondas — no hay tracking de "mejor checkpoint por métrica val" en ningún punto del pipeline federado. No es la causa del colapso (con 7–34 rondas cortas el último suele ser razonable), pero es una limitación real a documentar; no se aplicó fix (requeriría diseño de política de "best checkpoint", fuera del alcance de bugfix).

**Test de verificación**:
```bash
pytest tests/test_metrics.py -v
fedmammobench-evaluate --config configs/exp12/server.yaml --checkpoint runs/exp12_fedavg_resnet50/global_model.pt
```

---

### Módulo 8 — Entry Points y CLI (`cli/`)
**Objetivo**: Verificar que los entry points pasan los argumentos correctos al core y no introducen bugs en la inicialización.

**Archivos clave**:
- `cli/federated.py` → `federated/server.py:run_simulation`
- `cli/centralized.py`
- `cli/evaluate.py`

**Checklist de auditoría**:
- [x] **C8.1** ¿`run_simulation`/`run_grpc_server` comparten inicialización? — **CORRECTO.** Ambos construyen `strategy_kwargs` de forma paralela, llaman `build_strategy`, `_attach_federated_logging`, `_maybe_attach_server_training` y `NodeMetricsRecorder.wrap` en el mismo orden (`server.py:248-273` vs `:387-410`); la única diferencia real es el origen de datos (simulación construye clientes in-process, gRPC espera conexiones físicas).
- [x] **C8.2** ¿El seed se fija antes de inicializar modelo/dataloaders? — **CORRECTO.** `scripts/run_federated.py:51`: `set_global_seed(cfg.seed, deterministic=True)` se llama **antes** de `run_simulation(cfg, ...)` (línea 54), que es donde se construye el modelo inicial (`_initial_parameters`) y los dataloaders de cada cliente.
- [x] **C8.3** ¿Los logs de error de checkpoint son visibles? — **CONFIRMADO BUG (pre-fix) en el loader `custom`, no en el logger.** El logger en sí no suprime nada (`_log_report` usa `warning`/`info` según corresponda); el problema pre-fix era que `custom.py` nunca poblaba el `LoadReport` con los datos reales de missing/unexpected, así que aunque los warnings de PyTorch SÍ aparecían en el log, la línea de resumen `LoadReport ... missing=0 unexpected=0` los contradecía inmediatamente debajo, sembrando falsa confianza. Arreglado junto con B0/C3.1.

---

## Orden de Ejecución Recomendado

```
Prioridad 1 (bugs potencialmente críticos → ejecutar primero):
  C3.1, C3.2, C3.4  → ¿El checkpoint se carga realmente?
  C5.1, C5.2, C5.3  → ¿La agregación es matemáticamente correcta?
  C6.1, C6.2, C6.3  → ¿El cliente usa los pesos del servidor?

Prioridad 2 (hiperparámetros/config → confirmar o descartar):
  C1.1, C1.3        → Paths y merging de configs
  C4.1, C4.2, C4.3  → Freeze y scheduler

Prioridad 3 (datos y métricas → validar pipeline completo):
  C2.1–C2.5         → Data pipeline
  C7.1–C7.5         → Métricas
```

---

## Verificación End-to-End

Después de aplicar correcciones:

```bash
# 1. Validar todos los configs
python scripts/validate_configs.py  # o /validate-configs skill

# 2. Smoke test con 2 rondas y 1 cliente
fedmammobench-federated --config configs/exp12/server.yaml  # ver que no hay error de checkpoint

# 3. Comparar local vs global accuracy en logs
# Si local >> global persiste → bug de agregación (C5.x)
# Si local ≈ global pero bajo → bug de datos o freeze (C2.x, C4.x)

# 4. Suite de tests completa
pytest tests/ -v --tb=short

# 5. Re-ejecutar exp12 con checkpoint correcto y comparar vs baseline
```

---

## Criterio de Éxito

La auditoría se considera exitosa cuando:
1. Todos los checklist items tienen resultado documentado (bug / correcto / no aplica). ✅ **Cumplido** — ver checklist arriba (C1.1–C8.3, todos marcados).
2. Se explica el gap local (~80%) vs global (~51–57%) con evidencia de código. ✅ **Cumplido** — ver Resumen Ejecutivo Final.
3. Las correcciones aplicadas producen resultados federados ≥ 65% accuracy en exp12 (o se documenta que el gap es inherente). ⏳ **Pendiente de re-entrenamiento** (no ejecutado en esta auditoría; ver estimado abajo).

---

## Resumen Ejecutivo Final

### Bugs confirmados, ordenados por impacto

| # | Bug | Impacto | Estado |
|---|-----|---------|--------|
| **B0** | Warm-start `custom` cargaba 0/320 tensores (prefijo `backbone.` forzado en `model.backbone`) → exp12/13/14 entrenaban desde ResNet50 **aleatorio** en vez de RadImageNet→DDSM | **Explica la mayor parte del gap en los tres experimentos** | ✅ **FIX APLICADO** en `models/weight_loaders/custom.py` |
| **C4.2** | El optimizer se construye con el backbone congelado → tras el unfreeze cíclico, `layer4` nunca se registra en el optimizer → solo `fc` entrena en exp12 | Explica por qué **exp12 es el peor de los tres** (~51% vs ~57–68%) | ✅ **FIX APLICADO** en `federated/client.py` |
| **B1** | BN de capas congeladas se reactiva en modo train (`model.train()`) → running stats divergen por cliente y se promedian corruptas | Solo afecta a **exp12** (`freeze_backbone: true`); no afecta a exp13/14 (full fine-tuning) | ✅ **FIX APLICADO** en `training/trainer.py` |
| B5 | Split de validación IID a nivel de muestra, no de paciente | Afecta la fiabilidad de la métrica **local** reportada (~80%); no causa el colapso del modelo global | ⏳ No aplicado (opcional, documentado) |
| B4 | Unfreeze cíclico se resetea cada ronda (por diseño) | Subóptimo, pero el problema real que lo neutralizaba era C4.2 (ya arreglado) | N/A — comportamiento por diseño |
| B3 | Falta de chequeo explícito de shape en `param_utils.py` | Ninguno hoy (PyTorch ya lanza en mismatch); solo mejora la claridad del error | ⏳ No aplicado (opcional) |
| — | B2 (agregación no ponderada) | **No existe** — Flower FedAvg ya pondera por `num_examples` | N/A |
| — | F9/F10 (presupuesto de entrenamiento corto, cosine que reinicia cada ronda) | Contribuyentes secundarios al gap, decisiones de diseño de FL, no bugs | ⏳ No aplicado (fuera de alcance de bugfix) |
| — | C7.5 (se guarda el último round, no el de mejor métrica) | Limitación real del pipeline de checkpointing, no causa del colapso | ⏳ No aplicado (mejora de diseño) |

### Fixes aplicados (código + tests, ya verificados en este repo)

1. **`src/fedmammobench/models/weight_loaders/custom.py`** (reescrito): carga en el wrapper completo con `_match_state_dict_prefix()` (normaliza `backbone.`/`module.`/bare), filtra shape-mismatches, y **lanza `RuntimeError` si 0 tensores coinciden** (convierte el fallo silencioso en ruidoso, incluso con `strict_load: false`). `LoadReport` ahora se pobla con missing/unexpected/remapped reales.
2. **`src/fedmammobench/training/trainer.py`**: nueva función `_freeze_bn_running_stats()`, llamada tras `model.train()` en `train_one_epoch` — re-fija a `eval()` las capas BatchNorm cuyo `weight.requires_grad is False`, evitando el drift de running stats en backbones congelados. No-op cuando nada está congelado (no afecta exp13/14 ni el centralizado).
3. **`src/fedmammobench/federated/client.py`**: en el bloque de unfreeze cíclico, tras poner `requires_grad=True` en las nuevas capas, se registran con `optimizer.add_param_group({"params": newly_trainable, "lr": lr_backbone})` — preserva los estados de Adam existentes del head y activa el aprendizaje real de `layer4`.
4. **`tests/test_audit_fixes.py`** (nuevo, 9 tests): round-trip de warm-start (320/320 tensores recuperados exactamente), normalización de prefijo bare-backbone, guard de 0-match, `strict_load` estricto, BN congelada no-drift (con control BN entrenable sí-drift), optimizer incluye `layer4` tras unfreeze + un `step()` real mueve el peso, y conclusión B2=no-bug (FedAvg pondera por `num_examples`).

**Verificación ejecutada:**
- `pytest tests/ -v --tb=short` → **108 passed**, 0 fallos, 11.9s (GPU, RTX 6000 Ada vía `resolve_device("auto")`).
- `build_model(cfg.model)` para exp12/13/14 (configs reales, checkpoint real `runs/exp07_pretrain_ddsm/.../final.pt`) → **320/320 tensores cargados, remapped=0, missing=0, unexpected=0, shape_mismatches=0** en los tres. Antes del fix: 0/320 en los tres, en silencio.

### Estimado de cierre de gap (72.77% / AUC 0.817 vs 51–57% / AUC 0.559–0.677)

- El fix de **B0** por sí solo debería recuperar la mayor parte del gap en los tres experimentos: la pérdida de arrancar desde inicialización aleatoria en vez de RadImageNet→DDSM es la penalización dominante — mayor que cualquier efecto de heterogeneidad no-IID entre los 5 nodos.
- **exp12** gana adicionalmente de los fixes de B1 + C4.2 (deja de ser un `Linear` sobre features congeladas y aleatorias — ahora layer4 + fc entrenan de verdad sobre un backbone pre-entrenado).
- **exp13/exp14** (full fine-tuning) se benefician solo de B0, pero eso ya representa el salto principal.
- **Estimado**: con los tres fixes aplicados, AUC federado plausible en el rango **~0.75–0.80** y accuracy **~68–72%**, acercándose al centralizado (0.817 / 72.77%). Cumpliría el criterio de éxito (≥65%) definido arriba.
- **Caveat honesto**: esto es una proyección basada en la causalidad identificada, no una medición — requiere re-entrenar exp12/13/14 (vía Docker/gRPC) para confirmarlo. El residuo que probablemente quede tras los fixes es la penalización inherente de FedAvg en datos no-IID extremos por institución (cmmd 79% maligno vs kau-bcmd 4%) + el efecto del scheduler que reinicia cada ronda (F10) + presupuesto de entrenamiento corto (F9) — todos documentados como contribuyentes secundarios, no bugs, y no arreglados en esta pasada (requerirían decisiones de diseño de FL, p. ej. adoptar `fedbn` para no promediar buffers BN entre instituciones muy heterogéneas).

### Próximos pasos sugeridos (no ejecutados — requieren tu decisión)
1. Re-ejecutar exp12/13/14 vía `scripts/docker-deploy-federated.sh` y comparar contra exp10.
2. Considerar poner `strict_load: true` en los configs federados como hardening adicional (ahora que el guard de 0-match existe, esto ya no debería ser necesario para *detectar* el bug, pero endurece contra regresiones futuras de shape).
3. Evaluar la estrategia `fedbn` (ya registrada en el repo, `federated/strategies/fedbn.py`) para exp12 dado el freeze parcial + heterogeneidad entre instituciones.
4. Opcional: aplicar B5 (val patient-disjoint) y B3 (shape check explícito) si se quiere robustecer más allá de los bugs que causan el colapso.
