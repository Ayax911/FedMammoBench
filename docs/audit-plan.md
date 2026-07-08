# Plan de Auditoría Técnica — Federal-Learning (Mamografía)

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

| # | Bug | Archivo | Línea | Severidad |
|---|-----|---------|-------|-----------|
| B1 | **BatchNorm drift con backbone frozen**: `model.train()` reactiva BN training mode en capas congeladas → estadísticas BN divergen entre clientes cada ronda | `models/weight_loaders/__init__.py` | 111–118 | CRÍTICO (causa principal) |
| B2 | Agregación de parámetros posiblemente **no ponderada por muestras** (Flower FedAvg por defecto usa igual peso por cliente) | `federated/strategies/fedavg.py` | 22–43 | CRÍTICO |
| B3 | `load_ndarrays_to_state_dict` verifica longitud pero **no shape** de tensores — mismatch silencioso posible | `federated/param_utils.py` | 22–35 | ALTO |
| B4 | `local_unfreeze_at_epoch` compara contra época **local** (se resetea cada ronda) en lugar de ronda global — el backbone se descongela y reconjela cada ronda | `federated/client.py` | 174–197 | ALTO |
| B5 | Split de validación usa `patient_ids=None` → posible leakage paciente-nivel en métricas de validación federada | `federated/client.py` | 395–404 | MEDIO |

> **B1 es la causa más probable del colapso**: local ~80% vs global ~51% es exactamente el patrón de BN stats desincronizadas entre clientes — cada cliente converge a sus propias estadísticas, pero el promedio de parámetros no promedia las BN stats de forma compatible.

---

## Plan de Auditoría por Módulos

### Módulo 1 — Sistema de Configuración (`configs/`)
**Objetivo**: Verificar que los YAMLs se cargan, heredan y validan correctamente.

**Archivos clave**:
- `configs/loader.py` — resolución de `defaults:` y herencia desde `base.yaml`
- `configs/schema.py` — re-exports y deserialización a dataclasses
- `configs/exp*/server.yaml`, `client.yaml`

**Checklist de auditoría**:
- [ ] **C1.1** ¿El loader resuelve correctamente rutas con doble slash (`//`)? Probar con `pathlib.Path` vs `os.path.join`.
- [ ] **C1.2** ¿`strict_load: false` silencia errores de forma segura o puede enmascarar mismatch de arquitectura (ej. head con shape diferente)?
- [ ] **C1.3** ¿Las claves `lr_head` / `lr_backbone` del cliente sobreescriben correctamente las del servidor o se fusionan de forma incorrecta?
- [ ] **C1.4** ¿`ExperimentConfig.validate()` detecta el mismatch `t_max` server vs client?
- [ ] **C1.5** ¿El `num_classes: 1` (exp12–14) con pérdida `bce` pasa la validación cruzada correctamente?

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
- [ ] **C2.1** ¿El split train/val es reproducible (misma semilla) entre corridas?
- [ ] **C2.2** ¿Las transformaciones de augmentación se aplican SOLO en train y no en val/test?
- [ ] **C2.3** ¿El `val_fraction` se aplica por clase (estratificado) o al azar? Con datasets pequeños (DMID: 52 imgs) el split aleatorio puede dejar una clase sin muestras en val.
- [ ] **C2.4** ¿El manifest de cada nodo referencia las mismas imágenes que el servidor de evaluación usa para test? (sin data leakage ni huecos).
- [ ] **C2.5** ¿`auto_class_weights` computa los pesos sobre el conjunto de ENTRENAMIENTO del cliente, no sobre el total?

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
- [ ] **C3.1** Con `strict_load: false`, ¿se loguean las claves que NO coinciden? Si no hay log, la carga silenciosa puede resultar en pesos aleatorios sin aviso.
- [ ] **C3.2** ¿El checkpoint de exp07 (`final.pt`) tiene `num_classes=1` igual que exp12–14? Si el head tiene shape diferente, `strict_load: false` descartaría esos pesos.
- [ ] **C3.3** ¿La función `build_model` retorna el modelo en modo `.eval()` o `.train()`? Debe quedar en `.train()` antes de pasar al cliente.
- [ ] **C3.4** Cuando `weight_source=custom` y el archivo no existe, ¿falla con error claro o continúa con pesos aleatorios?

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
- [ ] **C4.1** `apply_freeze_policy`: ¿`local_unfreeze_at_epoch` cuenta épocas locales (0-indexed vs 1-indexed)? Un off-by-one puede hacer que layer4 nunca se descongele en runs de pocos epochs.
- [ ] **C4.2** ¿Después de descongelar, el optimizador incluye los nuevos parámetros? Si el optimizador se creó antes del `unfreeze`, los nuevos params no tienen gradientes activos.
- [ ] **C4.3** ¿El scheduler se reinicia entre rondas federadas o acumula pasos? Un scheduler que no se reinicia aplica LR errónea desde la ronda 2 en adelante.
- [ ] **C4.4** ¿`BCEWithLogitsLoss` recibe logits (no probabilidades)? Verificar que no hay un `sigmoid` antes de la pérdida.
- [ ] **C4.5** ¿`auto_class_weights` produce pesos finitos cuando hay desbalance extremo (ej. 90/10)?

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
- [ ] **C5.1** ¿El promedio ponderado en `aggregate_fit` usa `num_examples` de cada cliente como peso? Verificar que la suma de pesos normaliza a 1.0 exactamente.
- [ ] **C5.2** ¿Los parámetros se desserializan (de bytes a tensores) antes de promediar o se promedian como bytes? Un error aquí corrupta los pesos silenciosamente.
- [ ] **C5.3** ¿Se incluyen TODOS los parámetros del modelo (incluyendo `running_mean`/`running_var` de BatchNorm)? Si los buffers de BN no se agregan, el modelo global usará estadísticas incorrectas.
- [ ] **C5.4** ¿FedProx implementa el término proximal μ/2 ||w - w_global||² correctamente en el cliente, o solo en el servidor?
- [ ] **C5.5** Si `server_training.enabled=False`, ¿el nuevo modelo global se guarda antes o después de la evaluación de clientes?

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
- [ ] **C6.1** ¿`set_parameters` carga pesos del servidor con `strict=True`? Si usa `strict=False` aquí, el cliente podría entrenar con pesos parcialmente actualizados.
- [ ] **C6.2** ¿El optimizador se reinicia en cada ronda (nueva instancia) o persiste entre rondas (acumulando momentum de rondas anteriores)? Momentum acumulado entre rondas puede producir divergencia.
- [ ] **C6.3** ¿`get_parameters` devuelve los pesos en el mismo orden que `set_parameters` los espera? Un mismatch de orden es un bug crítico silencioso.
- [ ] **C6.4** ¿`num_examples` retornado en `fit()` corresponde al tamaño real del dataset de entrenamiento (no incluyendo val)?
- [ ] **C6.5** ¿El cliente aplica `model.eval()` durante la evaluación y `model.train()` durante el entrenamiento?

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
- [ ] **C7.1** ¿`server_federated_metrics.csv` refleja evaluación sobre val local de cada cliente o sobre un test set centralizado?
- [ ] **C7.2** ¿El promedio ponderado de métricas en `aggregate_evaluate` usa los mismos pesos (`num_examples`) que `aggregate_fit`?
- [ ] **C7.3** ¿AUC se calcula con `predict_proba` (sigmoid de logits) y no con `predict` (clase binaria)?
- [ ] **C7.4** Con `num_classes=1`, ¿el umbral de clasificación es 0.5 sobre la salida sigmoid? ¿Está hardcodeado o es configurable?
- [ ] **C7.5** ¿`NodeMetricsRecorder` guarda el modelo del round con mejor métrica o el del último round?

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
- [ ] **C8.1** ¿`run_simulation` y `run_grpc_server` comparten exactamente la misma inicialización de estrategia y logging?
- [ ] **C8.2** ¿El seed de reproducibilidad se fija antes de inicializar el modelo y los dataloaders?
- [ ] **C8.3** ¿Los logs de error de carga de checkpoint son visibles o suprimidos por el logger?

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
1. Todos los checklist items tienen resultado documentado (bug / correcto / no aplica).
2. Se explica el gap local (~80%) vs global (~51–57%) con evidencia de código.
3. Las correcciones aplicadas producen resultados federados ≥ 65% accuracy en exp12 (o se documenta que el gap es inherente a FL con esta heterogeneidad de datos).
