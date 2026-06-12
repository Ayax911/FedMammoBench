# FEDERATED_VALIDITY_REVIEW.md — ¿Qué experimentos federados son defendibles científicamente hoy?

> **Audiencia**: revisor de paper, comité de tesis, comité interno de validación clínica.
> **Pregunta central**: dado el estado actual de `fedmammobench` (post-fixes C1/C2/C3), ¿cuáles de los experimentos configurables hoy producen resultados defendibles en una publicación científica seria?
> **Metodología**: razonamiento desde los hallazgos verificados en `POST_FIX_AUDIT.md`, aplicado por archivo de configuración existente en `configs/`.

---

## Matriz de defensibilidad

Filas = configs reales del repositorio. Columnas = aspectos críticos de validez federada.

Leyenda: ✅ defendible · ⚠️ defendible con disclaimer · ❌ no defendible.

| Config | Patient leakage inter-cliente | Patient leakage train↔val | FedProx correcto | Métrica val replicada | Reproducibilidad bit-a-bit | AMP-safe | Veredicto global |
|--------|-------------------------------|----------------------------|------------------|------------------------|------------------------------|----------|------------------|
| `centralized_synthetic.yaml` | N/A | ✅ (synthetic 1-img-per-patient) | N/A | N/A | ⚠️ (N2 silencioso) | N/A | ✅ |
| `fedavg_synthetic.yaml` | ✅ (cada sample es su propio paciente) | ✅ | N/A (no FedProx) | ❌ (M1) | ⚠️ (N2: nuevo seed → nueva partición) | ✅ (sintético no usa AMP) | ⚠️ (solo smoke test) |
| `fedavg_cbis_ddsm.yaml` | ✅ (C1 cerrado, sin split column → `_stratified_patient_split`) | ✅ (mismo motivo) | N/A | ❌ (M1) | ✅ | ⚠️ (depende de mixed_precision) | ✅ con disclaimer M1 |
| `fedavg_mammobench_server.yaml` + `_client.yaml` | ✅ (C1 cerrado, sin split column declarado en el YAML) | ⚠️ (depende del manifest físico — si tiene `split: train/test` sin `val`, **N1 lo rompe**) | N/A | ❌ (M1) | ✅ | ⚠️ | ⚠️ → ❌ si manifest tiene split parcial |
| `test_synthetic_server.yaml` + `_client.yaml` | ✅ | ✅ | N/A | ❌ (M1) | ⚠️ | ✅ | ⚠️ (test only) |
| FedProx hipotético: `fedavg_cbis_ddsm.yaml` + `strategy.name: fedprox` + `strategy.params: {proximal_mu: 0.01}` + `mixed_precision: false` | ✅ | ✅ | ✅ | ❌ (M1) + ⚠️ (N8 train_loss incluye penalty) | ✅ | ✅ (forzado FP32) | ⚠️ comparable pero curvas de loss engañosas |
| FedProx hipotético + `mixed_precision: true` | ✅ | ✅ | ❌ (**N7 underflow**: para mu < 0.1 podría degenerar a FedAvg) | ❌ | ✅ | ❌ | ❌ **NO PUBLICABLE** |

---

## Análisis caso por caso

### 1. `configs/centralized_synthetic.yaml`

Centralizado, solo para validar el pipeline de entrenamiento. No tiene componente federado.

- **Patient leakage**: N/A (centralizado).
- **Reproducibilidad post-fix**: el sintético antes consumía RNG vía `_iid_partition` y ahora vía `_iid_partition_patients`. Como `centralized_synthetic.yaml` NO toca partitioning (modo "centralized"), N2 no aplica. ✅
- **Veredicto**: ✅ defendible para lo que es (smoke test / sanity check).

### 2. `configs/fedavg_synthetic.yaml`

Smoke test federado con 4 clientes y 3 rondas.

- **Patient leakage inter-cliente**: Cada sample sintético tiene `patient_id="pid_{i}"` único. Patient-aware y sample-level coinciden trivialmente. ✅
- **Patient leakage train↔val**: el sintético construye `train/val/test` splits independientes en `factory.py:_build_synthetic`, no carva val de train. ✅
- **Métrica val replicada (M1)**: el val sintético se replica a los 4 clientes. Cada cliente evalúa contra los mismos datos. La métrica "federated AUC" agregada es ruido alrededor de un cálculo centralizado. ❌
- **Reproducibilidad (N2)**: con seed=42, la partición post-fix es DIFERENTE de pre-fix. Cualquier baseline guardado en `runs/fedavg_synthetic/` con la versión vieja no se reproduce. ⚠️
- **Veredicto**: ⚠️ válido como smoke test, **no como benchmark para publicar**.

### 3. `configs/fedavg_cbis_ddsm.yaml`

Configuración real con CBIS-DDSM, 5 clientes, Dirichlet α=0.5, 30 rondas. La pieza más cercana a "experimento publicable" en el repo.

Este YAML:
- Apunta a `manifest_path: data/cbis_ddsm/manifest.csv` y `image_root: data/cbis_ddsm/jpeg`.
- Usa `data.columns: {image_path, label: pathology, patient_id, split}`.

**El caso depende críticamente del CSV físico**:

- **Si `data/cbis_ddsm/manifest.csv` NO tiene columna `split`**: `from_manifest` cae en `_stratified_patient_split` (cbis_ddsm.py:207). Este es patient-disjoint. ✅ Patient leakage cerrado en ambas rutas (inter-cliente vía C1, train↔val vía dataset).

- **Si el CSV SÍ tiene columna `split` con valores `train/test` pero NO `val`**: cae en la rama 198-205 (image-level shuffle). ❌ **N1 activo**: train y val pueden compartir pacientes; el val replicado a clientes vía `client.py:233` puede compartir pacientes con cualquier train de cualquier cliente.

- **Si el CSV tiene `split: train/val/test` completo**: ✅ usa la columna directamente; cada split es disjoint según el CSV (responsabilidad del manifest).

**Cómo verificar antes de publicar**: ejecutar este check manual:

```python
from fedmammobench.configs.loader import load_config
from fedmammobench.datasets import build_dataset

cfg = load_config("configs/fedavg_cbis_ddsm.yaml")
datasets = build_dataset(cfg)
train_pids = set(p for p in datasets['train'].patient_ids if p)
val_pids = set(p for p in datasets['val'].patient_ids if p)
test_pids = set(p for p in datasets.get('test', datasets['val']).patient_ids if p)

assert not (train_pids & val_pids), f"train↔val leakage: {train_pids & val_pids}"
assert not (train_pids & test_pids), f"train↔test leakage: {train_pids & test_pids}"
assert not (val_pids & test_pids), f"val↔test leakage: {val_pids & test_pids}"
```

Si pasa: ✅ con disclaimer M1 sobre val replicado.
Si falla: ❌ no publicable hasta resolver N1.

### 4. `configs/fedavg_mammobench_server.yaml` + `fedavg_mammobench_client.yaml`

Escenario gRPC real planeado (2 nodos: ddsm/EE.UU. + cmmd/China).

Idéntico análisis a (3) pero con Mammo-Bench (`mammo_bench.py`). El módulo `mammo_bench.py:191-205` tiene **exactamente el mismo patrón vulnerable** que `cbis_ddsm.py:198-205`: si el CSV trae `split: train/test` sin `val`, carva val a nivel imagen.

**Particularidad MammoBench**: el `fedavg_mammobench_server.yaml` declara `data.test_fraction: 0.0` y `data.val_fraction: 0.0`. En el cliente, `val_fraction: 0.15`. Esto significa que cada nodo cliente carva val de su manifest local:

- Si el manifest del cliente (ej. `manifests/node0_manifest.csv`) tiene columna `split` parcial → N1 activo.
- Si el manifest del cliente NO tiene `split` → `_stratified_patient_split` aplica → ✅.

**Caso especial gRPC + val por cliente**: en este escenario el val NO se replica a los demás clientes (cada uno tiene su val local), así que M1 no aplica. Esto es realmente bueno metodológicamente. Pero hay otros problemas (configurados desde el server YAML):

- El server YAML usa `data.name: synthetic` con `synthetic_num_samples: 32` para tener un dataset "dummy". El comentario en líneas 22-23 dice: "Cuando el servidor no tiene imágenes reales, usa synthetic para que el servidor arranque sin necesitar datos locales. La evaluación centralizada queda deshabilitada (test set vacío)."
- Pero `synthetic_num_samples: 32` con `test_fraction: 0.0` → `n_test = max(8, int(32 * 0.0)) = 8` (factory.py:101). **El test set NO está vacío**, tiene 8 samples sintéticos.
- El `_build_evaluate_fn` (server.py:47-48): `if test_dataset is None or len(test_dataset) == 0: return None`. Con 8 samples sintéticos, evaluate_fn se construye → la "evaluación centralizada" del servidor MammoBench corre sobre 8 muestras sintéticas (basura aleatoria).
- **Resultado**: el "centralized AUC" reportado por el servidor MammoBench es ruido alrededor de 0.5. Sin valor diagnóstico. El operador podría no notarlo si solo mira los logs INFO.

**Veredicto**: ⚠️ — Mammo-Bench gRPC real es defendible **si y solo si** (a) el manifest físico no tiene split parcial, y (b) se desactiva o ignora la "centralized eval" del servidor (que es ruido sintético).

### 5. FedProx hipotético (FP32)

Si el usuario cambia `strategy.name: fedprox` y `strategy.params: {proximal_mu: 0.01}` en `fedavg_cbis_ddsm.yaml` y deja `mixed_precision: false`:

- ✅ El proximal term se aplica matemáticamente correcto (verificado en `trainer.py:106-129`).
- ✅ El test `test_fedprox_proximal_term_applied` confirma drift mensurable.
- ⚠️ El `train_loss` reportado incluye el penalty (N8) → curvas FedAvg vs FedProx no son comparables como están.
- ⚠️ Si el usuario activa `grad_clip_norm > 0`, comportamiento desvía del paper (N10).

**Veredicto**: ⚠️ defendible como FedProx funcional, pero las figuras de "train_loss vs round" requieren disclaimer ("loss includes proximal regularization term"). Comparar FedAvg vs FedProx debe usar **val/test metrics**, no train_loss.

### 6. FedProx hipotético (AMP)

Si además se activa `mixed_precision: true`:

- ❌ **N7 activo**: bajo autocast, para mu < 0.1, el término `((p-g)**2).sum()` puede underflow a 0 en FP16. FedProx degenera a FedAvg silenciosamente.
- Los outputs serían: "we compared FedAvg vs FedProx (μ=0.01)" → ambos son matemáticamente FedAvg → resultado falsamente "no significant difference".

**Veredicto**: ❌ **NO PUBLICABLE**. El bug original C2 (FedProx-as-FedAvg) reaparece en AMP. Cualquier comparación FedAvg vs FedProx hecha con AMP es inválida hasta resolver N7.

---

## M1 deep dive: el problema del val replicado

**Ubicación**: `client.py:233` — `pairs.append((sub_train, val_ds))`. El `val_ds` es el mismo objeto Python para todos los clientes (extraído de `datasets.get("val")` en línea 204).

**Lo que pasa en cada ronda**:
1. Server selecciona `fraction_evaluate` × `num_clients` clientes (por defecto todos).
2. Cada cliente seleccionado llama `evaluate()` sobre el mismo `val_loader`, con los mismos `parameters` (recién agregados por el server).
3. Cada cliente devuelve `(loss, n_samples, metrics)`. Los metrics son **idénticos** entre clientes (mismo modelo, mismos datos).
4. `_weighted_average` calcula `sum(metric_i * n_i) / sum(n_i)`. Como `metric_i = c` constante y `n_i = N` constante → resultado = `c`.

**Conclusión matemática**: el "federated val AUC" agregado es exactamente igual al "centralized val AUC" del modelo agregado. **No hay información adicional**. Toda la latencia y comunicación gRPC de fraction_evaluate se desperdicia.

**Por qué importa para la defensa científica**: presentar una figura "Federated val AUC over rounds" como evidencia de "que el sistema federado evalúa correctamente" es metodológicamente equivalente a presentar "Centralized val AUC over rounds". Sin disclaimer, lectores asumirán que es un agregado verdaderamente federado (cada cliente con su val local).

**Recomendación**:
1. **Para el paper actual**: usar `evaluate_fn` (server.py:39-93, ya configurado) que hace centralized eval correctamente, y reportar SOLO esa métrica. Setear `fraction_evaluate: 0.0` para evitar el ruido redundante.
2. **Para una versión futura "federated val real"**: particionar `val_ds` por cliente también (vía un parámetro `partition_val_too: bool` en config). Esto permitiría reportar variabilidad inter-cliente legítima.

---

## Reproducibilidad post-fix: efecto de N2

**Pre-fix**: dado seed=42 y synthetic dataset, `_iid_partition(n, num_clients, rng)` consume 1 `rng.shuffle`.

**Post-fix**: synthetic tiene `patient_id` único per sample, así que el código toma la ruta `_iid_partition_patients`. Esta consume `rng.shuffle(patient_order)` + 1 `rng.shuffle(indices)` por cliente.

**Consecuencia práctica**: ejecutar `pytest tests/test_smoke.py::test_partitioning_iid_and_dirichlet` con la versión post-fix produce una partición distinta de la pre-fix con el mismo seed.

Los runs guardados en `runs/` con la versión vieja (`runs/fedavg_synthetic_test/...`, etc.) NO se reproducen:
- Los modelos entrenados con esos seeds no convergen al mismo punto.
- Los TensorBoard guardados tienen métricas distintas a las nuevas runs.
- Cualquier comparación "antes vs después" del bug fix es inválida si no se conserva el ESTADO de partition.

**Recomendaciones**:

1. **Bump version**: `pyproject.toml:7` cambiar de `version = "0.1.0"` a `version = "0.2.0"`.
2. **Crear `CHANGELOG.md`** con sección "0.2.0 - 2026-05-23":
   ```markdown
   ## 0.2.0 (2026-05-23) — Patient-aware partitioning
   
   ### BREAKING CHANGES
   - Datasets with non-None `patient_ids` (including the synthetic dataset) now
     use patient-aware partitioning. RNG consumption order changed; experiments
     using the same seed will produce different partitions and metrics than 0.1.0.
   - Pre-existing runs under `runs/` cannot be reproduced bit-by-bit with 0.2.0.
   
   ### Fixes
   - C1: inter-client patient leakage in federated partitioning (closed).
     ...
   ```
3. **Opcional**: añadir flag `legacy_partitioning: bool = False` en `PartitioningConfig` que fuerce la ruta sample-level incluso cuando hay patient_ids. Útil para reproducir benchmarks pre-fix.

---

## Comparación FedAvg vs FedProx: qué métricas son legítimamente comparables

Dado N8 (train_loss reportado incluye penalty):

| Métrica | FedAvg vs FedProx comparable? | Razón |
|---------|--------------------------------|-------|
| train_loss (reportado al server) | ❌ No | FedProx incluye penalty proximal; FedAvg no. Escalas diferentes. |
| val_loss (server-side via evaluate_fn) | ✅ Sí | Se computa sin penalty (Evaluator no añade prox). |
| val_auc, val_sensitivity, val_specificity | ✅ Sí | Métricas de clasificación, agnósticas al loss. |
| Test final (después de la última ronda) | ✅ Sí | Idem val. |
| Tiempo por ronda | ⚠️ FedProx más lento (~10-30% por N13). Documentar. |
| Communication overhead | ✅ Igual (mismo payload de params). |

**Recomendación para tablas de paper**:
- Reportar **val_auc** o **test_auc** como métrica principal, NO train_loss.
- Si se incluye train_loss en figuras, separar `train_task_loss` (sin penalty) y `train_total_loss` (con penalty) — requiere fix de N8 primero.

---

## Recomendaciones experimentales: orden mínimo antes de generar resultados publicables

**Fase A: cerrar bloqueadores absolutos (1-2h)**
1. Fix N1: aplicar `_stratified_patient_split` al fallback "val from train" en `cbis_ddsm.py:198-205` y `mammo_bench.py:191-205`.
2. Fix N7: envolver el cómputo del proximal term en `with torch.cuda.amp.autocast(enabled=False):` y castear a `.float()` explícito.
3. Añadir test T12 (train↔val patient disjoint con manifest fixture).
4. Añadir test T8 (FedProx con AMP — al menos mock).

**Fase B: validez observabilidad (1-2h)**
5. Fix N8: separar `train_task_loss` y `train_total_loss` en el dict del Trainer.
6. Fix N3: NaN-safe patient_id check.
7. Fix N14: NaN-safe loss return.
8. Endurecer tests T1/T2/T5/T6/T7 según `POST_FIX_AUDIT.md`.

**Fase C: pre-publicación (validación manual)**
9. Para cada manifest físico que se va a usar en el paper, ejecutar el script de verificación de patient disjointness (sección 3 más arriba) y guardar el output como artifact.
10. Documentar en el paper / README el seed exacto, la versión del código (commit SHA), y la versión del manifest (hash MD5).
11. Si se compara FedAvg vs FedProx, ejecutar TODAS las corridas con `mixed_precision: false` o esperar Fase A fix de N7.

**Fase D: nice-to-have antes de defensa**
12. Setear `fraction_evaluate: 0.0` en todos los configs de paper y usar solo `evaluate_fn` server-side (resuelve M1 sin código adicional).
13. Reportar centralized eval del modelo final como métrica principal.

---

## Conclusión

Tras los fixes C1/C2/C3, el sistema produce resultados **defendibles para un subconjunto bien definido de configuraciones**:

- ✅ Centralizado sintético.
- ✅ Federado synthetic (como smoke test, no benchmark).
- ✅ Federado CBIS-DDSM o Mammo-Bench **si el manifest físico NO tiene columna `split` parcial**.
- ✅ FedProx en FP32 con disclaimer sobre N8.

Y **no defendibles** en:

- ❌ Cualquier manifest con `split: train/test` sin `val` + `val_fraction > 0` (N1).
- ❌ FedProx con AMP + mu < 0.1 (N7).
- ❌ Cualquier figura de "train_loss vs round" comparando FedAvg vs FedProx (N8).
- ❌ Cualquier "federated val metric" sin aclarar que es ruido sobre centralized eval (M1).
- ❌ Mammo-Bench server con "centralized eval" sobre 8 muestras sintéticas — ruido alrededor de 0.5 AUC.

**No deben publicarse resultados de FedProx hasta cerrar N7 explícitamente.** No deben publicarse resultados sobre manifests con split parcial hasta cerrar N1.
