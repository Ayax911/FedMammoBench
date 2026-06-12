# POST_FIX_AUDIT.md — Revisión técnica de los fixes C1/C2/C3

> **Alcance**: validación archivo por archivo de las correcciones aplicadas en la Fase 1 sobre `fedmammobench`. No re-audita problemas fuera del scope de C1/C2/C3 (esos siguen en `TECHNICAL_DEBT_REPORT.md`).
> **Fecha**: 2026-05-23
> **Metodología**: lectura directa del código actual + análisis cruzado con archivos relacionados (`evaluator.py`, `fedavg.py`, `cbis_ddsm.py`, `mammo_bench.py`, `metrics.py`). No se ejecutó el simulador Flower; se razona sobre invariantes.

---

## Resumen ejecutivo

La Fase 1 aplicó tres correcciones críticas y agregó tres tests nuevos. Tras una revisión post-implementación rigurosa:

- **C1 (patient leakage)** quedó **parcialmente cerrado**. La ruta inter-cliente está bien implementada y los esquemas IID/Dirichlet/quantity-skew patient-aware son matemáticamente correctos. Sin embargo, **una segunda ruta de patient leakage entre train y val persiste** (`cbis_ddsm.py:198-205` y `mammo_bench.py` análogo): cuando el manifest trae columna `split` parcial sin `val`, el dataset carva val tomando un slice a nivel imagen, sin agrupar por paciente. El val replicado a clientes puede compartir pacientes con cualquier train de cualquier cliente, replicando el bug que C1 pretendía eliminar. **Adicionalmente** el fix introdujo una regresión silenciosa de reproducibilidad: el dataset sintético (con `patient_id` único por muestra) ahora consume el RNG en distinto orden, por lo que mismo seed produce particiones diferentes que pre-fix.

- **C2 (FedProx)** quedó **matemáticamente correcto en FP32** pero **numéricamente degradado en AMP**. El término proximal `(mu/2)·Σ‖w-w_g‖²` se computa dentro de `torch.cuda.amp.autocast()`. Bajo FP16, la sustracción entre dos tensores FP32 cercanos pierde precisión catastróficamente; para mu típico (1e-3 a 1e-1) y drift pequeño tras pocas épocas, el término puede underflow a cero y FedProx degenera silenciosamente a FedAvg — exactamente el bug que C2 pretendía eliminar, ahora reaparecido bajo AMP. Adicionalmente, el `train_loss` reportado al servidor incluye el penalty, por lo que comparar curvas FedAvg vs FedProx ya no compara la misma cantidad.

- **C3 (NaN handling)** quedó **correcto y completo**. La omisión de métricas NaN en el cliente es consistente con el filtro `v == v` que ya hacía `_weighted_average` en el servidor. Persisten preocupaciones menores no introducidas por el fix: el `loss` returnado no se chequea por NaN, los logs de omisión son a nivel `DEBUG` (invisibles en producción), y el fallback `loss=0.0` puede enmascarar fallos del Evaluator.

- **Tests**: los 3 tests añadidos son funcionalmente correctos pero **débiles ante regresiones**. T1/T2 verifican disjuntez de pacientes solo con un caso fácil (2 clientes × 8 pacientes), T5/T6/T7 usan mu=10.0 (no realista) y tolerancia `>1e-6` (excesiva), T10 verifica solo `roc_auc`. Faltan tests críticos para: train↔val leakage (gap N1), FedProx con AMP (gap N7), agregación con dicts mixtos en `_weighted_average`, y end-to-end del strategy FedProx.

---

## C1 — Validación de "patient-aware partitioning"

### Qué se hizo

`partition_indices()` (`datasets/partitioning.py`) recibe un parámetro opcional `patient_ids`. Cuando se provee:

- Se construye `_patient_groups()` que mapea cada paciente único a sus índices de imagen.
- Cada esquema (IID / Dirichlet / quantity_skew) opera **a nivel paciente** y luego expande a índices de imagen.
- `_materialize_client_partitions()` (`federated/client.py:192-240`) extrae `train_ds.patient_ids` y lo pasa al partition.

### Qué funciona

- **Inter-cliente**: ningún paciente puede aparecer en dos clientes. Verificado por inspección de `_iid_partition_patients` (líneas 154-171), `_dirichlet_partition_patients` (174-237) y `_quantity_skew_partition_patients` (240-266): los tres construyen `client_idx[ci]` extendiendo `index_groups[pi]` completos, nunca splittiendo un paciente.
- **Dirichlet patient-aware**: el algoritmo aplica `rng.dirichlet([alpha]·num_clients)` por clase a nivel paciente. Como el paper FedDirichlet trabaja a nivel sample, aplicarlo a nivel paciente preserva aproximadamente el non-IID label-skew esperado a nivel imagen (con varianza adicional dependiente del image-count-per-patient).
- **Backward compatibility**: cuando `patient_ids=None`, el flujo cae en los esquemas originales (`_iid_partition`, `_dirichlet_partition`, `_quantity_skew_partition`), preservados intactos en líneas 273-350.
- **Fallback explícito**: `client.py:210` detecta `None` patient_ids y emite warning, en lugar de fallar silenciosamente.

### Hallazgos (Nx)

#### 🔴 N1 — CRÍTICO: train↔val patient leakage persiste

**Ubicación**: `datasets/cbis_ddsm.py:188-205` y `datasets/mammo_bench.py:191-205` (mismo patrón).

**Detalle técnico**: Cuando el manifest trae columna `split` con valores `train`/`test` pero ningún `val`, el código rellena val así:

```python
# cbis_ddsm.py:198-205
if not splits["val"] and val_fraction > 0:
    rng = np.random.default_rng(seed)
    train_idx = np.array(splits["train"])
    rng.shuffle(train_idx)               # <-- shuffle A NIVEL IMAGEN
    n_val = int(len(train_idx) * val_fraction)
    splits["val"] = train_idx[:n_val].tolist()
    splits["train"] = train_idx[n_val:].tolist()
```

No hay agrupación por paciente. Si un paciente tiene 4 imágenes y todas estaban en `splits["train"]`, esta operación puede asignar 2 a val y 2 a train. El `MammographyDataset` resultante tiene leakage train↔val a nivel paciente.

**Riesgo**: Después de C1, las imágenes train de un cliente pueden compartir paciente con el val global (replicado a todos los clientes en `client.py:233`). Cualquier cliente evalúa contra pacientes que vio entrenando — exactamente el bug que C1 pretendía eliminar.

**Impacto**: AUC, sensitivity, specificity inflados de forma no detectable salvo inspección manual del manifest. **Cualquier resultado generado con un manifest CBIS-DDSM o Mammo-Bench que tenga `split: train/test` sin `val` y `data.val_fraction > 0` (default 0.1) sigue inválido para publicación.**

**Recomendación**: aplicar `_stratified_patient_split()` (que ya existe en `cbis_ddsm.py:50-99` y es patient-disjoint) al subconjunto train cuando se necesite carvar val. El patrón ya está en el repositorio; reusar.

#### 🟠 N2 — IMPORTANTE: regresión silenciosa de reproducibilidad

**Ubicación**: `datasets/partitioning.py:111-119` (nueva ruta patient-aware) vs `:81-109` (ruta sample-level preservada).

**Detalle técnico**: `SyntheticMammographyDataset` (`datasets/synthetic.py:48`) asigna `patient_id=f"pid_{i}"` único por muestra. Por lo tanto, todos los samples del sintético tienen patient_ids no-None, y el fix de C1 enrutándolos al patient-aware path:

- **Pre-fix**: `_iid_partition(n, num_clients, rng)` → `rng.shuffle(idx)` → `np.array_split`. Una sola consumición del RNG.
- **Post-fix**: `_iid_partition_patients(...)` → `rng.shuffle(patient_order)` + loop con `rng.shuffle(indices)` interno. **Múltiples consumiciones**.

Resultado: con mismo seed=42, el orden de consumición del RNG difiere → particiones diferentes → modelos finales diferentes → métricas diferentes.

**Riesgo**: cualquier baseline guardado en `runs/` con la versión pre-fix (TensorBoard, checkpoints, CSVs) ya no se reproduce bit-a-bit con la nueva versión. No hay CHANGELOG, no hay bump de version (`pyproject.toml` aún en 0.1.0), no hay warning al usuario.

**Impacto**: comparaciones "antes vs después" para tesis/papers son inválidas si los runs anteriores se generaron con la versión vieja. El usuario podría atribuir mejoras de métrica a "patient-aware" cuando en realidad son producto de un seed RNG distinto.

**Recomendación**: 
1. Bump version a `0.2.0` en `pyproject.toml`.
2. Añadir CHANGELOG.md con nota explícita sobre la ruptura de reproducibilidad.
3. Opcional: ofrecer flag `legacy_partitioning: bool = False` en `PartitioningConfig` para reproducir exactamente runs pre-fix.

#### 🟠 N3 — IMPORTANTE: NaN patient_id no detectado

**Ubicación**: `federated/client.py:209-217`.

**Detalle técnico**:

```python
raw_pids = train_ds.patient_ids                              # list[str | None | float-nan]
if any(pid is None for pid in raw_pids):                     # <-- solo detecta None de Python
    ...                                                       # fallback con warning
else:
    patient_ids_arr = np.asarray(raw_pids, dtype=object)     # NaN entra como dtype=object
```

`pandas.read_csv()` convierte celdas vacías de columnas string a `float('nan')`, NO a `None`. Los CSVs CBIS-DDSM/Mammo-Bench (`cbis_ddsm.py:179-181`, `mammo_bench.py:184`) hacen `str(row[columns.patient_id])` que convierte NaN → string `"nan"`. ¡Eso defendería el check de N3! 

Pero hay un caso peor: si `columns.patient_id` no está en el df (`if columns.patient_id in df.columns` en `cbis_ddsm.py:180`), `patient_id` se queda como `None`. Eso sí lo detecta el check actual.

**El riesgo real** es para implementaciones futuras de datasets que no envuelvan en `str(...)`: cualquier patient_id NaN llega como float NaN y:
- `dict.fromkeys` lo trata como clave única (`NaN != NaN`).
- Cada sample con patient_id NaN se convierte en "su propio paciente" único.
- El fix de C1 no protege esos samples.

**Recomendación**: cambiar el check a `any(pid is None or (isinstance(pid, float) and pid != pid) for pid in raw_pids)` (NaN-safe) y mover el chequeo a `MammographyDataset.patient_ids` para que sea defensivo en la base.

#### 🟡 N4 — MENOR: `np.bincount` con label outliers

**Ubicación**: `datasets/partitioning.py:143-146`.

**Detalle técnico**: `_patient_groups` calcula la etiqueta mayoritaria por paciente con `int(np.bincount(labels[grp]).argmax())`. `bincount` aloca un array de tamaño `max(labels[grp]) + 1`. Para labels binarios (0/1) el array es trivial. Para `num_classes > 2` con un label outlier (ej. corrupto = 9999), alocaría 10 000 ints (~80 KB) por paciente — explosivo si hay muchos pacientes con datos corruptos.

**Recomendación**: usar `int(np.argmax(np.bincount(labels[grp], minlength=num_classes)))` con `minlength` fijo, o `Counter(labels[grp]).most_common(1)[0][0]`.

#### 🟠 N5 — IMPORTANTE: `min_per_client` solo aplicado en patient-aware Dirichlet

**Ubicación**: `datasets/partitioning.py:154-171` (`_iid_partition_patients`), `:240-266` (`_quantity_skew_partition_patients`).

**Detalle técnico**: Solo `_dirichlet_partition_patients` (174-237) hace retry hasta cumplir `min_per_client`. Los otros dos esquemas pueden producir clientes con muy pocos samples si un cliente recibe pacientes con pocas imágenes (extremo: `num_patients == num_clients` y un cliente recibe un paciente con 1 imagen → ese cliente entrena con 1 sample).

Esto era ya cierto en la versión pre-fix sample-level, pero la heterogeneidad image-count-per-patient es estructural en datasets médicos (un paciente puede tener 2 vistas vs 24 vistas), así que el riesgo es **estructuralmente mayor** post-fix.

**Riesgo**: trainer con loader vacío o casi vacío → `mean_loss = total_loss / max(n_samples, 1)` (trainer.py:139) divide por 1 → loss artificial. Agregación FedAvg con peso=1 vs peso=1000 sesga al cliente grande.

**Recomendación**: aplicar el mismo retry de `_dirichlet_partition_patients` (con re-shuffle del patient order) a IID y quantity_skew patient-aware.

#### 🟢 N6 — FUTURO: validación duplicada

**Ubicación**: `datasets/partitioning.py:78` vs `:91`.

`num_clients > n` (línea 78) y `num_clients > n_patients` (línea 91). Siempre `n_patients ≤ n`, por lo que el segundo subsume al primero cuando `patient_ids is not None`. El primero queda como redundancia inútil en esa rama. Trivial.

---

## C2 — Validación de "FedProx real"

### Qué se hizo

- `FedMammoBenchClient.fit()` (`federated/client.py:111-151`): lee `proximal_mu = float(config.get("proximal_mu", 0.0))`. Si `>0`, captura `global_params = [p.detach().clone() for p in self.model.parameters()]` antes de cualquier update. Pasa ambos a `Trainer.train_one_epoch()`.
- `Trainer.train_one_epoch()` (`training/trainer.py:76-158`): nuevo argumento `proximal_mu` y `global_params`. Si `use_prox`, calcula `prox = sum(((p - g) ** 2).sum() for p, g in zip(...))` y suma `(proximal_mu / 2.0) * prox` al loss task.
- `FedProx` strategy (`federated/strategies/fedprox.py`): docstring actualizado, warning de "stub" eliminado. El `configure_fit()` sigue inyectando `proximal_mu` en `FitIns.config` como antes.

### Qué funciona

- **Fórmula matemática**: `(mu/2) · Σ(p - g)²` es exactamente la del paper original FedProx (Li et al., 2020, ec. 2).
- **Captura del estado global**: `clone()` se hace ANTES de modificar el modelo, así que `global_params` es genuinamente w_global (parámetros del servidor recién recibidos vía `load_ndarrays_to_state_dict` en línea 114).
- **Fresh optimizer per round**: línea 126 reconstruye el optimizer, así que ningún estado de optimizer del round anterior contamina FedProx (correcto).
- **Test funcional**: `test_fedprox_proximal_term_applied` demuestra que mu=10 produce parámetros ≠ mu=0 (al menos en CPU/FP32).
- **Pass-through perfecto**: cuando `proximal_mu=0`, `use_prox=False` y el código original del Trainer corre intacto. No hay regresión para FedAvg.

### Hallazgos (Nx)

#### 🔴 N7 — CRÍTICO: AMP underflow del término proximal

**Ubicación**: `training/trainer.py:106-114`.

**Detalle técnico**:

```python
if self._scaler is not None:
    with torch.cuda.amp.autocast():
        logits = self.model(images)
        loss = self.criterion(logits, targets)
        if use_prox:
            prox = sum(
                ((p - g) ** 2).sum()                              # <-- DENTRO de autocast
                for p, g in zip(self.model.parameters(), global_params, strict=True)
            )
            loss = loss + (proximal_mu / 2.0) * prox
```

Bajo `autocast()`, `(p - g)` opera sobre tensors FP32 (los parámetros del modelo siguen siendo FP32) **PERO el resultado se castea a FP16** por las reglas de mixed precision: cualquier operación de tipo "promotable" (suma, resta, multiplicación) dentro de autocast devuelve dtype = autocast's `dtype` (por defecto FP16).

FP16 tiene ~3 dígitos decimales de precisión. Para mu típico (0.001-0.1):
- En las primeras épocas (drift grande), `(p-g)²` ≈ 1e-4 a 1e-2 → representable en FP16, OK.
- Tras varias rondas (drift pequeño cuando el modelo converge), `(p-g)` ≈ 1e-3 → `(p-g)²` ≈ 1e-6 → **underflow a 0 en FP16** (smallest normal positive ≈ 6e-5).
- Con mu = 0.01, `(mu/2) * prox` ≈ 0 → loss sin penalty → degenera a FedAvg.

**El bug que C2 pretendía corregir reaparece bajo AMP. Silenciosamente.**

**Riesgo**: cualquier despliegue producción con `mixed_precision: true` (común en GPUs Volta/Ampere) ejecutando FedProx con mu típico estará potencialmente corriendo FedAvg. El test no lo detecta (CPU + mu=10 fuerza ambos caminos a comportarse).

**Recomendación** (cambio mínimo en `trainer.py:106-114`):

```python
if self._scaler is not None:
    with torch.cuda.amp.autocast():
        logits = self.model(images)
        loss = self.criterion(logits, targets)
    if use_prox:
        # Computar proximal en FP32 fuera de autocast
        with torch.cuda.amp.autocast(enabled=False):
            prox = sum(
                ((p.float() - g.float()) ** 2).sum()
                for p, g in zip(self.model.parameters(), global_params, strict=True)
            )
            loss = loss + (proximal_mu / 2.0) * prox
```

#### 🟠 N8 — IMPORTANTE: `train_loss` reportado incluye penalty

**Ubicación**: `training/trainer.py:135` y `federated/client.py:145-147`.

**Detalle técnico**: `total_loss += float(loss.item()) * targets.size(0)` (línea 135) acumula el loss DESPUÉS de añadir el penalty proximal. El `mean_loss` retornado (línea 139) y a su vez el `train_loss` enviado al servidor (`client.py:146-147`) refleja `task_loss + (mu/2)·||w-w_g||²`.

**Riesgo**: 
- Curvas "train_loss vs round" en TensorBoard ya no comparan FedAvg vs FedProx en la misma escala.
- En early rounds, FedProx muestra loss más alto (penalty grande) aunque el task loss sea igual.
- En late rounds, ambos convergen porque el penalty tiende a 0.
- Una figura de "FedProx loss curve" en un paper sería metodológicamente engañosa sin disclaimer.

**Recomendación** (cambio mínimo en `trainer.py`):

```python
total_task_loss = 0.0
total_combined_loss = 0.0
for images, targets in loader:
    ...
    logits = self.model(images)
    task_loss = self.criterion(logits, targets)
    if use_prox:
        prox = sum(((p - g) ** 2).sum() for p, g in zip(...))
        combined_loss = task_loss + (proximal_mu / 2.0) * prox
    else:
        combined_loss = task_loss
    combined_loss.backward()
    ...
    total_task_loss += float(task_loss.item()) * targets.size(0)
    total_combined_loss += float(combined_loss.item()) * targets.size(0)

return {"loss": mean_combined_loss, "task_loss": mean_task_loss, "samples": n_samples}
```

Y reportar `task_loss` al servidor por defecto (es lo que el usuario espera al ver "train_loss").

#### 🟠 N9 — IMPORTANTE: memoria 2× con modelos grandes

**Ubicación**: `federated/client.py:122`.

**Detalle técnico**: `global_params = [p.detach().clone() for p in self.model.parameters()]` duplica todos los parámetros entrenables. Para los backbones que el repo planea soportar (RadImageNet):

| Backbone | Params | FP32 size | Memoria extra |
|----------|--------|-----------|---------------|
| ResNet18 | 11.7M | 47 MB | 47 MB |
| ResNet50 | 25.6M | 102 MB | 102 MB |
| InceptionResNet-V2 | 55.8M | 223 MB | 223 MB |

Esta memoria vive todo el `fit()` (líneas 122-143). Para GPUs producción con múltiples replicas o batch sizes grandes podría disparar OOM. El test (ResNet18 + CPU) no detecta el problema.

**Recomendación**: 
- Documentar en RADIMAGENET_READINESS_REPORT.md como bloqueador antes de subir a backbones grandes.
- Considerar mover `global_params` a CPU explícitamente para liberar VRAM, recuperándolos por batch (con costo de PCIe bandwidth).

#### 🟠 N10 — IMPORTANTE: clip_grad_norm clipea total (task + proximal)

**Ubicación**: `training/trainer.py:116-118, 131-133`.

**Detalle técnico**: `loss.backward()` propaga gradientes de `(task_loss + prox_loss)`. `clip_grad_norm_(model.parameters(), grad_clip_norm)` corta la norma combinada. Con mu grande, la gradiente del proximal (escalar `mu * (p - g)` por parámetro) puede dominar y consumir todo el budget de norma, dejando la gradiente de tarea efectivamente sin contribución.

El paper FedProx original no usa clip (corre con SGD vanilla). El default del repo (`training.grad_clip_norm: 0.0`) evita el problema por ahora.

**Riesgo**: si un usuario activa clipping con FedProx (sensato si el modelo diverge), el comportamiento se desvía del paper sin warning.

**Recomendación**: documentar la interacción. Opcional: clipear gradientes task y proximal por separado.

#### 🟡 N11 — MENOR: AdamW weight_decay vs FedProx mu

**Ubicación**: `training/trainer.py:106-129` + `training/optim.py` (AdamW por defecto en `OptimizerConfig.name="adamw"`).

**Detalle técnico**: AdamW aplica weight decay `θ ← θ - η·λ·θ` (pulls hacia origen). FedProx aplica `θ ← θ - η·μ·(θ - θ_g)` (pulls hacia θ_g). No son redundantes, son fuerzas diferentes con distinta dirección de origen. Pero combinadas pueden producir convergencia inesperada (especialmente si `λ` y `μ` están en escalas similares).

El default `weight_decay: 1e-4` y mu sugerido `0.01` no entran en conflicto serio.

**Recomendación**: documentar la interacción en RADIMAGENET_READINESS_REPORT.md (donde el fine-tuning suele bajar weight_decay).

#### 🟡 N12 — MENOR: BatchNorm running stats no regularizados

**Ubicación**: `training/trainer.py:110-113, 125-128`.

**Detalle técnico**: `self.model.parameters()` excluye buffers (`running_mean`, `running_var`, `num_batches_tracked`). El término proximal **no** los regulariza. Sin embargo, `param_utils.state_dict_to_ndarrays` SÍ los incluye en la agregación FedAvg. Asimetría:

- Pesos: regularizados local (FedProx) + agregados global (FedAvg).
- Stats BN: no regularizados local + agregados global.

Esto coincide con el paper FedProx (que ignora BN explícitamente; FedBN se diseñó precisamente para tratarlas distinto). Pero no está documentado.

**Recomendación**: docstring del cliente. Si en el futuro se implementa FedBN, esto se aclara naturalmente.

#### 🟢 N13 — FUTURO: recomputación per-batch del proximal

**Ubicación**: `training/trainer.py:110-113, 125-128`.

**Detalle técnico**: O(P) por batch, donde P = total params. Para 11M params + 100 batches/epoch ≈ 10-30% overhead. No es bug; optimización futura (cachear gradiente del proximal, pre-stack en tensor único, etc.).

---

## C3 — Validación de "NaN propagation cliente→servidor"

### Qué se hizo

`FedMammoBenchClient.evaluate()` (`federated/client.py:153-185`): el dict comprehension previo que mapeaba NaN→0.0 se reemplazó por un loop explícito que **omite** los keys NaN (con log a nivel DEBUG) en lugar de emitirlos como 0.0.

### Qué funciona

- **Coherencia cliente↔servidor**: el servidor (`fedavg.py:38`) ya filtraba NaN con `v == v`. Antes del fix, el cliente "saturaba" los NaN como 0.0 → server los aceptaba como valores legítimos → contaminaba el promedio. Ahora cliente y servidor son simétricos: NaN se omite en ambos lados, `_weighted_average` reduce el peso total efectivo para ese key.
- **Casting correcto**: `np.isnan(float(result[k]))` maneja correctamente int/float (sklearn devuelve numpy scalars que castean a float sin error).
- **Test funcional**: `test_evaluate_nan_metric_omitted_not_zero` fuerza un val monoclase (forzando `roc_auc = NaN` en `metrics.py:70-73`) y verifica que `eval_metrics["roc_auc"] != 0.0`.

### Hallazgos (Nx)

#### 🟡 N14 — MENOR: `loss` retornado no se chequea por NaN

**Ubicación**: `federated/client.py:160, 185`.

**Detalle técnico**: 
```python
loss = float(result.get("loss", 0.0))
...
return loss, n_samples, metrics
```

El loss se retorna SEPARADAMENTE del dict `metrics`. Flower agrega losses (con weighted_loss_avg) separadamente. Si criterion produce NaN (gradiente explotado), `loss = NaN`, y el server lo propaga al `History`.

**Riesgo**: en TensorBoard server-side aparecen losses NaN sin diagnóstico. C3 mejoró el dict `metrics` pero no el loss. Inconsistencia menor.

**Recomendación**: en `client.py`, antes de retornar, verificar `if not np.isfinite(loss): _logger.warning("client %d: NaN/inf loss", self.client_id); loss = 0.0`. O reportar como sentinel `-1.0`.

#### 🟡 N15 — MENOR: `loss=0.0` fallback enmascara fallos

**Ubicación**: `federated/client.py:160` y `evaluation/evaluator.py:71-86, 94`.

**Detalle técnico**: 
- `Evaluator.evaluate()` retorna `loss: 0.0` si no hubo batches (línea 72) o si `criterion is None` o si `n_samples == 0` (línea 94).
- `client.py:160`: `loss = float(result.get("loss", 0.0))`.

Ambos defaults son 0.0. Un fallo upstream (loader vacío, criterion mal construido) se reporta como "loss perfecto = 0".

**Recomendación**: cambiar default a `float("nan")` y dejar que el filtro de N14 lo maneje uniformemente.

#### 🟡 N16 — MENOR: log a nivel DEBUG cuando se omite NaN

**Ubicación**: `federated/client.py:177-182`.

**Detalle técnico**: 
```python
_logger.debug(
    "client %d: metric %r is NaN (likely single-class batch); omitting from aggregation",
    self.client_id, k,
)
```

`logging.basicConfig(level=logging.INFO)` (default en producción) no muestra DEBUG. El operador no ve la omisión. Si TODAS las métricas son NaN, el dict queda vacío y `_weighted_average` retorna `{}` → server CSV sin row.

**Riesgo**: diagnóstico ambiguo entre "no metric calculated" y "all metrics NaN".

**Recomendación**: subir a `_logger.warning` cuando se omite **alguna** métrica clínicamente relevante (roc_auc, sensitivity, specificity). Mantener DEBUG para las demás. Adicional: si TODAS las métricas son NaN, emitir un `warning` agregado al final del `evaluate`.

---

## Tests — Calidad y cobertura

### Tests añadidos

Los 3 tests nuevos están en `tests/test_smoke.py`:

- `test_partitioning_patient_disjoint` (193-224): verifica que IID/Dirichlet/quantity-skew patient-aware no comparten pacientes entre clientes.
- `test_fedprox_proximal_term_applied` (227-257): mu=10 produce parámetros ≠ mu=0.
- `test_evaluate_nan_metric_omitted_not_zero` (260-295): roc_auc NaN se omite, no se mapea a 0.

### Hallazgos (Tx)

#### 🟠 T1 — IMPORTANTE: caso de stress débil

`test_partitioning_patient_disjoint` usa 2 clientes × 8 pacientes (ratio 4:1). Cualquier algoritmo trivial sin overlap pasa.

**Recomendación**: añadir escenarios con num_clients=4 × num_patients=4 (1 patient per client mínimo) y num_clients=8 × num_patients=8 (caso límite).

#### 🟠 T2 — IMPORTANTE: solo cuenta, no contenido

`tests/test_smoke.py:216`: `assert sum(len(p) for p in parts) == len(labels)`. Una implementación que duplique índices y descarte otros pasaría.

**Recomendación**: `assert set().union(*[set(p) for p in parts]) == set(range(len(labels)))` AND `assert sum(len(p) for p in parts) == len(labels)` (set Y suma para detectar duplicados).

#### 🟠 T3 — IMPORTANTE: rama None patient_ids sin cobertura

`client.py:210-215` (fallback warning + `patient_ids_arr = None`) no se ejerce por ningún test.

**Recomendación**: test con `MammographyDataset` cuyo `patient_ids` retorna `[None, None, ...]`.

#### 🟡 T4 — MENOR: ValueError `num_clients > n_patients` sin cobertura

`partitioning.py:91-94` no se ejerce.

**Recomendación**: test con `num_clients=10, patient_ids=["A", "B"]` esperando `ValueError`.

#### 🟠 T5 — IMPORTANTE: mu=10 no realista

`test_fedprox_proximal_term_applied` usa mu=10. Un bug donde mu se cap internamente a 1.0 (e.g., `proximal_mu = min(proximal_mu, 1.0)`) pasaría.

**Recomendación**: añadir test con mu=0.01 verificando que la diferencia es mensurable pero menor que mu=10.

#### 🟠 T6 — IMPORTANTE: no verifica que mu=0 es idéntico

El test solo verifica `mu=10 ≠ mu=0`. Un bug que siempre aplique FedProx con mu=0.0001 fijo (e.g., `proximal_mu = max(proximal_mu, 0.0001)`) pasaría: mu=0 y mu=10 producirían salidas distintas con/sin el bug.

**Recomendación**: añadir test que compare mu=0.0 contra `FedMammoBenchClient` baseline (sin proximal_mu en config) y verifique IDENTIDAD exacta (`np.allclose(a, b, atol=0.0, rtol=0.0)`).

#### 🟡 T7 — MENOR: tolerancia excesivamente permisiva

`assert any(d > 1e-6 for d in diffs)`. Con mu=10 y un epoch (8 batches), la diferencia esperada es órdenes de magnitud mayor. Una implementación que aplique solo una fracción del término proximal (ej. factor /1000 por bug en `(proximal_mu / 2.0)`) pasaría.

**Recomendación**: bound más estricto, por ejemplo `assert max(diffs) > 1e-3` para mu=10.

#### 🔴 T8 — CRÍTICO: sin cobertura AMP

Ningún test ejerce el path `self._scaler is not None` (trainer.py:105-120) donde vive N7. Requiere GPU o mock de `torch.cuda.amp.autocast`.

**Recomendación**: 
- Mínimo: añadir test que verifique que `proximal_mu = 0.001` produce diferencias mensurables CON mixed_precision False (ya el caso) AND CON mixed_precision True via mock o skip-if-no-cuda.
- Documentar explícitamente en el docstring del test que AMP NO está cubierto.

#### 🟡 T9 — MENOR: sin verificar train_loss task vs total

Después de aplicar el fix de N8 (separar `loss` task y combined), añadir test que con mu=10 verifique `train_loss_total > train_loss_task`.

#### 🟡 T10 — MENOR: solo chequea roc_auc

`test_evaluate_nan_metric_omitted_not_zero` verifica solo `roc_auc`. Un bug que dropee todas las métricas (e.g., `metrics = {}` siempre) pasaría.

**Recomendación**: añadir `assert "accuracy" in eval_metrics` (no debe ser NaN para val monoclase) y `assert "sensitivity" in eval_metrics` (depende de si hay positivos; en monoclase benign = 0 positivos → sens 0.0 que es no-NaN).

#### 🟡 T11 — MENOR: no verifica presencia de métricas no-NaN

Mismo test debería verificar que `accuracy`, `precision`, `recall`, `f1` están en `eval_metrics` (son calculables incluso en monoclase).

#### 🔴 T12 — CRÍTICO: sin test de train↔val patient leakage

Ningún test verifica que `build_dataset(cfg)` para CBIS-DDSM o Mammo-Bench produce splits patient-disjoint en train/val. Esto es el gap N1.

**Recomendación**: crear un fixture CSV minúsculo con columna `split` parcial y dos pacientes con múltiples imágenes; verificar que `set(datasets['train'].patient_ids) ∩ set(datasets['val'].patient_ids) == ∅`.

#### 🟠 T13 — IMPORTANTE: sin test end-to-end de FedProx strategy

Ningún test ejerce `build_strategy("fedprox", proximal_mu=0.5)` ni el flujo `configure_fit → client.fit → aggregate`. El test actual instancia `FedMammoBenchClient` directamente pasando `{"proximal_mu": 10.0}` en config — saltea el `FedProx.configure_fit`.

**Recomendación**: test que construya FedProx strategy, llame `configure_fit`, extraiga el config inyectado, y pase ese config al cliente real.

#### 🟠 T14 — IMPORTANTE: sin test de `_weighted_average` con dicts mixtos

Después de C3, algunos clientes reportarán `{"f1": 0.8}` y otros `{"f1": 0.7, "roc_auc": 0.85}`. La agregación debe omitir el peso de los que no reportan roc_auc, no contar como 0.0.

**Recomendación**: test unitario de `fedavg._weighted_average([(100, {"f1": 0.8}), (50, {"f1": 0.6, "roc_auc": 0.9})])` esperando `{"f1": (100*0.8+50*0.6)/150, "roc_auc": 0.9}`.

#### 🟡 T15 — MENOR: sin test con num_workers > 0

La config producción `fedavg_mammobench_client.yaml:47` fuerza `num_workers: 0` para evitar conflictos gRPC fork. Si en el futuro se relaja, los workers podrían heredar RNG state global de forma no-determinística. Sin test.

**Recomendación**: opcional, dado el constraint actual. Documentar.

---

## Tabla compacta de hallazgos

| ID | Sev | Categoría | Archivo:línea | Resumen |
|----|-----|-----------|---------------|---------|
| N1 | 🔴 | C1 | cbis_ddsm.py:198-205, mammo_bench.py:191-205 | train↔val patient leakage persiste con `split` parcial |
| N2 | 🟠 | C1 | partitioning.py:111-119 | reproducibilidad rota cross-version |
| N3 | 🟠 | C1 | client.py:209-217 | NaN patient_id no detectado en fallback |
| N4 | 🟡 | C1 | partitioning.py:143-146 | `np.bincount` sin `minlength` |
| N5 | 🟠 | C1 | partitioning.py:154-171, 240-266 | `min_per_client` no aplicado en IID/quantity patient-aware |
| N6 | 🟢 | C1 | partitioning.py:78, 91 | validación duplicada |
| N7 | 🔴 | C2 | trainer.py:106-114 | AMP underflow del proximal term |
| N8 | 🟠 | C2 | trainer.py:135, client.py:145-147 | train_loss reportado incluye penalty |
| N9 | 🟠 | C2 | client.py:122 | memoria 2× con modelos grandes |
| N10 | 🟠 | C2 | trainer.py:116-118, 131-133 | clip_grad_norm clipea task+proximal combinado |
| N11 | 🟡 | C2 | trainer.py:106-129 | AdamW + FedProx doble fuerza |
| N12 | 🟡 | C2 | trainer.py:110-113 | BN buffers no regularizados |
| N13 | 🟢 | C2 | trainer.py:110-113, 125-128 | recomputación per-batch del prox |
| N14 | 🟡 | C3 | client.py:160, 185 | loss retornado sin NaN check |
| N15 | 🟡 | C3 | client.py:160, evaluator.py:71-86, 94 | loss=0.0 fallback enmascara fallos |
| N16 | 🟡 | C3 | client.py:177-182 | log DEBUG invisible en producción |
| T1 | 🟠 | Tests | test_smoke.py:193-224 | caso de stress débil |
| T2 | 🟠 | Tests | test_smoke.py:216 | verifica conteo, no set equality |
| T3 | 🟠 | Tests | (faltante) | sin cobertura de rama None patient_ids |
| T4 | 🟡 | Tests | (faltante) | sin cobertura ValueError num_clients > n_patients |
| T5 | 🟠 | Tests | test_smoke.py:227-257 | mu=10 no realista |
| T6 | 🟠 | Tests | test_smoke.py:248, 252 | no verifica identidad mu=0 |
| T7 | 🟡 | Tests | test_smoke.py:255 | tolerancia 1e-6 excesiva |
| T8 | 🔴 | Tests | (faltante) | sin cobertura AMP |
| T9 | 🟡 | Tests | (faltante) | sin verificar task vs combined loss |
| T10 | 🟡 | Tests | test_smoke.py:293-295 | solo chequea roc_auc |
| T11 | 🟡 | Tests | (faltante) | no verifica presencia de no-NaN |
| T12 | 🔴 | Tests | (faltante) | sin test train↔val leakage |
| T13 | 🟠 | Tests | (faltante) | sin end-to-end FedProx strategy |
| T14 | 🟠 | Tests | (faltante) | sin test _weighted_average con dicts mixtos |
| T15 | 🟡 | Tests | (faltante) | sin test num_workers > 0 |

**Total**: 31 hallazgos. 4 críticos, 12 importantes, 13 menores, 2 futuros.

---

## Conclusión

C1 y C2 cerraron parcialmente sus bugs originales. Cada uno introdujo o dejó una nueva variante del mismo problema:

- **C1 → N1**: el leakage inter-cliente está cerrado, pero el leakage train↔val intra-dataset sigue abierto en la rama de fallback más común (manifest con `split` parcial).
- **C2 → N7**: el cliente sí lee `proximal_mu` y sí aplica el término, pero bajo AMP el término puede underflow a cero, regenerando FedAvg silenciosamente.
- **C3**: cerrado correctamente; sin nuevos bugs.

Los 3 tests añadidos son funcionalmente correctos pero excesivamente permisivos (mu=10, tolerancia 1e-6, 2 clientes solamente) y dejan sin cobertura los caminos exactos donde N1, N7 y T12 viven.

**No es momento de declarar la Fase 1 cerrada.** Una "Fase 1.5" mínima cierra N1, N7 y endurece los tests con costo ≈ 2-3 horas.
