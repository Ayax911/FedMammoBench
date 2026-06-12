# REMAINING_RISKS.md — Catálogo priorizado de riesgos remanentes y nuevos

> **Propósito**: única fuente de verdad sobre qué riesgos persisten en `fedmammobench` tras los fixes C1/C2/C3, ordenados por urgencia. Es la entrada al diseño de las próximas fases.
> **Cross-refs**: detalles técnicos por hallazgo en `POST_FIX_AUDIT.md`. Defensibilidad de cada configuración en `FEDERATED_VALIDITY_REVIEW.md`. Estado RadImageNet en `RADIMAGENET_READINESS_REPORT.md`.

---

## Riesgos críticos (bloqueadores para resultados defendibles)

| ID | Severidad | Categoría | Resumen | Impacto científico | Ubicación |
|----|-----------|-----------|---------|--------------------|-----------|
| **N1** | 🔴 CRÍTICO | C1 (incompleto) | Train↔val patient leakage cuando manifest tiene `split: train/test` pero no `val`, y `val_fraction > 0`. El val replicado a clientes vía `client.py:233` puede compartir pacientes con cualquier train de cualquier cliente. C1 cerró la ruta inter-cliente pero **dejó abierta esta ruta intra-dataset**. | Métricas inflado en cualquier experimento CBIS-DDSM/Mammo-Bench que use manifest con split parcial. **Bug original de patient leakage reaparece por otra ruta.** Resultados no publicables en revista médica. | `cbis_ddsm.py:198-205`, `mammo_bench.py:191-205` |
| **N7** | 🔴 CRÍTICO | C2 (incompleto bajo AMP) | El término proximal `((p-g)**2).sum()` se computa dentro de `torch.cuda.amp.autocast()`. Bajo FP16, para mu pequeño (0.001-0.1) y drift pequeño, underflow a cero → FedProx degenera silenciosamente a FedAvg. | El bug original C2 (FedProx-as-FedAvg) reaparece silenciosamente en AMP. Cualquier comparación FedAvg vs FedProx hecha con `mixed_precision: true` es inválida. | `trainer.py:106-114` |
| **T8** | 🔴 CRÍTICO | Tests (faltante) | Sin ningún test que ejercite el código path `self._scaler is not None`. La ruta exacta donde N7 vive. | Regresiones en AMP pasan invisibles. | (faltante) |
| **T12** | 🔴 CRÍTICO | Tests (faltante) | Sin ningún test que verifique disjuntez patient-level train↔val a nivel dataset (CBIS-DDSM, Mammo-Bench). | N1 puede reaparecer en futuros refactors sin alerta. | (faltante) |

---

## Riesgos importantes (degradan defensibilidad o robustez)

### Categoría C1 — Patient awareness incompleto

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **N2** | Regresión silenciosa de reproducibilidad. Synthetic dataset con `patient_id` único per sample ahora consume RNG en distinto orden → mismo seed → distinta partición que pre-fix. Sin CHANGELOG, sin version bump. | `partitioning.py:111-119` vs `:81-109` |
| **N3** | `any(pid is None for pid in raw_pids)` no detecta `float('nan')` de pandas. Patient_id ausente en CSV puede llegar como NaN y `dict.fromkeys` lo trata como clave única por sample → falsa sensación de safety. | `client.py:209-217` |
| **N5** | `_iid_partition_patients` y `_quantity_skew_partition_patients` no aplican `min_per_client`. Riesgo amplificado por heterogeneidad estructural de imágenes-por-paciente. Cliente con 1-2 imágenes posible. | `partitioning.py:154-171, 240-266` |

### Categoría C2 — FedProx con problemas secundarios

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **N8** | `train_loss` reportado al server incluye el penalty proximal. Curvas FedAvg vs FedProx no son comparables como están. Figuras de paper requerirían disclaimer. | `trainer.py:135`, `client.py:145-147` |
| **N9** | `global_params = [p.detach().clone() for p in ...]` duplica memoria de parámetros. ResNet50 +100MB, InceptionResNet-V2 +223MB. Bloqueador para escalar a backbones grandes (RadImageNet usa ResNet50/InceptionResNet-V2). | `client.py:122` |
| **N10** | `clip_grad_norm_` se aplica a la gradiente combinada (task + proximal). Con mu grande, gradiente del proximal puede dominar y starvar el task. Default `grad_clip_norm: 0.0` evita el problema, pero usuarios que activen clipping con FedProx desvían del paper sin warning. | `trainer.py:116-118, 131-133` |

### Categoría M (pre-existentes, no introducidos por los fixes)

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **M1** | `val_ds` replicado a todos los clientes. La "federated val metric" agregada es matemáticamente idéntica a centralized eval (todos los clientes computan misma métrica sobre mismos datos). Defender "federated AUC" en paper sería metodológicamente débil. | `client.py:233` |
| **M3** | `normalize_mean/std` siguen siendo escalares en `AugmentationConfig`. Para ImageNet RGB transfer y para RadImageNet (stats por canal) es incorrecto. Bloqueador para transfer learning correcto. | `schema.py:163-164`, `transforms.py:37-42` |

### Categoría Tests (debilidades)

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **T1** | `test_partitioning_patient_disjoint`: solo 2 clientes × 8 pacientes (4:1 ratio). Trivialmente sin overlap; no es stress test. | `test_smoke.py:193-224` |
| **T2** | Verifica solo conteo, no set-equality. Implementación que duplique índices y descarte otros pasaría. | `test_smoke.py:216` |
| **T3** | Sin cobertura de rama fallback `patient_ids = None` (cuando todos los pids son None). | (faltante) |
| **T5** | `test_fedprox_proximal_term_applied` usa mu=10 (no realista). Bug donde mu se cap internamente a 1.0 pasaría. | `test_smoke.py:227-257` |
| **T6** | Solo verifica `mu=10 ≠ mu=0`. Bug que siempre aplique FedProx con mu fijo (e.g., 0.0001) pasaría. | `test_smoke.py:248, 252` |
| **T7** | Tolerancia `>1e-6` excesiva para mu=10. Bug que aplique solo fracción del proximal (factor /1000) pasaría. | `test_smoke.py:255` |
| **T13** | Sin test end-to-end del strategy FedProx (`build_strategy("fedprox", proximal_mu=0.5)` + `configure_fit` + `client.fit`). | (faltante) |
| **T14** | Sin test de `_weighted_average` con dicts mixtos (algunos clientes con roc_auc, otros sin). Escenario que C3 produce más frecuentemente. | (faltante) |

---

## Riesgos menores y deuda técnica nueva

### Categoría C1

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **N4** | `np.bincount(labels[grp]).argmax()` asume etiquetas pequeñas. Para num_classes > 2 con label outliers, aloca array enorme. No es defecto hoy (binario), pero limita extensibilidad futura. | `partitioning.py:143-146` |
| **N6** | Doble validación `num_clients > n` y `num_clients > n_patients` cuando `patient_ids` se provee. Redundancia inocua. | `partitioning.py:78, 91` |

### Categoría C2

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **N11** | AdamW (weight_decay default 1e-4) + FedProx (mu default 0.01) aplican dos regularizaciones con orígenes distintos. No es bug; documentación gap. | `trainer.py:106-129`, `optim.py` |
| **N12** | El término proximal recorre `self.model.parameters()` — excluye BN buffers (`running_mean/var`). Coincide con paper FedProx; asimetría no documentada (buffers SÍ se agregan vía FedAvg). | `trainer.py:110-113` |
| **N13** | Recomputación per-batch del proximal sum es O(P). ~10-30% overhead en modelos pequeños. Optimización futura. | `trainer.py:110-113, 125-128` |

### Categoría C3

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **N14** | `loss = float(result.get("loss", 0.0))` retornado por `evaluate()` no se chequea por NaN. Si criterion produce NaN (divergencia), Flower agrega NaN en loss agregado. | `client.py:160, 185` |
| **N15** | Fallback `loss=0.0` (en Evaluator y client) enmascara fallos del Evaluator como "loss perfecto". | `client.py:160`, `evaluator.py:71-86, 94` |
| **N16** | Log a nivel DEBUG cuando se omite métrica NaN. En producción `INFO` no muestra debug. Operador no ve la omisión. | `client.py:177-182` |

### Categoría Tests

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **T4** | Sin test que `num_clients > n_patients` levante ValueError. | (faltante) |
| **T9** | Sin verificación de que `train_loss` con FedProx ≠ task_loss puro. | (faltante) |
| **T10** | `test_evaluate_nan_metric_omitted_not_zero` verifica solo `roc_auc`. Bug que dropee todas las métricas pasaría. | `test_smoke.py:293-295` |
| **T11** | No verifica que métricas no-NaN (accuracy, precision, recall, f1) SÍ están presentes en val monoclase. | (faltante) |
| **T15** | Sin test con `num_workers > 0`. Production force `num_workers: 0`, así que OK ahora; relajación silenciosa sería riesgosa. | (faltante) |

### Categoría M (pre-existentes)

| ID | Resumen | Ubicación |
|----|---------|-----------|
| **M2** | `set_global_seed` en `client_fn` muta estado global del proceso. Con `client_resources.num_cpus < 1.0` Ray podría co-ubicar actors → race. Default 1.0 evita el problema. | `client.py:275` |
| **M4** | Sin métricas de calibración (ECE, Brier, reliability curves). Insuficiente para defensa clínica. | `evaluation/metrics.py` |
| **M5** | Sin CI, sin docker-compose, sin Makefile. Cualquier regresión futura pasará invisible sin CI. | infra |

---

## Riesgos metodológicos persistentes (no técnicos)

Más allá de los bugs concretos, hay limitaciones de **diseño experimental** que afectan defensibilidad:

| ID | Limitación | Implicación |
|----|------------|-------------|
| **DM1** | Val replicado (M1) hace que "federated val" sea redundante con centralized val. | No reportar "federated val AUC" como métrica distinta. Usar centralized eval del server vía `evaluate_fn`. |
| **DM2** | Modelo único (ResNet18 o EfficientNet-B0) no es el SOTA en mamografía. Papers serios comparan ≥3 backbones. | Limitar conclusiones al backbone usado. |
| **DM3** | Solo 2 estrategias funcionales (FedAvg y FedProx); SCAFFOLD y FedBN son NotImplementedError. | No claim "FedProx es el mejor de los esquemas FL" — solo se comparó contra FedAvg. |
| **DM4** | Sin validación cruzada (k-fold) — solo train/val/test fijo con un seed. Aleatoriedad inter-seed no medida. | Reportar resultados como single-seed; idealmente 3-5 seeds para significancia. |
| **DM5** | Sin métricas clínicas (ECE, Brier, decision curve analysis, NLP de reportes). | Resultados como prueba de concepto técnica, no claim clínico. |
| **DM6** | Mammo-Bench server YAML usa synthetic con 8 muestras para "centralized eval". La métrica reportada por el server es ruido. | El operador debe ignorar la métrica centralized del server o configurar `data.test_fraction` correctamente. |

---

## Plan de mitigación priorizado en 3 fases

### Fase 1.5 — Cerrar gaps de los fixes C1/C2/C3 (~3-4 horas)

**Objetivo**: cerrar los 4 críticos (N1, N7, T8, T12). Sin esto, los resultados experimentales no son defendibles.

| Tarea | Hallazgo | Esfuerzo | Pre-requisito |
|-------|----------|----------|---------------|
| 1 | Fix N1: aplicar `_stratified_patient_split` al fallback "val from train" en `cbis_ddsm.py:198-205` y `mammo_bench.py:191-205`. Reutilizar la función ya existente. | 1 h | — |
| 2 | Fix N7: envolver el cómputo del proximal term en `with torch.cuda.amp.autocast(enabled=False):` con `.float()` explícito. Cambio quirúrgico de ~6 líneas. | 30 min | — |
| 3 | Test T12: crear fixture CSV con `split: train/test` parcial + dos pacientes con múltiples imágenes; verificar disjuntez post-split. | 45 min | — |
| 4 | Test T8: añadir test con mock de `torch.cuda.amp.autocast` o skip-if-no-cuda, verificando que FedProx + AMP con mu=0.01 produce drift mensurable post-fix. | 45 min | tarea 2 |
| 5 | Bump version a 0.2.0 y crear CHANGELOG.md mencionando N2 (reproducibilidad). | 15 min | — |
| 6 | Fix N3: NaN-safe patient_id check en `client.py:210`. | 15 min | — |

**Salida de Fase 1.5**: resultados FedAvg/FedProx en CPU+FP32 son científicamente defendibles para CBIS-DDSM y Mammo-Bench independientemente del manifest físico.

### Fase 2 — Validez observabilidad y robustez de tests (~3-4 horas)

**Objetivo**: endurecer tests, separar task loss de combined loss, mejorar logging.

| Tarea | Hallazgo | Esfuerzo |
|-------|----------|----------|
| 7 | Fix N8: separar `train_task_loss` y `train_total_loss` en el dict retornado por `train_one_epoch`. Reportar `task_loss` al server por defecto. | 1 h |
| 8 | Fix N14: NaN-safe return de `loss` en `evaluate()`. | 15 min |
| 9 | Fix N16: subir log de omisión NaN a WARNING cuando es métrica clínica (roc_auc, sensitivity, specificity). | 15 min |
| 10 | Endurecer T1/T2: añadir caso 4 clientes × 4 pacientes y set-equality assertion. | 30 min |
| 11 | Endurecer T5/T6/T7: añadir mu=0.01 con tolerancia razonable; añadir identity check mu=0. | 45 min |
| 12 | Test T13: end-to-end FedProx strategy completo. | 45 min |
| 13 | Test T14: `_weighted_average` con dicts mixtos. | 30 min |
| 14 | Fix N5: aplicar retry de `min_per_client` en IID y quantity_skew patient-aware. | 30 min |

**Salida de Fase 2**: tests endurecidos, comparación FedAvg vs FedProx legítima, observabilidad útil en producción.

### Fase 3 — Deuda menor + RadImageNet readiness (~4-6 horas)

**Objetivo**: cerrar deuda residual y habilitar transfer learning.

| Tarea | Hallazgo / Bloqueador | Esfuerzo |
|-------|------------------------|----------|
| 15 | Fix N4: `bincount(..., minlength=num_classes)` o `Counter`. | 15 min |
| 16 | Fix N15: cambiar default loss=NaN en Evaluator; manejar uniforme con N14. | 30 min |
| 17 | Implementar Bloqueador RadImageNet 1-2: extender ModelConfig + WeightLoader abstraction. Ver `RADIMAGENET_READINESS_REPORT.md` sección 3.1-3.2. | 1.5 h |
| 18 | Implementar Bloqueador RadImageNet 3: normalize per-canal con presets. | 30 min |
| 19 | Implementar Bloqueador RadImageNet 4: freeze policy + progressive unfreezing. | 1 h |
| 20 | Tests smoke RadImageNet. | 30 min |
| 21 | Docs N11, N12 (interacciones AdamW + FedProx + BN). | 30 min |
| 22 | Setup CI básico (GitHub Actions + pytest + coverage). | 1 h |

**Salida de Fase 3**: sistema preparado para experimentos RadImageNet, deuda técnica residual cerrada, CI activo.

### Fase 4 (opcional, para defensa clínica) — ~1-2 semanas

- Implementar SCAFFOLD y FedBN reales (no stubs).
- Métricas calibración (ECE, Brier).
- k-fold cross-validation infrastructure.
- Múltiples backbones (ResNet50, DenseNet121).
- Docker Compose para reproducibilidad full-stack.
- Pin de CUDA, usuario non-root, hashes en dependencies.

---

## Resumen ejecutivo de riesgos

**Bloqueadores absolutos para resultados publicables (Fase 1.5)**:
- N1: train↔val patient leakage en fallback split.
- N7: FedProx underflow en AMP.
- T8 + T12: tests críticos faltantes.

**Bloqueadores para defensibilidad de FedProx específicamente**:
- N8: train_loss reportado incluye penalty (engañoso en figuras).
- N7 (compartido con bloqueadores absolutos).

**Bloqueadores para RadImageNet**:
- Los 6 bloqueadores estructurales detallados en `RADIMAGENET_READINESS_REPORT.md`.

**Riesgos importantes que NO bloquean pero afectan calidad**:
- N2: reproducibilidad cross-version (mitigar con CHANGELOG).
- N3, N5, N9, N10: edge cases que pueden surgir en producción.
- M1: val replicado (mitigar con `fraction_evaluate: 0.0` + centralized eval).
- Tests débiles (T1, T2, T5, T6, T7, T13, T14): riesgo de regresiones futuras.

**Estimación de esfuerzo total para llegar a "paper-quality results"**:
- Fase 1.5: 3-4 horas (bloqueador).
- Fase 2: 3-4 horas (cualidad de medición).
- Fase 3: 4-6 horas (RadImageNet enable + CI).
- **Total: 10-14 horas** para tener una base científicamente sólida con RadImageNet.

**Recomendación operativa**: no ejecutar experimentos para paper hasta completar Fase 1.5 explícitamente. No usar AMP con FedProx hasta cerrar N7. No usar manifest con `split` parcial hasta cerrar N1.
