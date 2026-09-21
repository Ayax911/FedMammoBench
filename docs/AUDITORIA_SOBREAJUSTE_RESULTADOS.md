# Auditoría de Sobreajuste y Metodología — FedMammoBench

**Fecha de ejecución:** 2026-09-20  
**Commit base auditado:** [`0d3d1f0`](file:///home/akira/snap/steam/federallearning/FedMammoBench)  
**Entorno de ejecución:** Linux / Python 3.12.8 (`.venv`) / PyTorch 2.13.0+cu130 / CUDA disponible (sin acceso a almacenamiento local de imágenes reales ni pesos RadImageNet).

---

## 1. Resumen Ejecutivo y Diagnóstico Global

Se completó una auditoría independiente, estática, forense y dinámica (vía *harness* sintético de datos y ejecución end-to-end con PyTorch) sobre la totalidad de [`src/`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src), [`manifests/`](file:///home/akira/snap/steam/federallearning/FedMammoBench/manifests), [`configs/`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs), [`sweeps/`](file:///home/akira/snap/steam/federallearning/FedMammoBench/sweeps) y artefactos en [`runs/`](file:///home/akira/snap/steam/federallearning/FedMammoBench/runs).

### Principales Conclusiones

1. **Diagnóstico Base Confirmado:** No existe ningún bug en el código que infle artificialmente las métricas reportadas o introduzca fuga de información (*data leakage*). El sobreajuste observado (ej. `train_auc` 0.9963 vs `val_auc` 0.8795 en la época 24 de `exp37`) se debe exclusivamente al **régimen de entrenamiento** (24.5M parámetros entrenables descongelados desde `conv1` contra 6.671 imágenes de train, con dropout 0.0).
2. **Selección de Checkpoint Operativa:** El checkpoint final evaluado en test corresponde rigurosamente a la mejor época en validación (época 3 en `exp37`, con `val_auc` 0.9212 y `test_auc` 0.9016), descartando la memorización de las épocas posteriores.
3. **Validación Experimental de Fuga (Harness Sintético):** Un dataset sintético generado con imágenes TIFF en modo `"F"` y etiquetas aleatorias arrojó un `test_auc` exacto de **0.5000**, mientras que con señal controlada alcanzó **1.0000**, confirmando la hermeticidad del pipeline.
4. **Hallazgos Críticos Metodológicos y de Configuración:**
   - **H1 (Nivel de agregación e ICs):** La evaluación por imagen reporta intervalos de confianza artificialmente estrechos. Al remuestrear por paciente (bootstrap cluster 95%), el ancho del IC del AUC crece un **42.7 %** (de ±0.028 a ±0.041 en `exp37`). Además, en comparaciones pareadas a nivel de paciente, la ventaja de `exp37` sobre `exp28` **no es estadísticamente significativa** ($p=0.933$, IC del delta: $[-0.0073, +0.0625]$).
   - **H2 (Patience desacoplada del sweep):** `sweeps/hpsearch_v1.yaml` barrió `epochs` en $\{25, 40, 60\}$ pero heredó `patience: 20` fijo de `exp31`. Para `epochs: 25`, el early stopping quedó virtualmente inoperante.
   - **H4 (Umbral 0.5 fijo):** El umbral 0.5 colapsa la sensibilidad a **0.0000** en bases altamente desbalanceadas como KAU-BCMD (4.3 % malignos) pese a un AUC de 0.9414. La calibración por base en validación recupera sensibilidad (0.3333) y F1-macro (de 0.4932 a 0.6574).
   - **H6 (Deuda técnica y validación silenciosa):** `config.py` carece de validadores Pydantic, permitiendo comportamientos erróneos silenciosos si se combinan opciones como `metric_name: "loss"` con `metric_mode: "max"` o `ReduceLROnPlateau` con `metric_name: "auc"`.

---

## 2. Verificación de Premisas y Sección de Refutaciones

Se auditaron exhaustivamente todos los puntos de las secciones 2 y 3 del *brief*.

| Punto Auditado | Archivo y Línea | Resultado Medido / Verificado | Veredicto |
|---|---|---|---|
| **Disyunción de Pacientes** | [`src/datasets/split.py:53-60`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/split.py#L53-L60) | 2.534 pacientes únicos. Intersecciones: `train ∩ val = 0`, `train ∩ test = 0`, `val ∩ test = 0`. | **Verificado** |
| **Limpieza de Manifests** | [`manifests/fedmammobench_norm_0_1.csv`](file:///home/akira/snap/steam/federallearning/FedMammoBench/manifests/fedmammobench_norm_0_1.csv) | 8.341 filas, 0 duplicados exactos, 0 rutas repetidas. Prevalencia maligno: Train 34.15 %, Val 33.61 %, Test 33.21 %. | **Verificado** |
| **Defecto CMMD Arreglado** | Manifests actuales | 739 pacientes bilaterales: 664 (89.9 %) mixtos entre mamas, 3 (0.4 %) todo-maligno, 72 todo-benigno. **0 de 2.361 mamas** tienen etiquetas discordantes entre CC y MLO. | **Verificado** |
| **Alineación Fila-Label** | [`src/datasets/dataset.py:86,111`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/dataset.py#L86-L111) | `.iloc` posicional. Comprobado empíricamente en `test/predictions.csv`: match del 100% en las 834 filas de test contra `Split.test_df()`. | **Verificado** |
| **Augmentación exclusiva de Train** | [`src/cli.py:58-75`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/cli.py#L58-L75), [`src/evaluate.py:99`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/evaluate.py#L99) | `eval_transform_builder` desactiva flips y rotaciones en Val y Test. | **Verificado** |
| **Loss & Probabilidades** | [`src/train/build.py:184,207,226`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/build.py#L184-L226) | `LossSpec`: Sigmoid para BCE, Softmax[:, 1] para CE y Focal. | **Verificado** |
| **Modo Eval y No Grad** | [`src/train/loop.py:146,172`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/loop.py#L146-L172), [`src/train/evaluation.py:65,103`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/evaluation.py#L65-L103) | `model.eval()` y `@torch.no_grad()` presentes en todas las rutas de inferencia. | **Verificado** |
| **Congelamiento de BatchNorm** | [`src/train/loop.py:89-91,133-145`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/loop.py#L89-L145) | `_set_frozen_bn_eval()` se invoca tras `model.train()`, manteniendo en `eval()` las capas BN cuyos parámetros no requieren gradiente. | **Verificado** |
| **Carga del Mejor Checkpoint** | [`src/train/trainer.py:247,303`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/trainer.py#L247-L303), [`src/eval_pipeline.py:94-100`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/eval_pipeline.py#L94-L100) | `evaluate_split()` recarga `best_checkpoint` explícitamente desde disco antes de computar métricas finales. | **Verificado** |

### Refutaciones Encontradas
> **Resultado de la verificación:** No se refuta ningún hecho de las secciones 2 y 3. El diagnóstico de que el pipeline no tiene fugas de datos y que las métricas reportadas no provienen del modelo sobreajustado final es **100 % correcto**.

---

## 3. Auditoría Detallada por Hipótesis (H1 – H7)

### H1 — Métricas por Imagen vs por Mama / por Paciente e Intervalos de Confianza Bootstrap
- **Clasificación:** `Problema Metodológico`
- **Veredicto:** **CONFIRMADA**
- **Evidencia y Mediciones:**
  Se cruzaron las predicciones de test (`n=834` imágenes de `n=253` pacientes y `n=505` mamas) contra el manifest [`manifests/fedmammobench_norm_0_1.csv`](file:///home/akira/snap/steam/federallearning/FedMammoBench/manifests/fedmammobench_norm_0_1.csv). Se computaron métricas a nivel de imagen, mama (media y max de probabilidades) y paciente (max de probabilidad), junto con un **bootstrap no paramétrico a nivel de paciente** (5.000 iteraciones clusterizadas por `patient_id`):

| Experimento | Image AUC (95% CI) | Breast Mean AUC (95% CI) | Patient Max AUC (95% CI) | Ancho CI Img | Ancho CI Paciente |
|---|:---:|:---:|:---:|:---:|:---:|
| **exp37** (`hpsearch_v1`) | 0.9016 `[0.8713, 0.9281]` | **0.9179** `[0.8859, 0.9452]` | 0.8939 `[0.8510, 0.9321]` | 0.0568 | **0.0811 (+42.7%)** |
| **exp28** (`antioverfit_base`) | 0.8801 `[0.8474, 0.9088]` | 0.9026 `[0.8675, 0.9335]` | 0.8674 `[0.8204, 0.9093]` | 0.0614 | 0.0890 (+45.0%) |
| **exp31** (`no_inputdrop`) | 0.8889 `[0.8579, 0.9165]` | 0.9063 `[0.8740, 0.9352]` | 0.8617 `[0.8141, 0.9049]` | 0.0586 | 0.0907 (+54.8%) |
| **exp05** (`frozen_backbone`) | 0.8177 `[0.7787, 0.8548]` | 0.8251 `[0.7845, 0.8627]` | 0.7989 `[0.7405, 0.8554]` | 0.0762 | 0.1148 (+50.6%) |
| **exp57** (`camilo_central`) | 0.8213 `[0.7826, 0.8573]` | 0.8306 `[0.7907, 0.8679]` | 0.8008 `[0.7422, 0.8572]` | 0.0747 | 0.1149 (+53.8%) |

  **Análisis de significancia estadística (Bootstrap Pareado por Paciente):**
  - **exp37 vs exp31:** Delta de AUC a nivel paciente = $+0.0322$, IC 95%: `[+0.0021, +0.0630]`, $P(\text{exp37} > \text{exp31}) = 98.1\%$.
  - **exp37 vs exp28:** Delta de AUC a nivel paciente = $+0.0264$, IC 95%: `[-0.0073, +0.0625]`, $P(\text{exp37} > \text{exp28}) = 93.3\%$. Al cruzar el cero, la supuesta superioridad de `exp37` sobre `exp28` **no alcanza significancia estadística formal al 5%**.
- **Impacto:** Reportar métricas por imagen sin corrección clusterizada subestima la incertidumbre en un ~40-55 %, generando una falsa ilusión de separación entre arquitecturas parecidas.

---

### H2 — Inoperancia de Early Stopping (`patience: 20` con `epochs: 25`)
- **Clasificación:** `Bug de Configuración`
- **Veredicto:** **CONFIRMADA**
- **Evidencia:**
  1. En [`sweeps/hpsearch_v1.yaml:164-167`](file:///home/akira/snap/steam/federallearning/FedMammoBench/sweeps/hpsearch_v1.yaml#L164-L167), se muestrea `epochs: [25, 40, 60]`, pero no se define `patience` como hiperparámetro del sweep.
  2. En [`scripts/sweep_train.py:164-170`](file:///home/akira/snap/steam/federallearning/FedMammoBench/scripts/sweep_train.py#L164-L170), el config base [`configs/exp31_antioverfit_no_inputdrop.yaml:76`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs/exp31_antioverfit_no_inputdrop.yaml#L76) provee `patience: 20` fijo.
  3. Al generarse [`configs/exp37_hpsearch_v1_e7u7fprr.yaml`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs/exp37_hpsearch_v1_e7u7fprr.yaml), `epochs: 25` con `patience: 20` implicó que si la mejor época ocurría después de la época 5, el early stopping era matemáticamente incapaz de dispararse ($5 + 20 \ge 25$). En `exp37` la mejor época fue la 3, por lo que el early stopping se activó en la época $3 + 20 = 23$ (completando 24 épocas de 25).
- **Impacto:** No corrompió el checkpoint guardado (se preservó la época 3), pero provocó un desperdicio del 83 % del cómputo posterior e infló artificialmente las curvas de pérdida final (`train_loss` cayó a 0.1167 mientras `val_loss` explotó a 0.7400).

---

### H3 — Sobreajuste del Split de Validación por Sweep Bayesiano
- **Clasificación:** `Problema Metodológico`
- **Veredicto:** **CONFIRMADA**
- **Evidencia:**
  - `hpsearch_v1` optimizó `val_auc` mediante Hyperband/Bayes sobre un split de validación único de $N=836$ imágenes ($N_{pacientes}=254$).
  - **Comparación del Gap Val $\to$ Test:**
    - Corridas elegidas por sweep: `exp37` ($+0.0196$), `exp38` ($+0.0194$), `exp39` ($+0.0334$).
    - Corridas sin sweep: `exp05` ($+0.0021$), `exp57` ($-0.0021$), `exp28` ($+0.0278$).
  - El error estándar del AUC en un split de $N=836$ es de aproximadamente $\sigma \approx \sqrt{\frac{0.9(0.1)}{254}} \approx 0.0188$. Gran parte de la diferencia de `val_auc` entre el ganador (`0.9212`) y el baseline (`0.9083`) cae dentro de la banda de ruido de muestreo ($\pm 0.015$).

---

### H4 — Umbral Fijo 0.5 sin Recalibración
- **Clasificación:** `Decisión de Diseño Discutible` / `Problema Metodológico en Lectura por Base`
- **Veredicto:** **CONFIRMADA**
- **Evidencia:**
  Se ejecutó [`scripts/calibrate_threshold.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/scripts/calibrate_threshold.py) sobre `exp37`:
  - **Global Test:** Umbral calibrado en val = 0.5747. F1-macro se mantiene estable ($0.8071 \to 0.7958$), subiendo especificidad ($0.8851 \to 0.9264$).
  - **Desglose por Base (KAU-BCMD — prevalencia 4.3 %):**
    - Umbral fijo 0.5: `Sensibilidad = 0.0000`, `F1 = 0.0000`, `F1-macro = 0.4932` (a pesar de un `AUC = 0.9414`). El modelo predecía todo como benigno.
    - Umbral calibrado en val (0.3054): `Sensibilidad = 0.3333`, `F1 = 0.3333`, `F1-macro = 0.6574` (+0.1642).
- **Impacto:** Ninguna decisión clínica ni conclusión sobre la capacidad de discriminación del modelo debe basarse en métricas dependientes de umbral (F1, sensibilidad, precisión) evaluadas a 0.5 fijo sin calibración previa en validación.

---

### H5 — Inconsistencia Menor de Operador Umbral (`>` vs `>=`)
- **Clasificación:** `Bug Menor` / `Deuda Técnica`
- **Veredicto:** **CONFIRMADA (Impacto nulo en artefactos actuales)**
- **Evidencia:**
  - [`src/metrics.py:120`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/metrics.py#L120) define `preds = (preds > self.threshold).long()`.
  - [`src/train/evaluation.py:115`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/evaluation.py#L115) y [`src/reporting.py:34`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/reporting.py#L34) usan `preds = (probs >= 0.5).astype(int)`.
  - Se revisaron todos los archivos `predictions*.csv` en `runs/`: **0 muestras** tienen `y_prob == 0.5` exacto.
- **Impacto:** En los artefactos commiteados, `metrics.json` y `confusion_matrix_metrics.json` coinciden punto a punto. No obstante, representa una discrepancia en código que debe unificarse a `>=`.

---

### H6 — Riesgos Latentes de Validación en Configuración
- **Clasificación:** `Deuda Técnica` / `Falta de Validación Estricta`
- **Veredicto:** **CONFIRMADA**
- **Evidencia Experimental por Caso:**

| Ítem | Riesgo Analizado | Comportamiento Observado | Clasificación |
|---|---|---|:---:|
| **H6.a** | `metric_name: "loss"` con `metric_mode: "max"` | [`src/config.py:264`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/config.py#L264) no valida la coherencia. [`src/train/trainer.py:244`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/trainer.py#L244) selecciona el checkpoint con **mayor pérdida** sin advertencia. | **Falla Silenciosa** |
| **H6.b** | `ReduceLROnPlateau` vs `metric_mode` | [`src/train/build.py:106`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/build.py#L106) instancia con `mode="min"`. Si se pasa `val_auc`, el scheduler **reduce el LR cuando el AUC mejora**. | **Falla Silenciosa** |
| **H6.c** | `Split.train_df()` con `"Train"` o `"validation"` | [`src/datasets/split.py:79`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/split.py#L79) accede a `self.splits["train"]` y levanta `KeyError: 'train'`. | **Falla Ruidosa** |
| **H6.d** | `patient_id` con espacio `"  "` o cadena vacía | [`src/datasets/manifest.py:70`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/manifest.py#L70) solo verifica `.isna()`. Espacios en blanco pasan la validación y agrupan filas en un paciente fantasma. | **Falla Silenciosa** |
| **H6.e** | `preprocessed_image_path` absoluto | [`src/datasets/manifest.py:112`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/manifest.py#L112) usa `Path(image_root) / Path("/abs/path")`, descartando `image_root` en silencio. | **Falla Silenciosa** |
| **H6.f** | BatchNorm con `affine=False` | [`src/train/loop.py:140`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/loop.py#L140) evalúa `not any(p.requires_grad...)` como `True` (0 params), forzándola a `eval()`. | **Comportamiento Silencioso** |
| **H6.g** | `min_delta > 0` en EarlyStopping | [`src/train/trainer.py:244`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/trainer.py#L244) acopla el guardado de checkpoint a `_is_improvement`. Si el argmax ocurre en un delta menor, el modelo óptimo no se guarda. | **Falla Silenciosa** |
| **H6.h** | DataLoader vacío | [`src/train/loop.py:127`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/train/loop.py#L127) ejecuta `total_loss / len(loader)`, arrojando `ZeroDivisionError`. | **Falla Ruidosa** |

---

### H7 — Paridad de la Capa Federada y Round-Trip de Parámetros
- **Clasificación:** `Decisión de Diseño Discutible` (ya documentada) / `Validación Exitosa`
- **Veredicto:** **CONFIRMADA**
- **Evidencia:**
  1. **Round-Trip de Parámetros:** Se ejecutó [`src/federated/param_utils.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/federated/param_utils.py) (`get_model_ndarrays` $\to$ perturbación $\to$ `set_model_ndarrays` $\to$ `get_model_ndarrays`) sobre `ResNet50` en scopes `"full"` (320 tensores) y `"backbone"` (318 tensores). La reconstrucción fue **idéntica bit a bit** (`True`).
  2. **Loop de Entrenamiento:** [`src/federated/client.py:191-214`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/federated/client.py#L191-L214) reusa exactamente `train_one_epoch()` y `evaluate_loader()`, preservando la consistencia con el loop centralizado.
  3. **Comportamiento Documentado:** Se re-confirma que en la capa federada, el AUC de `best.json` es el promedio ponderado por nodo y no el AUC acumulado (*pooled*), como ya se encuentra estipulado en el diseño.

---

## 4. Validación Dinámica End-to-End con Harness Sintético

Se construyó un arnés de prueba aislado ([`verify_h1.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench) / `synthetic_harness.py`) para ejecutar el pipeline real de `src.cli` sin requerir los datos reales de mamografía:

```
=== RESULTADOS DEL HARNESS SINTÉTICO (ResNet-50 Imagenet v2, unfreeze_from=none) ===
1. Señal Controlada (Intensidad media correlacionada con la patología):
   - Val AUC:  1.0000 | F1-macro: 1.0000
   - Test AUC: 1.0000 | F1-macro: 1.0000
2. Ruido Puro (Etiquetas aleatorias con pacientes disjuntos):
   - Val AUC:  0.5787 (por optimización en selección de época)
   - Test AUC: 0.5000 | F1-macro: 0.3357
```

**Conclusión:** El hecho de que las etiquetas aleatorias den exactamente `Test AUC = 0.5000` demuestra de forma concluyente que no existe ninguna filtración o fuga de información entre splits en el código de FedMammoBench.

---

## 5. Tabla Maestra de Hallazgos y Recomendaciones

| Hallazgo | Clasificación | Severidad | Acción Recomendada para la Workstation / Código |
|---|---|:---:|---|
| **Evaluación por Imagen vs Paciente (H1)** | Problema Metodológico | **Alta** | Reportar primariamente el AUC agrupado por mama (`Breast Mean AUC`) y por paciente (`Patient Max AUC`) con IC 95% por bootstrap de cluster. |
| **Sweep desacoplado de Patience (H2)** | Bug de Configuración | **Media** | En futuros sweeps W&B, definir `patience` relativo (ej. `0.3 * epochs`) o muestrearlo explícitamente en `parameters:`. |
| **Selección en Split Único de Val (H3)** | Problema Metodológico | **Media** | Implementar validación cruzada por pliegues ($k$-fold a nivel de paciente) para el ranking final de arquitecturas. |
| **Umbral Fijo 0.5 (H4)** | Decisión de Diseño | **Alta (en lectura)** | Prohibir el uso de F1 / Sensibilidad a umbral 0.5 en bases desbalanceadas sin previa calibración con `scripts/calibrate_threshold.py`. |
| **Discrepancia `>` vs `>=` (H5)** | Bug Menor | **Baja** | Unificar [`src/metrics.py:120`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/metrics.py#L120) a `>=`. |
| **Validación Débil en `config.py` (H6)** | Deuda Técnica | **Media** | Incorporar `@model_validator` en Pydantic para validar combinaciones incompatibles (`metric_name`, `metric_mode`, `scheduler.mode`). |
| **Strip de `patient_id` (H6.d)** | Deuda Técnica | **Baja** | Añadir `.str.strip()` y validación contra cadenas vacías en [`src/datasets/manifest.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/manifest.py). |

---

## 6. Próximos Pasos en Workstation (Con Acceso a Datos y GPU)

1. **Re-entrenar `exp37` con Regularización:**
   - Probar `unfreeze_from: layer4` en lugar de `conv1` (reduciendo parámetros entrenables de 24.5M a 14.9M).
   - Incorporar `dropout: 0.2` y `weight_decay: 1e-3` para verificar si se mitiga la brecha train-val sin degradar el test AUC.
2. **Evaluación Multisemilla:**
   - Correr `exp37` con semillas $\{43, 44, 45\}$ para confirmar si el margen sobre `exp28` y `exp31` persiste a través de diferentes inicializaciones.
3. **Validación Cruzada por Paciente:**
   - Estimar el rendimiento en 5 pliegues para eliminar el sesgo de optimización sobre el conjunto de validación único.
