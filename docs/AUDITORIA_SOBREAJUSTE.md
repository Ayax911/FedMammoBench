# Auditoría de sobreajuste — brief para Antigravity

> **Documento de trabajo para un agente externo.** Encarga una auditoría completa de `src/` con el fin
> de encontrar bugs o defectos metodológicos que estén haciendo que los resultados de FedMammoBench no
> sean apropiados, con foco en el sobreajuste observado.
>
> Autocontenido: se puede trabajar leyendo solo este archivo y el repositorio.
> Fecha: 2026-09-20. Commit base: `0d3d1f0`.

---

## 0. Resumen ejecutivo — lee esto antes que nada

Antes de redactar este brief se ejecutaron **tres auditorías independientes** (pipeline de datos, loop
de entrenamiento/evaluación, y forense de los artefactos de `runs/`). El resultado reorienta la
pregunta original:

> **No se encontró ningún bug en el código que infle las métricas reportadas.** El sobreajuste es real
> y está medido, pero su causa es el **régimen de entrenamiento**, no un defecto de implementación. Y
> las métricas que se reportan **no provienen del modelo sobreajustado**: el checkpoint que se evalúa
> es el de la mejor época, y el mecanismo de selección funciona correctamente.

Por eso esta auditoría tiene **dos objetivos**, no uno:

1. **Verificar de forma independiente que ese diagnóstico es correcto.** Contradice la sospecha
   inicial, así que merece un segundo par de ojos. Refutar cualquier punto de la sección 2 con
   evidencia es un resultado válido y deseable.
2. **Atacar lo que sí queda abierto**, que es de naturaleza metodológica y sí afecta a si los
   resultados "están dando apropiadamente" (sección 4, H1–H7).

---

## 1. Restricción de máquina (verificada)

La auditoría corre en un laptop **sin acceso a los datos**:

| recurso | estado |
|---|---|
| `/media/imagenesmedicas`, `/home/labmirp` (imágenes) | **NO existen** |
| pesos RadImageNet (`weights/ResNet50.pt`) | **NO existe** |
| pesos ImageNet de torchvision | **cacheados** en `~/.cache/torch/hub/checkpoints/` |
| GPU CUDA | **disponible** |
| `manifests/*.csv` (8.341 filas, 21 columnas) | legibles |
| `runs/*/` (`metrics.csv`, `predictions.csv`, `metrics.json`) | legibles |

**No se puede reproducir ninguna corrida real.** Sí se puede:

- auditar el código estáticamente;
- hacer forense del manifest y de los artefactos ya generados;
- **ejercitar el pipeline end-to-end con datos sintéticos** (sección 5).

Dos hechos verificados que habilitan lo último:

```python
build_model('resnet50_imagenet_v2', unfreeze_from='none', device='cpu')
# -> LoadReport(matched=318, missing=[], unexpected=[])   # funciona offline

Image.fromarray(arr_float32, mode='F')  # + ToTensor() conserva el rango [-1,1], sin dividir por 255
```

Usa siempre `.venv/bin/python` (Python 3.12.8). No hay paquete instalado ni paso de activación.

---

## 2. Lo que YA está descartado — no repetir este trabajo

Cada punto está verificado con archivo y línea. Re-verificar es legítimo; re-descubrir desde cero es
desperdiciar el presupuesto.

### 2.1 Datos — limpios, sin fuga

Sobre `manifests/fedmammobench_norm_neg1_1.csv` (8.341 filas):

- **0** filas duplicadas exactas; **0** `preprocessed_image_path` repetidos; **0** `ID_image` duplicados.
- **0 pacientes aparecen en más de un split** (2.534 pacientes únicos, dtype `str`, sin NaN ni vacíos).
- **0 colisiones de `patient_id` entre bases** — los IDs llevan sufijo (`_CM`, `_K`, `_I`, `_CD`).
- Prevalencia de maligno estratificada: train 34,1 % / val 33,6 % / test 33,2 %.
- 3,29 imágenes por paciente, **idéntico en los tres splits** (train 6.671/2.027, val 836/254,
  test 834/253). Máximo 16.
- `classification` solo contiene `Benign` (5.505) y `Malignant` (2.836).

En el código:

| qué | dónde | veredicto |
|---|---|---|
| `Split` verifica disyunción de pacientes y **lanza excepción** | `src/datasets/split.py:53-60` | correcto |
| `patient_id` nulo bloqueado antes del `groupby` | `src/datasets/manifest.py:70-77` | correcto |
| `.iloc` posicional; imagen y label de la **misma** `row` | `src/datasets/dataset.py:86,111` | correcto |
| Augmentación exclusiva de train, en los 3 call sites | `build.py:58-60`; `cli.py:74`, `evaluate.py:99`, `federated/assembly.py:108` | correcto |
| Etiqueta desconocida/NaN lanza `ValueError`, **no** cae a benigno | `src/datasets/manifest.py:90-93` | correcto |
| `shuffle`/`drop_last` solo en train; val/test no descartan muestras | `src/datasets/build.py:67,76,81,82` | correcto |

Grep exhaustivo de `.loc[` / `.at[` / `.ix[` en todo `src/`: **un solo hit**, en `manifest.py:92`, dentro
de un mensaje de error. La desalineación imagen-etiqueta es imposible por construcción.

### 2.2 Entrenamiento y evaluación — correctos

| qué | dónde | veredicto |
|---|---|---|
| `LossSpec`: sigmoid para BCE, `softmax[:,1]` para CE y Focal | `src/train/build.py:184,207,226` | correcto en los 3 |
| `model.eval()` + `@torch.no_grad()` en toda ruta de evaluación | `loop.py:146,172`; `evaluation.py:65,103` | correcto |
| `_set_frozen_bn_eval()` inmediatamente después de `model.train()` | `src/train/loop.py:89-91` | correcto |
| `EarlyStopping` simétrico max/min, sin bug de signo | `src/train/early_stopping.py:65,87-91` | correcto |
| `fit()` devuelve el **mejor** checkpoint, no el último | `src/train/trainer.py:247,303` | correcto |
| Misma métrica para early stopping y selección de checkpoint | `src/train/trainer.py:230,244,252` | correcto |
| `scheduler.step()` una vez por época, fuera del loop de batches | `src/train/trainer.py:239-242` | correcto |
| `BinaryMacroF1Score` es F1 macro real (media de F1 de ambas clases) | `src/metrics.py:123-142` | correcto |
| Métricas instanciadas de cero por época y por split (sin estado) | `src/train/loop.py:93,173` | correcto |
| Checkpoint cargado antes de evaluar; val y test con el mismo modelo | `evaluation.py:61-62`; `cli.py:213-224` | correcto |
| Class weights como tensor en el device correcto | `src/train/build.py:179,202,221` | correcto — el bug del `weight`-como-lista de `PHASES.md` está resuelto |
| `zero_grad()` por batch, sin acumulación involuntaria | `src/train/loop.py:101` | correcto |

### 2.3 El defecto histórico de CMMD está ARREGLADO

`docs/EXPERIMENTOS_CENTRALIZADOS.md` §6 ("El techo de 0.44", líneas 166-231) documenta que CMMD
etiquetaba **por paciente y no por mama**: 826 pacientes bilaterales, 751 todo-maligno, **0 mixtos** —
cuando el cáncer bilateral sincrónico real es del 1-3 %. Eso ponía ~1.502 imágenes (17 %) de mamas
contralaterales sanas etiquetadas como `Malignant`, y un piso de val-loss irreducible de ~0,44.

**Verificado sobre los manifests actuales — el defecto ya no está:**

| | actual | documentado (defectuoso) |
|---|---:|---:|
| CMMD pacientes bilaterales | 739 | 826 |
| …con etiquetas **mixtas** entre mamas | **664 (89,9 %)** | 0 |
| …todo-maligno | **3 (0,4 %)** | 751 |

Y **0 de 2.361 mamas** tienen etiquetas mixtas entre sus vistas CC/MLO — que es exactamente lo correcto
(el tumor es unilateral; la contralateral sana va como benigna, y las dos vistas de una misma mama
comparten etiqueta). El piso está roto: exp37 alcanza **val_loss 0,3786**.

---

## 3. El diagnóstico que hay que verificar

### 3.1 El sobreajuste es real y está medido

`runs/exp37_hpsearch_v1_e7u7fprr/metrics.csv` — 24 épocas:

| época | train_loss | val_loss | train_auc | val_auc |
|---|---:|---:|---:|---:|
| 0 | 0,5402 | 0,5109 | 0,8055 | 0,8565 |
| **3 — mejor val_auc, es el checkpoint reportado** | 0,4355 | 0,4044 | 0,8849 | **0,9212** |
| 18 | 0,1497 | 0,6512 | 0,9922 | 0,8882 |
| 23 — final | **0,1167** | **0,7400** | **0,9963** | 0,8795 |

`val_loss` se multiplica por 1,8 mientras `train_loss` cae 3,7×. `train_auc` 0,9963 = memorización casi
total del train. Sobreajuste de manual a partir de la época ~4.

### 3.2 Su causa es el régimen de entrenamiento, no un bug

| config | `unfreeze_from` | params entrenables | dropout | épocas | gap de loss final |
|---|---|---:|---:|---:|---:|
| **exp37** (el promovido) | `conv1` = **todo** | **24.558.146** | **0,0** | 25 | **+0,623** |
| exp28 | `conv1` = todo | 23.512.130 | 0,0 | 80 | **+0,894** |
| exp05 | `none` (congelado) | 2.624.002 | 0,2 | 200 | **+0,042** |

Medido con `build_model`:

```
unfreeze_from=none     entrenables=           0 / 23.508.032  (0,0 %)
unfreeze_from=layer4   entrenables=  14.964.736 / 23.508.032  (63,7 %)
unfreeze_from=conv1    entrenables=  23.508.032 / 23.508.032  (100,0 %)
```

**exp37 entrena 24,5M parámetros con 6.671 imágenes de train: ratio 3.681:1**, con dropout 0,0 y
weight_decay 3,4e-4. exp05, sobre el mismo dataset pero con el backbone congelado, **no sobreajusta**
(si acaso subajusta: train_auc 0,838 tras 180 épocas). El contraste aísla la causa limpiamente.

### 3.3 Pero las métricas reportadas no están infladas por eso

- **Gap val→test de exp37: +0,0196 de AUC.** Máximo +0,0425 en 34 corridas. Es la magnitud esperable
  por selección de checkpoint sobre val con n=836. No hay señal de sesgo grosero.
- **Calibración razonable**: ECE 0,070 en test, 0,100 en val. Curva de fiabilidad de test casi monótona
  y cercana a la diagonal.
- **Probabilidades no saturadas**: en test, mediana 0,336, rango completo `[0,0002 – 0,9936]`, 18,2 % de
  las predicciones en la banda 0,4–0,6. Separación por clase: 0,250 en benignas vs 0,651 en malignas.
- El early stopping **funciona**: las tres corridas revisadas paran en `mejor_época + patience` y el
  checkpoint que se evalúa es el de la mejor época, no el del final.

---

## 4. Hipótesis priorizadas

Ordenadas por **probabilidad × impacto sobre la validez de los resultados**. Todas verificables **sin
datos reales**.

### H1 — Las métricas son por imagen, no por paciente ni por mama · ALTA / ALTA

**La hipótesis más probable de "los resultados no están dando apropiadamente", y ninguna auditoría
previa la cubrió.**

Con 3,29 imágenes por paciente (CC+MLO de cada mama; CDD-CESM además con adquisiciones DM/CESM de la
misma vista), las predicciones **están correlacionadas dentro de paciente**. Un AUC calculado sobre 834
imágenes que provienen de 253 pacientes tiene el poder estadístico de ~253 observaciones, no de 834 —
y está inflado respecto a una métrica por lesión, que es la unidad clínicamente relevante.

**Verificar:**
- Recalcular AUC y f1_macro de `runs/exp37_hpsearch_v1_e7u7fprr/test/predictions.csv` agregando por
  `(patient_id, laterality)` — probar promedio y máximo de `y_prob` por mama — cruzando contra
  `manifests/fedmammobench_norm_0_1.csv`.
- Calcular un **IC 95 % por bootstrap a nivel de paciente** (remuestrear pacientes, no imágenes).
- Repetirlo para exp28 y exp31 para poder comparar corridas.

**Aceptación:** diferencia cuantificada entre AUC-por-imagen y AUC-por-mama, y ancho del IC. Si los IC
por paciente de exp37, exp31 y exp28 se solapan, buena parte del ranking de experimentos del repo no es
estadísticamente significativa.

**Aviso de implementación:** el `predictions.csv` agregado **no trae `patient_id`** (columnas:
`y_true, y_pred, y_prob`). Hay que unirlo por orden de fila contra el `test_df()` reconstruido, o usar
los `predictions_<db>.csv`. **Verifica que el join es correcto antes de concluir nada** — un join mal
hecho aquí produce un resultado espectacular y falso.

### H2 — `patience: 20` con `epochs: 25` deja el early stopping inoperante · CONFIRMADA / MEDIA

En `configs/exp37_hpsearch_v1_e7u7fprr.yaml`, el early stopping no puede dispararse antes de la época 20
de 25. **Es un bug de configuración real**, probablemente propagado desde el espacio de búsqueda del
sweep. No corrompe el checkpoint reportado (se guarda el de la época 3), pero desperdicia ~83 % del
cómputo y hace que las curvas parezcan mucho peores que el modelo realmente entregado.

**Verificar:** revisar `sweeps/hpsearch_v1.yaml` y todos los `configs/*.yaml` buscando
`patience >= epochs`. Confirmar en cada `metrics.csv` que la corrida paró en `mejor_época + patience`.

**Aceptación:** lista de configs afectados, y si el sweep muestreaba `patience` independientemente de
`epochs` (lo que reproduciría el problema en todo sweep futuro).

### H3 — El sweep bayesiano sobreajustó el split de val · ALTA / MEDIA

exp37 se eligió maximizando `val_auc` sobre un presupuesto de ~300 trials contra un **único** split de
val de 836 imágenes (~253 pacientes). Eso es selección de modelo repetida cientos de veces sobre la
misma muestra: el `val_auc` 0,9212 es optimista por construcción, y parte del margen de exp37 sobre los
demás puede ser ruido del split.

**Verificar:** reconstruir la distribución de `val_auc` de los trials desde `sweeps/` y `runs/*/wandb/`
si está disponible. Estimar cuánto del margen de exp37 sobre el segundo mejor cabe dentro del ruido de
muestreo.

**Aceptación:** comparar el gap val→test de exp37 (+0,0196) contra el de corridas **no** seleccionadas
por sweep (exp05: +0,0021; exp57: −0,0021). Si el gap escala con el número de trials, confirmada.

### H4 — Umbral fijo 0,5 sin recalibrar · CONFIRMADA / ALTA en la lectura por base

El gap de `f1_macro` es **sistemáticamente mayor** que el gap de AUC (exp23: 0,072 vs 0,042), señal de
que el umbral está ajustado a val y no transfiere. Y en kau-bcmd (4,3 % de malignos) el 0,5 fijo da
**sensibilidad 0,0000 con AUC 0,9414**: en la base donde el número ingenuo se ve mejor, el modelo es de
hecho inútil como clasificador.

**Verificar:** ejecutar `scripts/calibrate_threshold.py` (lee un `run_dir`, **no necesita imágenes**)
sobre exp37, con y sin `--by-database`; comparar contra los `metrics.json` commiteados.

**Aceptación:** tabla de métricas antes/después de recalibrar, por base. Confirmar que **ninguna
conclusión del repositorio depende de un f1 o una sensibilidad calculados con el 0,5 fijo**.

### H5 — Bug menor confirmado: umbral `>` vs `>=` · CONFIRMADA / BAJA

`src/metrics.py:120` binariza con `preds > threshold`; `src/train/evaluation.py:115` usa `probs >= 0.5`.
Consecuencia: `metrics.json` y `predictions.csv` / `confusion_matrix_metrics.json` del **mismo**
`run_dir` pueden discrepar en las muestras con `prob == 0.5` exacto.

**Verificar:** contar en los `predictions.csv` cuántas filas tienen `y_prob == 0.5` exacto y reconciliar
la matriz de confusión contra `metrics.json`.

**Aceptación:** número de muestras afectadas. Si es 0, es deuda técnica; si no, esos artefactos no son
estrictamente reconciliables entre sí.

### H6 — Riesgos latentes de validación de config · MEDIA / MEDIA

Ninguno está activo hoy; todos son capaces de arruinar una corrida futura **en silencio**:

| # | riesgo | dónde |
|---|---|---|
| a | **No hay ni un `@field_validator` en todo `config.py`.** `metric_name: "loss"` con el `metric_mode: "max"` por defecto seleccionaría la **peor** época sin error alguno | `src/config.py:264-265` |
| b | El `mode` de `ReduceLROnPlateau` no se sincroniza con `metric_mode` | `src/train/trainer.py:237-242` |
| c | `train/val/test_df()` filtran por `patient_id`, **no** por la columna `split`; es correcto solo porque `verify_patient_consistency()` corrió antes. `KeyError` crudo si un manifest usara `"validation"` o `"Train"` | `src/datasets/split.py:79,88,97` |
| d | Un `patient_id` de cadena vacía o `" "` no es NaN: pasaría el chequeo y colapsaría esas filas en un "paciente" fantasma | `src/datasets/manifest.py:70` |
| e | Un `preprocessed_image_path` absoluto haría que `pathlib` **descarte `image_root` en silencio** | `src/datasets/manifest.py:111-113` |
| f | Una BN con `affine=False` se trata como congelada (parámetros vacíos → `not any(...)` es `True`) | `src/train/loop.py:140-143` |
| g | Con `min_delta > 0`, el argmax real de val puede no llegar nunca a disco (afecta a exp04) | `early_stopping.py:87-91` + `trainer.py:245` |
| h | `total_loss / n_batches` revienta con `ZeroDivisionError` si el loader está vacío | `src/train/loop.py:127,190` |

**Verificar:** para cada ítem, escribir un caso mínimo que lo dispare (config sintético, manifest
sintético de 20 filas) y comprobar si falla con mensaje claro o en silencio.

**Aceptación:** tabla "falla ruidosa / falla silenciosa" por ítem.

### H7 — La capa federada hereda todo lo anterior · MEDIA / MEDIA

`src/federated/` reusa `train_one_epoch` y el ensamblaje centralizado, así que H1–H6 la afectan igual.

Específico de federado y **ya documentado — no hace falta re-descubrirlo**: el AUC de `best.json` es un
promedio ponderado por nodo, **no** el AUC del pool (exp46: 0,8647 ponderado vs **0,8471** pooled contra
0,9016 del centralizado), y los defaults de `fedadam`/`fedyogi` producen NaN o AUC 0,5000 exacto. Ver
`.claude/context/experiments/FEDERATED.md`.

**Verificar:**
- Que `federated/client.py` no introduzca divergencias propias respecto al loop centralizado (comparar
  el flujo de `client.py:191-214` contra `trainer.py:228-256`).
- Que `param_utils.get_model_ndarrays` / `set_model_ndarrays` preserven el orden de parámetros.

**Aceptación:** round-trip `get → set → get` idéntico bit a bit con un modelo sintético.

---

## 5. Harness sintético — lo que hace ejecutable la auditoría sin datos

Construir en un directorio temporal (**sin tocar el repo**) un generador de datos sintéticos que
permita ejercitar el pipeline **real**:

1. **Manifest sintético** (~200 filas) con las 21 columnas reales, `patient_id` con 2–4 imágenes por
   paciente, `split` train/val/test paciente-disjunto, `classification` Benign/Malignant.
2. **TIFFs float modo `"F"`** de 224×224 en rango `[-1,1]` y `[0,1]`:
   `Image.fromarray(np.random.rand(224,224).astype(np.float32)*2-1, mode='F')`.
3. **Dos variantes de etiquetado:**
   - **Señal controlada** — la media de la imagen codifica la etiqueta. Un pipeline correcto debe
     alcanzar **AUC ≈ 1,0**.
   - **Etiquetas aleatorias** — un pipeline sano debe dar **val_auc ≈ 0,5**. *Si con etiquetas
     aleatorias el val_auc supera 0,5 de forma consistente, hay fuga.* **Este es el test definitivo de
     fuga de datos y no requiere las imágenes reales.**
4. Correr `src.cli` completo contra ese manifest con `resnet50_imagenet_v2` (cacheado) y
   `unfreeze_from: none`, que es rápido en GPU.

El mismo harness sirve para verificar cada ítem de H6 y el round-trip de H7.

---

## 6. Archivos críticos

```
src/datasets/{manifest,split,dataset,build}.py
src/train/{loop,trainer,early_stopping,build}.py
src/metrics.py
src/eval_pipeline.py
src/config.py
src/federated/{client,param_utils}.py
configs/exp37_hpsearch_v1_e7u7fprr.yaml
sweeps/hpsearch_v1.yaml
```

Contexto de apoyo: `CLAUDE.md` (enrutador), `.claude/context/code/*.md` (invariantes por módulo),
`.claude/context/experiments/*.md` (resultados medidos), `PHASES.md` (qué se portó del proyecto INC y
por qué cada default es el que es).

---

## 7. Entregable

`docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md`, con:

- **Veredicto por hipótesis**: confirmada / descartada / no concluyente.
- **Evidencia** con `archivo:línea` o número medido y reproducible.
- **Clasificación explícita** de cada hallazgo en una de tres categorías, sin mezclarlas:
  **bug** · **decisión de diseño discutible** · **problema metodológico**.
- Una sección de **refutaciones**: todo punto de la sección 2 o 3 de este brief que la auditoría
  encuentre incorrecto. Es un resultado valioso, no un fallo.

---

## 8. Lo que esta auditoría NO puede cerrar aquí

Requiere la workstation con imágenes, pesos y GPU:

- **Reentrenar exp37 con `unfreeze_from: layer4`**, dropout > 0 y weight_decay más fuerte, para
  confirmar que el sobreajuste baja sin perder AUC de test.
- **Repetir exp37 con 3–5 semillas**, para saber qué diferencias entre experimentos son reales y
  cuáles son ruido.
- **Validación cruzada por pliegues** en vez de un único split de val, que es lo que cerraría H1 y H3
  de raíz.

**Ese es el trabajo que de verdad arregla el sobreajuste.** Esta auditoría estática solo puede decir
dónde apuntar, y descartar que haya además un bug escondido detrás.
