# `src/train/` — loop, optimizadores, pérdidas, early stopping, `Trainer`

Contratos por método con ejemplos: **`src/train/DOCS.md`**. Esta hoja es el porqué y los invariantes.

Idioma: todo `train/` está en **español**.

Muchos de los defaults que se ven raros aquí son decisiones de "reproduce al INC exactamente".
**Lee `PHASES.md` antes de tocar** métricas F1/precision, `drop_last`, determinismo de cuDNN,
`FocalLoss`, `EarlyStopping`, evaluación sobre test o checkpoints periódicos.

---

## `LossSpec` — la abstracción que hay que entender antes de editar el loop

`train/build.py` define `LossSpec`, que **empareja la función de pérdida con la conversión correcta de
logits → probabilidad de clase positiva**. Esa conversión depende de **cuántos logits emite el head**
(1 para BCE, 2 para CrossEntropy/Focal), **no del nombre de la pérdida**.

`train_one_epoch()` y `evaluate()` llaman a `spec.compute()` / `spec.probs()` **sin ninguna
ramificación propia**. El único `if` vive dentro de `build_loss()` y corre una sola vez.

Si agregas una pérdida: añade su `_make_*` a `_LOSSES` devolviendo un `LossSpec` con la conversión
correcta. **No metas un `if` por nombre de pérdida dentro del loop** — es justo lo que esta estructura
existe para evitar.

Registros planos, grepeables en un solo lugar: `_OPTIMIZERS`, `_SCHEDULERS`, `_LOSSES`.
`build_param_groups()` permite LR distintos por grupo (backbone vs head).

`FocalLoss` (`focal_loss.py`) está portado del INC. Ojo con el **bug del `weight` como lista**
documentado en `PHASES.md`: una lista de Python donde se espera un tensor se acepta en silencio y
cambia el comportamiento.

---

## Invariantes

### Deriva de BatchNorm bajo congelamiento — `_set_frozen_bn_eval()`

`model.train()` **re-habilita las BatchNorm congeladas**: su `running_mean`/`running_var` siguen
actualizándose aunque estén en `requires_grad=False`. `train/loop.py:_set_frozen_bn_eval()` las vuelve
a poner en `eval()` justo después de cada `model.train()`.

`freeze_bn_stats: false` apaga eso **a propósito**, para reproducir al INC. Con un backbone totalmente
congelado, esa es la diferencia entre un backbone que todavía se adapta y uno clavado a las
estadísticas de RadImageNet. No es un bug: es un eje experimental.

### Evaluar el mejor checkpoint, nunca el último

`Trainer.fit()` **devuelve la ruta del mejor checkpoint** y `cli.run()` alimenta exactamente esa a
`eval_pipeline.evaluate_split()` para val y para test. **Nunca agregues un flag de config del tipo
"qué checkpoint evaluar"**: se desincroniza de lo que de verdad fue el mejor.

### `EarlyStopping` exige mejora estricta

Por eso `metric_name: f1` es una trampa sobre `fedmammobench.csv`: `BinaryF1Score` se queda clavado en
exactamente `0.0` mientras el modelo no predice malignos, esa racha de ceros nunca resetea la
paciencia, y `fit()` termina devolviendo el checkpoint de la época 0 sin entrenar. Usa `f1_macro` ahí.
Detalle completo en [CONFIG.md](CONFIG.md).

`min_delta` aplica igual a `EarlyStopping` y a la selección del mejor checkpoint.

---

## Funciones puras vs `Trainer`

`train_one_epoch()` y `evaluate()` (`loop.py`) son **puras**: reciben modelo, loader, spec y
dispositivo, y devuelven métricas. No escriben archivos ni saben de `run_dir`.

`Trainer` (`trainer.py`) es quien orquesta: épocas, scheduler, early stopping, checkpointing
(mejor + periódico cada `save_every`), y el logging vía `MetricsLogger`.

`evaluation.py` aporta `evaluate_checkpoint()` y `predict_on_loader()` — la base sobre la que
`eval_pipeline.py` construye `evaluate_split()`/`evaluate_by_database()`, y lo que reusa
`src/federated/` para evaluar en cada nodo.

Esa separación es lo que permite que `federated/client.py` reuse el loop tal cual, sin duplicarlo ni
importar `cli.py`.

---

## Scripts post-hoc que leen una corrida terminada

Los tres se invocan con `-m` (importan `src.*`) y **nunca reentrenan**:

- **`scripts/calibrate_threshold.py`** — elige el umbral de decisión sobre val y lo aplica a test.
  `--by-database` lo hace por base. **Es obligatorio antes de reportar nada por base**: con el 0.5
  fijo, kau-bcmd queda en ~0 de sensibilidad pese a AUC ~0,9 (4,3 % de malignos). Ver
  [../experiments/CENTRALIZED.md](../experiments/CENTRALIZED.md).
- **`scripts/ensemble_eval.py`** — promedia `y_prob` entre corridas en un directorio con forma de
  `run_dir` real.
- **`scripts/split_manifest_by_database.py`** — regenera `manifests/by_database/`.

`scripts/DOCS.md` describe el paquete borrado; no lo uses para decidir qué correr.
