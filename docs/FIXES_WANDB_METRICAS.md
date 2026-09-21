# Corrección de discrepancias W&B (step≠época, valores que no coinciden con local) — prompt para Antigravity

> Encarga implementar tres correcciones puntuales a `MetricsLogger` (`src/tracking.py`) y sus dos
> call sites federado/centralizado. Diagnóstico ya cerrado — no reabras la investigación, implementa
> el fix tal como está especificado aquí. Todo verificable sin datos reales de mamografía ni cuenta de
> W&B online (esta máquina no tiene ninguno de los dos); si trabajas en la workstation con cuenta de
> W&B real, usa también esa verificación adicional, pero la sintética es obligatoria de todos modos.

---

## 0. Resumen ejecutivo

Dos síntomas reportados: (a) las métricas de W&B no coinciden con las locales, (b) el eje "Step" de
W&B no corresponde de forma confiable a épocas. Diagnostiqué **tres mecanismos independientes**, todos
confirmados con evidencia real (no solo lectura de código — ver §1). Los tres se arreglan con cambios
acotados a `src/tracking.py` más dos líneas en `eval_pipeline.py` y `round_tracking.py`. Ningún
`metrics.csv` ya commiteado cambia — la corrección es exclusivamente del lado de lo que se le manda a
W&B.

---

## 1. Diagnóstico — ya cerrado, con evidencia

### M1 — `log_image()`/`log_table()` sin `step=` inflan el step más allá del último real

Fuente de verdad: leí el docstring de `wandb.sdk.wandb_run.Run.log` **directo del paquete instalado**
en `.venv` (wandb==0.29.0), no de memoria:

> *"By default, each call to `log` creates a new 'step'... If `step` is `None`, then an implicit
> auto-incrementing step is used."*

`src/tracking.py:178-200` (`log_image`) y `:221-236` (`log_table`) llaman a `wandb.log()` **sin**
`step=` — cada llamada post-entrenamiento (curvas, matriz de confusión, ROC, tablas de predicciones)
consume un step propio que no corresponde a ninguna época/ronda real.

Confirmado cruzando `_step` de `wandb-summary.json` (ya sincronizado localmente en `runs/*/wandb/`)
contra las filas reales de `metrics.csv` de la misma corrida:

| corrida | `_step` en W&B | filas reales en `metrics.csv` | inflación |
|---|---:|---:|---:|
| `runs/exp10_layer4_bce_2048_2` (centralizado, `-m src.cli`) | 192 | 180 épocas | +12 |
| `runs/exp05_fedmammobench_full_weighted` (centralizado) | 192 | 180 épocas | +12 |
| `runs/exp11_layer4_bce_h1024` (centralizado) | 182 | 170 épocas | +12 |
| `runs/exp46_fedgrid_fedprox_r30/server` (federado) | 37 | 30 rondas | +7 |
| `runs/exp46_fedgrid_fedprox_r30/nodes/cmmd` (federado) | 73 | 60 épocas locales | +13 |

Reproducible: `.venv/bin/python -c "import json; print(json.load(open('runs/exp10_layer4_bce_2048_2/wandb/run-20260906_221817-6enyt06d/files/wandb-summary.json'))['_step'])"`
→ `192`; `wc -l runs/exp10_layer4_bce_2048_2/metrics.csv` → 181 líneas (180 filas + header).

El docstring actual de `log_image()` en `tracking.py:183-186` afirma que sin `step=` W&B "lo trata
como un evento suelto ligado al summary" — **es incorrecto**. `summary.update()` (lo que usa
`log_summary()`) y `.log()` sin step son mecanismos completamente distintos; el segundo sí consume
step/historial, per el docstring citado arriba. Corrige también ese comentario, no solo el
comportamiento — un comentario que describe mal el mecanismo es peor que no tener comentario.

### M2 — Colisión de nombre entre la curva por época y el summary del mejor checkpoint

`src/eval_pipeline.py:124,133` (`evaluate_split()`) llaman `logger.log_summary({f"{split_name}_{k}":
v ...})` → para val, esto escribe `val_accuracy`, `val_auc`, etc. en el **summary** del run de W&B —
el mismo nombre que ya usa la serie por-época que sube `Trainer` cada época vía `logger.log()`
(`src/train/trainer.py:270`).

Son dos números **legítimamente distintos y ambos correctos en su contexto** — no es un bug de
cómputo, es un bug de que comparten nombre. Confirmado con datos reales de `exp10`:

- Summary de W&B: `val_auc = 0.8387932777404785`
  (`runs/exp10_layer4_bce_2048_2/wandb/run-20260906_221817-6enyt06d/files/wandb-summary.json`)
- Última fila de `metrics.csv` (época 179, la última entrenada): `val_auc = 0.8269051909446716`
  (`tail -1 runs/exp10_layer4_bce_2048_2/metrics.csv`, columna 11)

La curva por-época mide el modelo *tal como estaba en esa época exacta*; el summary reevalúa el
**mejor checkpoint** recargado (`eval_pipeline.py:123` → `evaluate_checkpoint()` → `load_checkpoint()`
→ `evaluate()`), que casi nunca es la última época entrenada porque el early stopping sigue corriendo
`patience` épocas después del óptimo antes de parar. **No cambies el valor — el fix es de nombre.**

### M3 — En federado, "epoch"/step significa cosas distintas en servidor y en nodo, agrupados en el mismo panel

`src/tracking.py:138` (`log()`) usa un solo parámetro genérico `epoch` como columna CSV / step de W&B,
sin distinguir "ronda" de "época local". `src/federated/round_tracking.py:159` le pasa `server_round`
(servidor); `src/federated/client.py:224` le pasa `self.global_epoch`, un contador continuo de épocas
locales a través de *todas* las rondas (con `local_epochs=2` y `rounds=30`, llega a 60 — el doble de
rondas). Servidor y los 4 nodos comparten `wandb_group` (mismo panel expandible en la UI de W&B), así
que sus ejes "Step" —de naturaleza y cardinalidad distinta (30 vs. 60 en exp46)— quedan visualmente
comparables sin serlo conceptualmente. El nodo ya loguea `"round"` como campo dentro de sus métricas
(`client.py` línea ~219, `row["round"] = float(current_round)`); el servidor no loguea ningún campo
*nombrado* "round" — solo lo usa como posición de step, nunca como valor visible/seleccionable.

**Decisión ya tomada, no la reabras**: no se toca la cabecera `"epoch"` de `server/metrics.csv` ni de
`nodes/*/metrics.csv` — están commiteados en 16 carpetas de `runs/` y renombrarlas es un cambio de
esquema retroactivo que no vale la pena en esta vuelta. La corrección es **solo del lado de W&B**.

### Hallazgo menor — documentado, fuera de alcance esta vuelta, no lo toques

`src/evaluate.py:139` abre un run de W&B nuevo por cada re-evaluación post-hoc
(`wandb_run_name=f"{experiment_id}-eval"`) que nunca llama a `log()` — su `_step` interno queda en un
dígito bajo (confirmado: 7 en `runs/exp10.../wandb/run-20260907_183944-8e9s5qqz/`), sin curva de
entrenamiento, solo summary. No colisiona de nombre con la corrida de entrenamiento (sufijo `-eval`
distinto), así que no es la causa central de "no coinciden", pero ensucia el proyecto de W&B con
corridas casi vacías sin límite. **No lo arregles en esta vuelta.**

---

## 2. El fix — diseño exacto, snippet por archivo

Todo el cambio de comportamiento vive en `src/tracking.py` (`MetricsLogger`), más dos call sites.

### 2.1 `src/tracking.py` — `MetricsLogger`

**En `__init__`**, añade el estado nuevo (junto a `self._wandb_run = None`):

```python
self._last_step: int | None = None
```

**`log()`** — añade el parámetro `wandb_extra` y registra el último step real al final:

```python
def log(
    self,
    epoch: int,
    metrics: dict[str, float],
    wandb_extra: dict[str, float] | None = None,
) -> None:
    """... (docstring existente + un párrafo nuevo documentando wandb_extra) ...

    Args:
        ...(los que ya existen)...
        wandb_extra: pares adicionales que se mandan SOLO a W&B (nunca a
            metrics.csv ni a TensorBoard) — para campos como "round" en el
            servidor federado, donde el CSV local no debe cambiar de
            esquema pero W&B sí necesita un campo nombrado además del
            step posicional (ver docstring de módulo / CLAUDE.md). None
            (default) no añade nada.
    """
    row: dict[str, float | int] = {"epoch": epoch, **metrics}

    # ... CSV y TensorBoard EXACTAMENTE como hoy, sin cambios, usando `metrics` tal cual ...

    if self._wandb_run is not None:
        payload = {**metrics, **(wandb_extra or {})}
        self._wandb_run.log(payload, step=epoch)

    self._last_step = epoch
```

Ojo: `self._last_step = epoch` va **al final**, después de todo lo demás (incluida la rama de wandb) —
no antes. Y el CSV/TensorBoard siguen recibiendo `metrics` sin `wandb_extra` mezclado — solo el
payload que se manda a `self._wandb_run.log()` lo incluye.

**`log_image()`** — reusa el último step real en vez de dejar que wandb auto-incremente uno nuevo:

```python
def log_image(self, name: str, path: str | Path) -> None:
    """... (docstring existente, PERO corrige el párrafo que dice que un log
    sin step "se liga al summary" -- es falso, ver M1 del brief. El párrafo
    correcto: sin step= explícito, W&B auto-incrementa un step nuevo por
    cada llamada no relacionado con ninguna época real. Por eso reusamos
    el último step de entrenamiento con commit=False -- así las imágenes
    post-entrenamiento se fusionan en la fila final ya buffereada de la
    última época en vez de crear steps fantasma.) ...
    """
    if self._wandb_run is not None:
        import wandb

        if self._last_step is not None:
            self._wandb_run.log(
                {name: wandb.Image(str(path))}, step=self._last_step, commit=False
            )
        else:
            # log() nunca se llamó en esta corrida (ej. evaluate.py -- solo
            # log_summary/log_image/log_table, sin curva de entrenamiento
            # que proteger) -- comportamiento actual, sin cambios.
            self._wandb_run.log({name: wandb.Image(str(path))})
```

**`log_table()`** — mismo patrón exacto, adaptado a `wandb.Table`:

```python
def log_table(self, name: str, csv_path: str | Path) -> None:
    """... (mismo ajuste de docstring que log_image) ..."""
    if self._wandb_run is not None:
        import pandas as pd
        import wandb

        table = wandb.Table(dataframe=pd.read_csv(csv_path))
        if self._last_step is not None:
            self._wandb_run.log({name: table}, step=self._last_step, commit=False)
        else:
            self._wandb_run.log({name: table})
```

**No toques `log_summary()`** — ya usa `self._wandb_run.summary.update(metrics)`, mecanismo correcto
y no relacionado con step/historial. Ese método queda exactamente igual.

### 2.2 `src/eval_pipeline.py` — prefijo `best_` en las claves de `log_summary()`

Cuatro llamadas a cambiar, todas con el mismo patrón (`f"{split_name}_{k}"` → `f"best_{split_name}_{k}"`,
o el equivalente para las de `evaluate_by_database`):

- Línea 124 (dentro de `evaluate_split()`):
  ```python
  logger.log_summary({f"{split_name}_{k}": v for k, v in split_metrics.items()})
  ```
  →
  ```python
  logger.log_summary({f"best_{split_name}_{k}": v for k, v in split_metrics.items()})
  ```
- Línea 133 (misma función, las métricas de matriz de confusión):
  ```python
  logger.log_summary({f"{split_name}_{k}": v for k, v in cm_metrics.items()})
  ```
  → mismo prefijo `best_`.
- Líneas 279-280 (dentro de `evaluate_by_database()`, dos llamadas análogas por base de datos): mismo
  patrón — `f"{split_name}_by_database_{db_name}_{k}"` → `f"best_{split_name}_by_database_{db_name}_{k}"`.

**Verifica el nombre exacto de la variable/f-string en cada línea antes de aplicar** — el snippet de
arriba es la transformación, no necesariamente el texto literal carácter por carácter; lee
`src/eval_pipeline.py` primero.

**No toques** ningún nombre de columna en `run_dir/{val,test}/metrics.json`,
`confusion_matrix_metrics.json`, `predictions.csv`, ni ninguna clave que no pase por
`logger.log_summary()`. El prefijo `best_` es exclusivo de lo que ve W&B.

### 2.3 `src/federated/round_tracking.py` — campo `"round"` nombrado, solo para W&B

En `aggregate_evaluate()`, la línea (~159):

```python
self.logger.log(server_round, row)
```

cambia a:

```python
self.logger.log(server_round, row, wandb_extra={"round": float(server_round)})
```

`row` (lo que va a CSV/TensorBoard) **no cambia** — `wandb_extra` es el canal nuevo de `log()` (§2.1)
que no toca CSV. Resultado: en W&B, además del step posicional (que hoy ya es igual al número de
ronda), el usuario tendrá un campo *nombrado* `"round"` que puede elegir como eje X personalizado en
los paneles del servidor (wandb soporta cualquier métrica como eje custom — "Custom log axes" en el
docstring de `Run.log`), sin depender del "Step" genérico que hoy comparte semántica ambigua con la
corrida de cada nodo en el mismo `wandb_group`.

---

## 3. Qué NO hacer en este pase

- No renombres ninguna columna de ningún `metrics.csv` ya commiteado — ni el del servidor ni el de
  ningún nodo, en ninguna de las 16+ carpetas de `runs/`.
- No toques `src/evaluate.py` ni las corridas `-eval` huérfanas (hallazgo menor, fuera de alcance).
- No intentes re-sincronizar ni corregir ninguna corrida ya subida a la cuenta de W&B — el fix es solo
  para corridas futuras, generadas después de este cambio.
- No toques TensorBoard — su eje de step ya es solo un entero sin ambigüedad de nombre, el problema es
  específico de W&B.
- No reabras nada de `docs/AUDITORIA_SOBREAJUSTE*.md` ni de `docs/FIXES_AUDITORIA_SOBREAJUSTE.md` —
  temas y ciclos distintos.
- No cambies `configs/*.yaml`.

---

## 4. Entregable esperado

Un diff acotado a `src/tracking.py`, `src/eval_pipeline.py`, `src/federated/round_tracking.py`, más un
resumen corto confirmando el resultado de cada verificación de la sección 5 (no hace falta un `.md`
nuevo — pero sí escríbelo si prefieres dejar rastro, con el nombre `docs/FIXES_WANDB_METRICAS_RESULTADOS.md`).

---

## 5. Verificación — sin datos reales ni cuenta de W&B online

1. **Test sintético offline de wandb.** `wandb.init(mode="offline")` en un directorio temporal →
   `log({"train_loss": 1.0}, step=0)`, `log({"train_loss": 0.5}, step=1)` (simula 2 épocas) → dos
   llamadas a `log_image()`/`log_table()` (con archivos dummy) que deben reusar `step=1` con
   `commit=False` → `close()`. Confirma que el `_step` final (leyendo el `.wandb` local sincronizado,
   o `wandb-summary.json` si el modo offline lo genera) es **1**, no 3. Esta es la reproducción
   sintética de la tabla de §1/M1.
2. **`metrics.csv` byte-idéntico antes/después.** Usa el mismo harness sintético (manifest de juguete +
   TIFFs modo `"F"`, como el de `docs/AUDITORIA_SOBREAJUSTE.md` §5) para correr `src.cli` con y sin el
   fix aplicado; diff de los `metrics.csv` resultantes — debe ser exactamente igual, cero columnas
   nuevas ni renombradas.
3. **`best_val_accuracy` ≠ `val_accuracy`.** En el mismo run sintético, confirma que el *summary* del
   run de W&B trae `best_val_accuracy` (con el valor del mejor checkpoint) y que la clave
   `val_accuracy` sin prefijo solo aparece en la serie por-época del historial, nunca en el summary.
4. **Servidor federado.** Con un `TrackedStrategy` de juguete (o una corrida sintética de 2 rondas),
   confirma que `server/metrics.csv` tiene exactamente las mismas columnas que antes del cambio, y que
   el payload que llega a `wandb_run.log()` incluye `"round"` como clave nombrada además del `step=`
   posicional.
5. Sanity check de imports: `.venv/bin/python -c "import src.cli; import src.evaluate;
   import src.federated.server; import src.federated.client; import src.federated.evaluate_node"`.
