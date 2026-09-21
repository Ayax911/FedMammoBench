# Correcciones de bajo riesgo de la auditoría de sobreajuste — prompt para Antigravity

> Encarga implementar cuatro correcciones puntuales derivadas de
> `docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md` (H5, H6.a, H6.b, H6.d). Todas son cambios acotados,
> sin necesidad de datos reales ni GPU, verificables con Pydantic/pytest puro o con el manifest
> sintético que ya construiste para el harness. No toques nada fuera de lo que se pide aquí — en
> particular, **no reabras H1–H4, H3 ni H7**, ya están cerradas y sus resultados se van a citar tal
> cual en `.claude/context/experiments/CENTRALIZED.md`.
>
> Antes de tocar código, lee `docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md` completo — es tu propio
> informe, y este documento asume que ya lo conoces. Dos correcciones a tu propio informe que hay que
> tener en cuenta al implementar (no hace falta editar el `.md`, solo no repetir el error en el código):
>
> 1. **H5**: tu informe recomienda "unificar `src/metrics.py:120` a `>=`". Es la dirección equivocada.
>    `src/metrics.py:117-120` tiene un comentario explícito que documenta por qué se eligió `>` a
>    propósito (coincide con la convención interna de torchmetrics). La inconsistencia real está en
>    `src/train/evaluation.py:115`, que usa `>=` sin ningún comentario que lo justifique. **El fix es
>    al revés de lo que recomendó tu informe**: cambiar `evaluation.py:115` a `>`, no tocar
>    `metrics.py`. Ver §1 abajo.
> 2. **H6.b**: tu informe cita `src/train/build.py:106` como si "instanciara con `mode='min'`". Esa
>    línea es solo la entrada del registro `_SCHEDULERS = {"reduceonplateu": ReduceLROnPlateau, ...}`
>    — no hay ningún `mode` hardcodeado en este repo; `build_scheduler()` lo splatea desde
>    `**hparams` del YAML (`src/train/build.py:131`). El riesgo es real (nada sincroniza
>    `scheduler.hparams["mode"]` con `train.metric_mode`) pero la causa no es una instanciación fija
>    en el código — es la ausencia de un validador cruzado. Ver §3 abajo.

---

## §1 — H5: unificar el umbral de binarización a `>` (no a `>=`)

**Archivo:** `src/train/evaluation.py`

En `predict_on_loader()`, línea ~115:

```python
y_pred.extend((probs >= 0.5).long().detach().cpu().tolist())  # pyright: ignore[...]
```

Cambiar a:

```python
y_pred.extend((probs > 0.5).long().detach().cpu().tolist())  # pyright: ignore[...]
```

Verificado por dos vías independientes que hoy el impacto es cero (0 filas con `y_prob == 0.5` exacto
en ningún `predictions*.csv` de `runs/`), así que este cambio no altera ningún artefacto commiteado —
es puramente para que `predictions.csv`/`confusion_matrix_metrics.json` (que vienen de esta función) y
`metrics.json` (que viene de `src/metrics.py`, con `>`) usen matemáticamente el mismo criterio en
cualquier corrida futura donde sí haya empates exactos en 0,5.

**Grep de verificación tras el cambio** (debe devolver solo la línea de `metrics.py`, ya documentada,
y la nueva de `evaluation.py`, ambas con `>`):

```bash
grep -n ">= 0.5\|> 0.5\|>= self.threshold\|> self.threshold" src/*.py src/*/*.py
```

No hay más sitios que binaricen con umbral en todo `src/` — ya se verificó exhaustivamente.

---

## §2 — H6.d: `patient_id` vacío o solo espacios no debe colar como paciente válido

**Archivo:** `src/datasets/manifest.py`, método `check_patients_id()`.

Hoy:

```python
def check_patients_id(self) -> None:
    null_mask = self.df["patient_id"].isna()
    if null_mask.any():
        n_null = int(null_mask.sum())
        raise ValueError(
            f"{n_null} row(s) have a missing patient_id. "
            "Every row must have a patient id — this is what prevents "
            "the same patient from landing in both train and test."
        )
```

Un `patient_id` de cadena vacía `""` o solo espacios `"  "` **no** es NaN, pasa este chequeo tal cual,
y luego `Split.verify_patient_consistency()` (`split.py:53-60`) agruparía todas esas filas bajo un
único "paciente" fantasma — si algunas cayeran en train y otras en val/test, sería fuga de datos real
que el chequeo actual no detecta.

**Fix — extender el mismo método** para tratar cadenas vacías/blancas igual que NaN:

```python
def check_patients_id(self) -> None:
    stripped = self.df["patient_id"].astype(str).str.strip()
    invalid_mask = self.df["patient_id"].isna() | (stripped == "") | (stripped.str.lower() == "nan")
    if invalid_mask.any():
        n_invalid = int(invalid_mask.sum())
        raise ValueError(
            f"{n_invalid} row(s) have a missing or blank patient_id. "
            "Every row must have a non-empty patient id — this is what prevents "
            "the same patient from landing in both train and test."
        )
```

(El `stripped.str.lower() == "nan"` cubre el caso en que un NaN real, al pasar por `.astype(str)`, se
convierte en el string literal `"nan"` — comprueba que no introduces un nuevo falso negativo ahí antes
de fijar la condición exacta.)

**No** hace falta tocar dónde se usa `patient_id` para agrupar (`split.py`) — validar en el punto de
entrada del manifest es suficiente, ese es el único lugar donde se decide si el dato es válido.

**Verificación** (sin datos reales, con un DataFrame sintético de 5 filas):

```python
import pandas as pd
from src.datasets.manifest import Manifest  # ajusta el import al constructor real

df = pd.DataFrame({
    "preprocessed_image_path": ["a.tiff", "b.tiff", "c.tiff"],
    "classification": ["Benign", "Malignant", "Benign"],
    "split": ["train", "train", "val"],
    "patient_id": ["P001", "  ", "P002"],  # la fila del medio debe fallar
})
# construir el Manifest con este df y llamar check_patients_id() debe lanzar ValueError
```

---

## §3 — H6.a + H6.b: validador cruzado de `metric_name`/`metric_mode`/`scheduler.mode`

**Archivo:** `src/config.py`. Hoy `TrainConfig` (y el resto de modelos del archivo) no tiene ningún
`@field_validator`/`@model_validator` — confirmado por grep, cero resultados.

Dos problemas reales, distintos, que un solo validador a nivel de `ExperimentConfig` puede cubrir:

**(a) `metric_name` fuera de las claves válidas, o `metric_mode` incoherente con esa métrica.**
Las claves válidas son las siete de `build_metric_collection()` (`src/metrics.py:162-169`:
`accuracy`, `auc`, `sensitivity`, `specificity`, `f1`, `f1_macro`, `precision`) más `"loss"`, que
`train_one_epoch()`/`evaluate()` añaden aparte (`src/train/loop.py:127,190`). De esas ocho, **`loss`
es la única donde "menor es mejor"** — el resto son "mayor es mejor". Hoy `metric_name: "loss"` con el
`metric_mode: "max"` por defecto (`src/config.py:265`) seleccionaría en silencio la **peor** época.

**(b) `scheduler.name == "reduceonplateu"` con un `mode` en `hparams` que no coincide con
`train.metric_mode`.** Ningún config actual usa este scheduler (todos usan `cosine`), así que el fix
es puramente preventivo — no debe cambiar el comportamiento de ningún config existente.

**Fix — añadir un `@model_validator(mode="after")` a `ExperimentConfig`** (la clase que agrupa todas
las secciones — revisa el nombre exacto al final de `config.py`; es la que construye `load_config()`):

```python
from pydantic import model_validator

_VALID_METRIC_NAMES = {"loss", "accuracy", "auc", "sensitivity", "specificity", "f1", "f1_macro", "precision"}
_LOWER_IS_BETTER = {"loss"}

class ExperimentConfig(BaseModel):
    ...

    @model_validator(mode="after")
    def _validate_metric_consistency(self) -> "ExperimentConfig":
        name = self.train.metric_name
        mode = self.train.metric_mode

        if name not in _VALID_METRIC_NAMES:
            raise ValueError(
                f"train.metric_name={name!r} no es una clave válida. "
                f"Opciones: {sorted(_VALID_METRIC_NAMES)}"
            )

        expected_mode = "min" if name in _LOWER_IS_BETTER else "max"
        if mode != expected_mode:
            raise ValueError(
                f"train.metric_name={name!r} espera train.metric_mode={expected_mode!r}, "
                f"pero el config trae metric_mode={mode!r}. "
                "Con 'loss' hay que minimizar; con el resto de métricas clínicas, maximizar."
            )

        if self.scheduler is not None and self.scheduler.name == "reduceonplateu":
            sched_mode = self.scheduler.hparams.get("mode")
            if sched_mode is not None and sched_mode != mode:
                raise ValueError(
                    f"scheduler.hparams.mode={sched_mode!r} no coincide con "
                    f"train.metric_mode={mode!r}. ReduceLROnPlateau debe optimizar en la "
                    "misma dirección que la métrica de selección de checkpoint."
                )

        return self
```

Ajusta nombres de campo/clase a lo que encuentres realmente en `config.py` (el snippet de arriba es la
lógica, no necesariamente los nombres exactos de atributo) — **lee el archivo primero**, no asumas.

**Verificación — dos casos deben fallar, todos los configs existentes deben seguir cargando:**

```bash
# 1. Todos los configs reales siguen validando (nada de esto debe lanzar):
.venv/bin/python -c "
from pathlib import Path
from src.config import load_config
for p in Path('configs').glob('*.yaml'):
    try:
        load_config(p)
    except Exception as e:
        print(f'FALLO INESPERADO en {p}: {e}')
"
# también recorre configs/federated/*/*.yaml con el loader que corresponda (load_server_config /
# load_node_config, en src/federated/config.py) -- confirma primero si ese árbol usa la misma
# ExperimentConfig o sus propios modelos antes de decidir si el validador aplica ahí también.

# 2. metric_name="loss" con metric_mode="max" (default) debe lanzar ValidationError ahora:
.venv/bin/python -c "
from src.config import ExperimentConfig
# construye un ExperimentConfig mínimo válido salvo por train.metric_name='loss' con el
# metric_mode por defecto ('max') y confirma que level ValidationError
"
```

**No toques** `src/federated/config.py` en este pase salvo para *confirmar* si `ServerTrackingConfig`
(`best_metric_name`/`best_metric_mode`, líneas 159-160) tiene el mismo riesgo — si lo tiene, repórtalo
en el resumen final pero no lo arregles sin que se apruebe por separado; es una clase distinta y un
cambio ahí no estaba en el alcance acordado.

---

## Qué NO hacer en este pase

- No toques `src/datasets/split.py` (los riesgos H6.c/e/f del informe original quedan para otro pase;
  no estaban en el alcance de "bajo riesgo" que se aprobó).
- No reabras H1, H2, H3, H4 ni H7 — ya están cerradas y documentadas.
- No regeneres ningún `runs/*/metrics.json` ni artefacto commiteado — estos cuatro cambios no deben
  alterar ningún número ya reportado (verificado arriba que el impacto actual es cero en cada caso).
- No toques `.claude/context/experiments/CENTRALIZED.md` — esa actualización ya se hizo por separado.

## Entregable

Un diff acotado a `src/train/evaluation.py`, `src/datasets/manifest.py` y `src/config.py`, más un
resumen corto (no hace falta un `.md` nuevo) de:
- qué pasó al re-validar los 30+ configs de `configs/` contra el nuevo validador (todos deben seguir
  cargando sin cambios);
- si `src/federated/config.py` tiene el mismo riesgo de H6.a/b (reportar, no arreglar);
- confirmación de que el grep de §1 no encontró sitios adicionales de binarización por umbral.
