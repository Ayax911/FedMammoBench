# Contexto de rama — `feature/radimagenet-refactor`

Reescritura desde cero del paquete de entrenamiento. El contenido anterior de
`src/fedmammobench/` fue **borrado a propósito** (commits `12cb3d1` + `6cfd5b6`, ~9.900
líneas). Este documento es el handoff de trabajo de la rama.

> **`CLAUDE.md` está obsoleto.** Describe en detalle el paquete que esta rama borró
> (CLI, registries, strategies, weight loaders, `Trainer`, resume, config system).
> Nada de eso existe ya en el árbol de trabajo. Lo único vigente de ese archivo es la
> sección de notebooks y las convenciones del repo. No seguir sus instrucciones de
> arquitectura ni sus comandos `fedmammobench-*`.

---

## 1. Objetivo y alcance

Ejercicio deliberado de aprendizaje: entender cada pieza del pipeline (antes eran ~59
módulos opacos), simplificar a ~15-18, y practicar diseño de clases con revisión crítica
de cada método.

El código nuevo debe **reproducir los resultados del legacy** (van a un paper), no ser una
reescritura libre.

**Alcance actual: solo entrenamiento centralizado.** Federated learning se pospone hasta
que el pipeline centralizado esté validado.

---

## 2. Estado verificado del árbol (2026-08-26, working tree limpio en `6cfd5b6`)

Todo `src/` son **155 líneas** repartidas en 4 archivos:

```
src/
├── __init__.py           # vacío
└── datasets/
    ├── __init__.py       # exporta Manifest, Split
    ├── manifest.py       # ✅ COMPLETO Y VERIFICADO (99 líneas)
    └── split.py          # ✅ COMPLETO Y VERIFICADO (49 líneas)
```

### ⚠️ Corrección de rutas respecto al handoff anterior

El documento de handoff previo asumía `src/fedmammobench/data/manifest.py` y `splits.py`.
**Las rutas reales son otras** y el cambio fue intencional (`6cfd5b6`):

| Handoff anterior decía | Realidad en el árbol | Por qué |
|---|---|---|
| `src/fedmammobench/…` | `src/…` | El paquete se aplanó; ya no hay wrapper `fedmammobench/` |
| `data/` | `datasets/` | `.gitignore` tiene una regla `data/` a nivel de repo que se tragaba el módulo entero en silencio |
| `splits.py` | `split.py` (singular) | — |

**No recrear un directorio `src/data/`** — git lo ignora y el módulo desaparece sin aviso.

---

## 3. Los 3 puntos "sin confirmar" del handoff: ✅ los tres resueltos

El handoff pedía verificar tres cosas en `split.py` antes de continuar. Verificadas
leyendo el archivo — **las tres ya están correctas en el código commiteado**:

1. `import pandas as pd` → ✅ presente (línea 2).
2. `self.splits: dict[str, list[str]] = self.group_by_split()` en `__init__` → ✅ presente.
3. `train_df()` / `val_df()` / `test_df()` → ✅ los tres con la versión correcta de
   `.isin(...)`. No queda rastro del intento roto (`pd.DataFrame.self.group_by_split(...)`
   ni el typo `def train_df(sel)`).

### ✅ Verificación de cobertura de filas: ejecutada y pasa

```
train 6671 + val 836 + test 834 = 8341 filas = total del manifest
pacientes: train 2027 | val 254 | test 253
```

Las tres particiones cubren cada fila exactamente una vez. `Manifest` + `Split` corren
limpio de punta a punta contra `manifests/fedmammobench.csv`.

---

## 4. Decisiones de arquitectura (vigentes)

| Decisión | Elección | Razón |
|---|---|---|
| Config | Pydantic v2 + YAML | Validación gratis; `extra="forbid"` atrapa typos |
| Herencia de config | **NO** — sin `defaults:`/`base.yaml` | Cada experimento legible de punta a punta |
| Registries por decorador | **Descartados** — dict dispatch | Ocultan de dónde viene cada componente |
| Un `Dataset` por fuente | **NO** — uno solo | Todo se consolida ya en un manifest CSV único |
| Federated | Pospuesto | Paquete `federated/` aparte, después |

### La decisión que simplificó `split.py`

El manifest **siempre llega con la columna `split` ya poblada** (estratificación aplicada
aguas arriba). Por eso `split.py` solo **valida** un split existente, nunca lo **genera** —
`make_patient_splits` / `_stratified_patient_split` con RNG quedaron **descartados por
innecesarios**.

⚠️ Consecuencia: la reproducibilidad del split depende enteramente del paso que genera esa
columna, **fuera de este paquete**. Este código no puede verificar que ese paso sea
determinista. `verify_patient_consistency()` es la única red de seguridad — detecta
leakage ya presente en el CSV, no lo previene.

---

## 5. Árbol objetivo

```
src/
├── __init__.py
├── config.py                # modelos Pydantic + load_config/save_config
├── seed.py                  # set_global_seed, seed_worker, make_generator
├── metrics.py               # compute_metrics(y_true, y_prob) -> dict
├── checkpoint.py            # save_checkpoint / load_checkpoint
├── tracking.py              # MetricsLogger: CSV + TensorBoard
├── cli.py                   # entrypoint argparse
├── datasets/
│   ├── manifest.py          # ✅ COMPLETO
│   ├── split.py             # ✅ COMPLETO
│   ├── dataset.py           # 🔴 SIGUIENTE — torch.utils.data.Dataset
│   ├── transforms.py        # ⬜ pipelines train/eval
│   └── build.py             # ⬜ build_dataloaders(cfg)
├── models/
│   ├── build.py             # ⬜ dict dispatch nombre -> constructor torchvision
│   ├── heads.py             # ⬜ build_head(in_features, cfg)
│   └── weights.py           # ⬜ load_weights, apply_freeze, LoadReport
└── train/
    ├── build.py             # ⬜ build_optimizer / build_scheduler / build_loss
    ├── loop.py              # ⬜ train_one_epoch, evaluate — funciones puras
    └── trainer.py           # ⬜ Trainer: épocas, early stop, checkpoint
```

**Dependencias, una sola dirección:**
```
config.py  ←  seed/metrics/checkpoint/tracking/datasets/models  ←  train/  ←  cli.py
```
Nada importa de `cli.py`. `datasets/` no importa de `models/` ni `train/`.
`models/weights.py` no importa de `models/build.py`. Imports tardíos dentro de una función
para romper un ciclo = síntoma de grafo mal diseñado, no solución.

---

## 6. `datasets/dataset.py` — 🔴 siguiente paso

### Contrato obligatorio (lo exige PyTorch, no es diseño libre)

Subclase de `torch.utils.data.Dataset` con `__len__(self) -> int` y
`__getitem__(self, idx: int) -> tuple[torch.Tensor, int]`.

### ✅ Las dos preguntas abiertas del handoff: respondidas

El handoff dejaba sin decidir la librería de transforms y grayscale-vs-RGB. **La
implementación de referencia ya existe** y es la que hay que reproducir: la clase
`CSVDataset` de `configs/exp28/exp28.ipynb` (celda 5), que produjo los resultados
commiteados en `runs/`.

1. **Transforms → `torchvision.transforms`**, no albumentations. Se aplican sobre un
   `PIL.Image`, no sobre un array numpy.
2. **RGB, no grayscale** → `Image.open(path).convert("RGB")`. Las mamografías son
   nativamente 1 canal, pero el backbone RadImageNet ResNet50 espera 3.

Pipeline exacto del legacy a replicar:

```python
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),   # squash, interpolación bilinear por defecto
    transforms.RandomRotation(ROTATION_DEGREES),   # 7 grados
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),   # NO son las stats de ImageNet
])
# val/test: idéntico pero sin RandomRotation ni RandomHorizontalFlip
```

### Gotcha del índice: dos soluciones, elegir una

Los DataFrames que devuelven `train_df()`/`val_df()`/`test_df()` tienen **índices no
contiguos** (el filtro `.isin()` conserva los índices originales, con huecos). Si
`__getitem__` usa `.loc[idx]` lanza `KeyError` en cuanto el `DataLoader` pida un índice
que no existe como label.

- El legacy hace **ambas**: `reset_index(drop=True)` al construir, y `.iloc[idx]` al leer.
- Mínimo indispensable: usar `.iloc[idx]` (acceso posicional).

### Checkpoint de reproducibilidad: el mapeo de etiquetas coincide

El legacy deriva `label_to_idx` de `sorted(self.data["classification"].unique())` sobre la
columna **cruda**: `sorted(["Benign", "Malignant"])` → `{"Benign": 0, "Malignant": 1}`.

`Manifest.normalize_labels()` mapea `{"benign": 0, "malignant": 1}` tras `.str.lower()`.

**Coinciden.** La columna nueva `label_norm` es intercambiable con el encoding del legacy —
no hay que invertir nada ni reordenar las clases.

### DataLoader (para `datasets/build.py`, más adelante)

```python
train = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                   generator=_seeded_generator, worker_init_fn=_seed_worker)
val = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)   # sin drop_last
```
`drop_last=True` **solo en train** y es obligatorio: la cabeza usa `BatchNorm1d`, que
revienta si el último batch de la época tiene 1 sola muestra.

---

## 7. Bugs del legacy que deben portarse deliberadamente

Aplican a `models/` y `train/`, aún no escritos. Documentados aquí para no perderlos:

- **BN bajo freeze.** `Trainer` llamaba `model.train()` cada época, lo que devolvía a modo
  entrenamiento las capas BatchNorm congeladas, que seguían actualizando
  `running_mean`/`running_var` aunque `requires_grad=False` frenara γ/β.
  Fix: `set_frozen_bn_eval(model)` inmediatamente después de cada `model.train()`.
- **Weight loading silencioso.** `LoadReport` reportaba `missing=0 unexpected=0` incluso
  cuando cargaban 0/320 tensores. Debe lanzar si `matched == 0`.
- **Prefijo `backbone.`.** Los checkpoints propios lo llevan; los constructores de
  torchvision no lo esperan. Normalizar claves antes de cargar.
- **Evaluar con el checkpoint final en vez del mejor.** Causa frecuente de resultados
  aparentemente malos. `Trainer.fit()` debe devolver la ruta del mejor checkpoint, y la
  evaluación de test debe usar siempre esa ruta, nunca una bandera de config.
- **Patient-level leakage (C1).** Ya no aplica directo (el split llega pre-calculado), pero
  es la razón de ser de `verify_patient_consistency()`.

El legacy sigue disponible en `main` para consultarlo (ver §9).

---

## 8. Entorno: qué corre y qué no

### ✅ `./venv/bin/python` SÍ sirve para este refactor

Contradice lo que dice `CLAUDE.md` ("no hay intérprete usable"). Es Python **3.12.3** y
tiene todo lo necesario:

```
torch 2.12.0+cu130 · torchvision 0.27.0 · pandas 2.3.3 · pytest 9.1.0
sklearn 1.9.0 · albumentations 1.4.24 · cv2 4.13.0 · matplotlib 3.11.0
ray 2.55.1 · flwr 1.31.0 · yaml 6.0.3          (falta solo: wandb)
```

Importar desde la raíz del repo (verificado):
```bash
./venv/bin/python -c "from src.datasets import Manifest, Split; print('ok')"
```
`src/__init__.py` existe, así que `src` es un paquete y `from src.datasets import …`
funciona sin tocar `PYTHONPATH`. Alternativa: `PYTHONPATH=src` + `from datasets import …`
— pero ese nombre colisiona con el paquete `datasets` de HuggingFace si alguien lo instala.
**Preferir `from src.datasets import …`.**

### ❌ `pip install -e .` no funciona, por dos razones independientes

1. `pyproject.toml` tiene `requires-python = ">=3.11,<3.12"` → rechaza el venv 3.12.
2. `[tool.setuptools.packages.find] include = ["fedmammobench*"]` → ese paquete ya no
   existe, así que instalaría **cero paquetes** aunque la versión de Python cuadrara.

Los `[project.scripts]` (`fedmammobench-centralized`, `-federated`, `-evaluate`) apuntan a
módulos borrados. `pyproject.toml`, `Dockerfile` y `.github/workflows/ci.yml` **todavía no
se actualizaron** para el refactor — es deuda pendiente, no algo ya resuelto.

### ❌ La suite de tests está muerta

`./venv/bin/python -m pytest tests/ -q` → **220 failed, 14 errors, 0 passed**. Los 14
archivos de `tests/` apuntan al paquete borrado (vía `sys.path.insert(…/"src")` +
`from fedmammobench…`). Ninguno cubre `Manifest` ni `Split`.

`tests/test_splits.py` sigue siendo el archivo de test más importante a escribir.

---

## 9. Recuperar el código legacy

El worktree `../fedmammobench-legacy` que mencionaba el handoff **no existe** (`git
worktree list` muestra solo el principal). Pero **`main` conserva el paquete completo**:

```bash
git ls-tree --name-only main:src/fedmammobench
# cli configs datasets evaluation federated models training utils __init__.py py.typed
```

Para consultarlo sin cambiar de rama:
```bash
git worktree add ../fedmammobench-legacy main
```
O puntualmente, sin worktree:
```bash
git show main:src/fedmammobench/training/trainer.py
git show main:src/fedmammobench/models/weight_loaders/custom.py
```

También borrados en esta rama y recuperables igual: `CHANGELOG.md` (410 líneas, explicaba
el *porqué* de cada cambio de comportamiento) y `run.sh`.

---

## 10. Datos

`manifests/fedmammobench.csv` — 8.341 filas, 2.534 pacientes, 21 columnas.

- Splits: train 6671 / val 836 / test 834
- Clases: `Benign` 5505 / `Malignant` 2836 (~66/34)
- Fuentes (`source_dataset`): `cmmd` 4722 · `kau-bcmd` 2206 · `cdd-cesm` 1003 · `inbreast` 410
- `patient_id`: 0 nulos
- ✅ **No hay filas `classification == "Normal"`** — la duda que dejaba abierta el handoff
  está resuelta; `normalize_labels()` no va a lanzar sobre este CSV. Sigue siendo una
  precondición a revalidar si el manifest se regenera.
- `preprocessed_image_path` es **relativa** (`Preprocessed_Dataset/cmmd/cmmd_0.jpg`), se
  resuelve contra `image_root`.

Los otros 4 CSV de `manifests/` son por-fuente, mismo esquema de columnas.

⚠️ `manifests/fedmammobench_tompei.csv` fue borrado en `ee54a6b`. Los notebooks
`exp23`–`exp27` todavía lo referencian y **fallarían al re-ejecutarse hoy**; sus resultados
commiteados siguen siendo válidos porque corrieron antes del borrado. `exp28`–`exp32` ya
apuntan al archivo vigente.

---

## 11. Referencia viva: la serie de notebooks

`configs/exp01`–`exp32` son notebooks Jupyter autónomos que **no importan el paquete** —
por eso sobrevivieron intactos al borrado, y por eso son la única implementación ejecutable
del pipeline que queda en la rama. `runs/` tiene resultados commiteados de exp01–exp27
(exp28–exp32 aún sin ejecutar).

**`configs/exp28/exp28.ipynb` es la referencia a reproducir** para el refactor: ResNet50 +
RadImageNet, cabeza MLP dropout 0.3, CrossEntropy, best-checkpoint por F1-macro, seed 42,
256px. De ahí salen las decisiones de §6.

Los cuatro generadores de `scripts/gen_*.py` son autocontenidos (solo stdlib + json) y
**siguen funcionando** — no dependen del paquete borrado.

---

## 12. Próximos pasos

1. **Escribir `src/datasets/dataset.py`.** Las dos preguntas abiertas ya están respondidas
   (§6): `torchvision.transforms` sobre PIL, `.convert("RGB")`, `.iloc[idx]`.
2. **Escribir `tests/test_dataset.py` y `tests/test_splits.py`** — primeros tests vivos del
   refactor. La suite actual está 100% muerta; conviene no arrastrarla.
3. Decidir si `pyproject.toml` se arregla ahora (renombrar `packages.find`, subir el pin a
   `>=3.12`, borrar los `[project.scripts]` muertos) o al final del refactor. Mientras siga
   roto, `pip install -e .` y el `Dockerfile` no sirven.
4. Después: `transforms.py` → `build.py` → `models/` → `train/`.
