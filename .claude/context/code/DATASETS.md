# `src/datasets/` — manifest, split, dataset, transforms, dataloaders

Contratos por método (args, raises, returns) con ejemplos: **`src/datasets/DOCS.md`**. Esta hoja es el
porqué y los invariantes.

Idioma: `build.py` está en español; `dataset.py` y `manifest.py` en inglés. Acompaña al archivo.

`datasets/` **nunca importa `models/` ni `train/`.**

---

## Las piezas

- **`Manifest`** (`manifest.py`) — carga el CSV. Exige las columnas `preprocessed_image_path`,
  `classification`, `split` y `patient_id`; normaliza las etiquetas a `label_norm` y resuelve
  `abs_image_path` contra `image_root`.
- **`Split`** (`split.py`) — **verifica** que la columna `split` que ya trae el manifest sea
  paciente-disjunta. **Nunca genera un split**: la estratificación pasa aguas arriba, fuera de este
  repo. Si necesitas un split distinto, se cambia el manifest, no este módulo.
- **`MammoBenchDataset`** (`dataset.py`) — un `Dataset` único para todas las fuentes. No hay un
  `Dataset` por base de datos: ya está todo consolidado en un solo manifest CSV.
- **`TransformBuilder`** (`transform.py`) — arma el pipeline de torchvision desde `DataConfig`.
- **`builder_dataloader()`** (`build.py`) — arma los `DataLoader`.

---

## Invariantes

### `.iloc`, nunca `.loc`
Los DataFrames que devuelve `Split` conservan sus índices originales **no contiguos** (son subconjuntos
del manifest completo). El acceso posicional es obligatorio en `__getitem__`. Un `.loc` ahí lee la fila
equivocada o revienta, según el índice.

### `drop_last=True` solo en el loader de train
`StandardMLPHead` usa `BatchNorm1d`, que lanza excepción con un batch final de tamaño 1. Val y test no
llevan `drop_last` — descartar muestras de evaluación falsearía las métricas.

### Nunca crear `src/data/`
`.gitignore` tiene una regla `data/` para todo el repo que se tragaría el módulo entero en silencio.
Por eso el paquete se llama `src/datasets/`. (Y por eso tampoco se puede usar `PYTHONPATH=src` +
`from datasets import ...`: ese nombre choca con el `datasets` de HuggingFace. `src/__init__.py` hace
que `from src.datasets import ...` funcione desde la raíz del repo sin tocar `PYTHONPATH`.)

### Modo PIL `"F"`: no llamar a `.convert()`
`__getitem__` despacha según el modo de la imagen PIL abierta:

- **Imágenes de 8 bits** (JPG/PNG/TIFF uint8, cualquier modo que no sea `"F"`): `.convert("RGB")` como
  siempre, y luego `transform` (que normalmente incluye `ToTensor()` reescalando `[0,255] → [0,1]`).
- **TIFF float de 32 bits** (modo `"F"`, la salida de `Preproccesed/preprocess_images.py`): se pasan a
  `transform` **sin `.convert()`**. PIL **recorta** floats en vez de reescalarlos al convertir de `"F"`
  a `"RGB"`, lo que colapsaría una imagen ya normalizada a `[-1,1]` a casi todo ceros. `Resize`
  interpola los floats tal cual y `ToTensor` reconoce `"F"` y copia los valores sin el `/255` habitual.

### Siempre 3 canales, y la replicación va después del transform
No existe el flag `grayscale` (se quitó en `ffdb5b2`). Para modo `"F"`, la replicación a 3 canales
ocurre sobre el **tensor, después** de `transform`, vía `np.concatenate`.

Consecuencia que hay que tener presente al escribir configs: cuando corre `transforms.Normalize`, el
tensor **todavía tiene 1 canal**. Por eso `normalize_mean`/`normalize_std` deben ser tuplas de un
elemento para este encoding, y una tripleta de ImageNet da shape mismatch. Ver
[CONFIG.md](CONFIG.md).

---

## Augmentación

`AugmentationConfig` cubre flip horizontal, rotación, flip vertical y blur gaussiano, cada uno con su
probabilidad. Flip vertical y blur se portaron desde el INC (ver `PHASES.md`) y están en `False` por
defecto; los configs que replican al INC los activan con p=0.2 y p=0.3.

**No agregues `ColorJitter`.** Sobre TIFF float de un canal destruye la imagen — está medido, ver la
memoria del diagnóstico de sobreajuste. Las augmentaciones geométricas y el blur son seguras porque
operan sobre la geometría o el suavizado, no sobre el rango de valores que el pipeline ya fijó.

---

## Determinismo

`seed.py` aporta `seed_worker()` y `make_generator()`; `builder_dataloader()` los cablea para que los
subprocesos del DataLoader sean deterministas. `set_global_seed(seed, cudnn_deterministic=True)` se
llama al principio de `cli.run()`. El determinismo de cuDNN se portó del INC (`PHASES.md`).
