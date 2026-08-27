# Documentación de la carpeta

`src/datasets/` carga el manifest CSV, lo valida, lo separa en splits `train`/`val`/`test`
sin fuga de pacientes entre splits, construye el pipeline de transforms y arma los
`Dataset`/`DataLoader` de PyTorch que consume el resto del pipeline. Flujo típico:

```python
manifest = Manifest(manifest_path="manifests/fedmammobench.csv", image_root="data/")
split = Split(manifest=manifest)

train_tx = TransformBuilder(image_size=(224, 224), use_horizontal_flip=True)
eval_tx = TransformBuilder(image_size=(224, 224))

loaders = builder_dataloader(split, train_tx, eval_tx, batch_size=16)
train_loader = loaders["train"]
```

## manifest.py

### Manifest

Carga un manifest CSV y lo deja listo para usar: valida columnas, valida `patient_id`,
normaliza la etiqueta y resuelve rutas absolutas de imagen. Todas las validaciones
corren dentro de `__init__`, en este orden — si algo falla, el objeto nunca llega a
existir a medias.

Inputs:
- `manifest_path` (`str | Path`): ruta al CSV del manifest.
- `image_root` (`str | Path`): directorio contra el que se resuelven las rutas
  relativas de imagen del manifest.

Atributos tras `__init__`:
- `self.manifest_path` / `self.image_root`: versiones `Path` de los inputs.
- `self.df`: `DataFrame` del CSV, aumentado en el sitio por cada paso de validación
  con `label_norm` (`normalize_labels`) y `abs_image_path` (`resolve_image_paths`).

Raises:
- `FileNotFoundError`: `manifest_path` no existe (la lanza `pd.read_csv`).
- `NotADirectoryError`: `image_root` no existe.
- `ValueError`: falta una columna requerida, hay `patient_id` nulos, o hay valores de
  `classification`/`preprocessed_image_path` inválidos.

#### verify_columns()

Revisa que `self.df` tenga las columnas requeridas: `preprocessed_image_path`,
`classification`, `split`, `patient_id`.

Raises: `ValueError` listando las columnas faltantes.

#### check_patients_id()

Revisa que ninguna fila tenga `patient_id` nulo — es la garantía de base para que
`Split` pueda evitar que un mismo paciente caiga en dos splits distintos.

Raises: `ValueError` con el número de filas afectadas.

#### normalize_labels()

Agrega `self.df['label_norm']` (`int`, 0/1) a partir de la columna cruda
`classification` (`'benign'` → 0, `'malignant'` → 1, sin distinguir mayúsculas ni
espacios). No modifica `classification`.

Raises: `ValueError` con los valores no reconocidos, si alguno no mapea a `benign`/`malignant`.

#### resolve_image_paths()

Agrega `self.df['abs_image_path']` uniendo `preprocessed_image_path` con
`self.image_root` (`image_root / preprocessed_image_path`, como string).

Raises:
- `NotADirectoryError`: `self.image_root` no existe.
- `ValueError`: alguna fila tiene `preprocessed_image_path` nulo.

## split.py

### Split

Deriva los splits `train`/`val`/`test` a **nivel paciente** a partir de un `Manifest`
ya validado, y expone un `DataFrame` filtrado por split. No decide ni recalcula el
split — solo agrupa y valida el que ya trae la columna `split` del manifest.

Inputs:
- `manifest` (`Manifest`, *keyword-only*): manifest ya cargado y validado.

Atributos tras `__init__`:
- `self.manifest`, `self.patient_col` (`"patient_id"`), `self.split_col` (`"split"`).
- `self.splits`: `dict[str, list[str]]`, resultado de `group_by_split()`.

Raises: `ValueError` si algún paciente aparece en más de un split (ver
`verify_patient_consistency`).

#### verify_patient_consistency()

Revisa que ningún `patient_id` tenga filas repartidas en más de un valor de `split` —
la garantía anti-fuga entre train/val/test.

Raises: `ValueError` con la lista de pacientes inconsistentes.

#### group_by_split()

Agrupa `manifest.df` por `split` y devuelve, para cada split, la lista de
`patient_id` únicos que le pertenecen. Solo agrupa lo que ya está en el manifest, no
calcula ninguna proporción.

Returns: `dict[str, list[str]]`, p. ej. `{"train": [...], "val": [...], "test": [...]}`.

#### train_df() / val_df() / test_df()

Devuelven `manifest.df` filtrado a las filas cuyo `patient_id` está en
`self.splits["train"]` / `["val"]` / `["test"]` respectivamente. Cada llamada relee
`manifest.df`, así que reflejan cualquier mutación posterior del manifest.

Returns: `pd.DataFrame`.

## dataset.py

### MammoBenchDataset(Dataset[tuple[torch.Tensor, int]])

`Dataset` de PyTorch sobre un `DataFrame` ya filtrado por split (normalmente el que
devuelve `Split.train_df()` / `val_df()` / `test_df()`). Lee la imagen desde
`abs_image_path` y la etiqueta desde `label_norm` — ambas columnas las agrega
`Manifest`, así que espera un `DataFrame` que ya pasó por ahí.

Inputs:
- `df` (`pd.DataFrame`): filas con, como mínimo, `abs_image_path` y `label_norm`.
- `grayscale` (`bool`, default `False`): si `True`, abre la imagen en modo `"L"`
  (1 canal) en vez de `"RGB"` (3 canales).
- `transform` (`transforms.Compose | None`): pipeline de torchvision a aplicar sobre
  la imagen ya abierta. Si es `None`, usa `_default_transform()`.

#### _default_transform()

Transform mínimo de reemplazo cuando no se pasa uno explícito: `Resize((224, 224))` +
`ToTensor()`. SUPUESTO: 224×224 — ajústalo si el modelo espera otro tamaño (o pasa un
`transform` propio, p. ej. desde `TransformBuilder`). A diferencia de
`TransformBuilder.build()`, no normaliza.

Returns: `transforms.Compose`.

#### __len__()

Returns: número de filas de `self.df`.

#### __getitem__(idx)

Carga la fila `idx`: abre `abs_image_path` con Pillow en modo `"L"` o `"RGB"` según
`self.grayscale`, le aplica `self.transform`, y castea `label_norm` a `int`.

Returns: `tuple[torch.Tensor, int]` — `(imagen_transformada, label)`.

## transform.py

### TransformBuilder

Arma un `transforms.Compose` de torchvision a partir de flags booleanos, en vez de
que cada caller construya la lista de pasos a mano. Pensado para pasarse a
`builder_dataloader` (o, ya construido con `.build()`, directo al `transform=` de
`MammoBenchDataset`).

Inputs:
- `image_size` (`tuple[int, int]`, default `(224, 224)`): tamaño final (alto, ancho)
  tras el resize.
- `use_horizontal_flip` (`bool`, default `False`): agrega flip horizontal aleatorio.
- `use_rotation` (`bool`, default `False`): agrega rotación aleatoria.
- `rotation_degrees` (`int`, default `15`): rango máximo de rotación, solo aplica si
  `use_rotation=True`.
- `horizontal_flip_p` (`float`, default `0.5`): probabilidad del flip, solo aplica si
  `use_horizontal_flip=True`.
- `normalize_mean` (`tuple[float, float, float]`, default `(0.5, 0.5, 0.5)`): media por
  canal para `transforms.Normalize`.
- `normalize_std` (`tuple[float, float, float]`, default `(0.5, 0.5, 0.5)`): desviación
  estándar por canal para `transforms.Normalize`.

Nota: `normalize_mean`/`normalize_std` traen 3 valores fijos (pensados para `RGB`); si
se usa con `MammoBenchDataset(grayscale=True)` (1 canal), `Normalize` fallará por
desajuste de canales — hay que pasar tuplas de un solo valor en ese caso.

#### build()

Construye el pipeline en orden fijo: `Resize` → (`RandomHorizontalFlip` si
`use_horizontal_flip`) → (`RandomRotation` si `use_rotation`, con `fill=0`) →
`ToTensor` → `Normalize(mean=normalize_mean, std=normalize_std)`.

Returns: `transforms.Compose`.

## build.py

### builder_dataloader(split, train_transform_builder, eval_transform_builder, batch_size=16, num_workers=1)

Arma los tres `MammoBenchDataset` (train/val/test) a partir de un `Split` ya creado y
los envuelve en `DataLoader`. Es el punto de entrada que junta todo lo demás del
paquete: `Split` → `TransformBuilder` → `MammoBenchDataset` → `DataLoader`.

Inputs:
- `split` (`Split`): ya construido sobre un `Manifest` válido.
- `train_transform_builder` (`TransformBuilder`): se usa para `train` — normalmente con
  augmentations activas (`use_horizontal_flip`/`use_rotation`).
- `eval_transform_builder` (`TransformBuilder`): se usa para `val` **y** `test` — sin
  augmentations, para que la evaluación sea determinista.
- `batch_size` (`int`, default `16`).
- `num_workers` (`int`, default `1`).

Comportamiento fijo (no configurable vía parámetros): `train` se baraja
(`shuffle=True`); `val` y `test` no (`shuffle=False`).

Returns: `dict[str, DataLoader[tuple[torch.Tensor, int]]]` con las claves `"train"`,
`"val"`, `"test"`.

## __init__.py

Reexporta `Manifest`, `Split` y `MammoBenchDataset` (`__all__` con esos tres nombres).
`build.py` no se reexporta aquí — se importa directo como
`from .build import builder_dataloader` (o `from datasets.build import ...`).
