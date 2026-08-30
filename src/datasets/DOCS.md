# Documentación del Módulo de Datos (`src/datasets/`)

`src/datasets/` es el subsistema encargado de cargar manifests CSV, validarlos, dividirlos en particiones `train`/`val`/`test` sin fuga de pacientes, construir las transformaciones de imágenes con PyTorch torchvision y empaquetarlos en objetos `DataLoader`.

---

## Flujo Típico de Uso

```python
from src.datasets.manifest import Manifest
from src.datasets.split import Split
from src.datasets.transform import TransformBuilder
from src.datasets.build import builder_dataloader

# 1. Cargar y validar manifest
manifest = Manifest(manifest_path="manifests/fedmammobench.csv", image_root="data/")

# 2. Separar por paciente
split = Split(manifest=manifest)

# 3. Configurar transformaciones
train_tx = TransformBuilder(image_size=(224, 224), use_horizontal_flip=True, use_rotation=True)
eval_tx = TransformBuilder(image_size=(224, 224))

# 4. Crear DataLoaders para train, val y test
loaders = builder_dataloader(split, train_tx, eval_tx, batch_size=16, num_workers=2)
train_loader = loaders["train"]
val_loader = loaders["val"]
```

---

## Detalle por Archivo y Clase

### `manifest.py`

#### `Manifest`

Carga un manifest CSV y valida su integridad estructural y lógica:
1. `verify_columns()`: Asegura presencia de `preprocessed_image_path`, `classification`, `split`, `patient_id`.
2. `check_patients_id()`: Verifica que no existan `patient_id` nulos.
3. `normalize_labels()`: Mapea `'benign'` -> `0` y `'malignant'` -> `1` en la columna `label_norm`.
4. `resolve_image_paths()`: Construye rutas absolutas en `abs_image_path`.

##### Cómo usar `Manifest`:
```python
from src.datasets.manifest import Manifest

# Instanciación y validación automática
manifest = Manifest(
    manifest_path="manifests/fedmammobench.csv",
    image_root="/data/mammography/images"
)

# Inspeccionar DataFrame validado y procesado
print("Filas totales:", len(manifest.df))
print("Columnas añadidas:", manifest.df[["label_norm", "abs_image_path"]].head())
```

---

### `split.py`

#### `Split`

Garantiza la separación estricta por pacientes (anti-fuga entre `train`, `val` y `test`).

* `verify_patient_consistency()`: Lanza `ValueError` si un mismo paciente aparece en más de un split.
* `group_by_split()`: Retorna un diccionario con listas de `patient_id` por split.
* `train_df()`, `val_df()`, `test_df()`: Retornan DataFrames filtrados por la partición correspondiente.

##### Cómo usar `Split`:
```python
from src.datasets.manifest import Manifest
from src.datasets.split import Split

manifest = Manifest("manifests/fedmammobench.csv", "data/")
split = Split(manifest=manifest)

# Obtener DataFrames de cada partición
df_train = split.train_df()
df_val = split.val_df()
df_test = split.test_df()

print(f"Muestras: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
```

---

### `dataset.py`

#### `MammoBenchDataset`

Clase `torch.utils.data.Dataset` que carga imágenes mamográficas desde Pillow y aplica las transformaciones de torchvision.

##### Cómo usar `MammoBenchDataset`:
```python
from src.datasets.manifest import Manifest
from src.datasets.split import Split
from src.datasets.dataset import MammoBenchDataset
from src.datasets.transform import TransformBuilder

manifest = Manifest("manifests/fedmammobench.csv", "data/")
split = Split(manifest=manifest)

# Transformación de evaluación
transform = TransformBuilder(image_size=(224, 224)).build()

# Instanciar Dataset de PyTorch para validación
val_dataset = MammoBenchDataset(
    df=split.val_df(),
    grayscale=False,
    transform=transform
)

# Acceder a un ítem (tensor_imagen, etiqueta_int)
img_tensor, label = val_dataset[0]
print("Shape de imagen:", img_tensor.shape)  # torch.Size([3, 224, 224])
print("Etiqueta:", label)                   # 0 o 1
```

---

### `transform.py`

#### `TransformBuilder`

Builder de transformaciones de torchvision parametrizado por flags.

* `build()`: Retorna un `transforms.Compose` con Resize -> (Flip/Rotation aleatorios) -> ToTensor -> Normalize.

##### Cómo usar `TransformBuilder`:
```python
from src.datasets.transform import TransformBuilder

# Builder para entrenamiento con aumentaciones
builder = TransformBuilder(
    image_size=(512, 512),
    use_horizontal_flip=True,
    horizontal_flip_p=0.5,
    use_rotation=True,
    rotation_degrees=15,
    normalize_mean=(0.485, 0.456, 0.406),
    normalize_std=(0.229, 0.224, 0.225)
)

pipeline = builder.build()
```

---

### `build.py`

#### `builder_dataloader()`

Punto de entrada de alto nivel para generar el diccionario de DataLoaders de PyTorch (`train`, `val`, `test`).

##### Cómo usar `builder_dataloader()`:
```python
from src.datasets.manifest import Manifest
from src.datasets.split import Split
from src.datasets.transform import TransformBuilder
from src.datasets.build import builder_dataloader

manifest = Manifest("manifests/fedmammobench.csv", "data/")
split = Split(manifest=manifest)

train_tx = TransformBuilder(image_size=(224, 224), use_horizontal_flip=True)
eval_tx = TransformBuilder(image_size=(224, 224))

loaders = builder_dataloader(
    split=split,
    train_transform_builder=train_tx,
    eval_transform_builder=eval_tx,
    batch_size=32,
    num_workers=4,
    seed=42
)

# Iterar sobre el dataloader de entrenamiento
for images, labels in loaders["train"]:
    print(images.shape, labels.shape)
    break
```

---

## Exportaciones (`__init__.py`)

`src/datasets/__init__.py` reexporta los componentes principales:
```python
from src.datasets import Manifest, Split, MammoBenchDataset
```
