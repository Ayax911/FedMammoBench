# RESNET_SCRATCH_RESNET18_RESULTADOS — Resultados de Implementación y Verificación

Informe de ejecución técnica correspondiente al brief [`docs/RESNET_SCRATCH_RESNET18.md`](file:///home/akira/snap/steam/federallearning/FedMammoBench/docs/RESNET_SCRATCH_RESNET18.md).

---

## 1. Resumen de Archivos Modificados por Repositorio

### A. Repositorio INC (`inc-project-models-classification-detection-main/`)

1. **[`src/models/classification_images/models/image_models.py`](file:///home/akira/snap/steam/federallearning/inc-project-models-classification-detection-main/src/models/classification_images/models/image_models.py)**:
   - Importación de `resnet18, ResNet18_Weights` desde `torchvision.models`.
   - Creación de la clase `ResNet18Model(nn.Module)` (512 features de salida tras `avgpool`, backbone truncado a los primeros 9 módulos de torchvision).
   - En `get_image_model`: agregada la rama `elif model_name == "ResNet18": base_model = ResNet18Model(pretrained=pretrained)`.
   - Mensaje de `ValueError` actualizado para listar `'ResNet', 'ResNet18', 'DenseNet', or 'Inception'`.

2. **[`src/models/classification_images/options.py`](file:///home/akira/snap/steam/federallearning/inc-project-models-classification-detection-main/src/models/classification_images/options.py)**:
   - `images_model_choices` actualizado a `["Inception", "ResNet", "ResNet18", "DenseNet"]`.
   - Argumento `--from_scratch` agregado (inicializa backbone con pesos aleatorios, ignora `--path_image_model`).
   - Argumentos `--normalize_mean` y `--normalize_std` agregados en la sección de data loader.

3. **[`src/models/classification_images/models/get_model.py`](file:///home/akira/snap/steam/federallearning/inc-project-models-classification-detection-main/src/models/classification_images/models/get_model.py)**:
   - Validación de exclusión mutua: si `options.pretrained and options.from_scratch`, levanta `ValueError`.
   - `weigths_file = None if (options.pretrained or options.from_scratch) else options.path_image_model`.

4. **[`src/models/classification_images/dataloaders/dataloader_images.py`](file:///home/akira/snap/steam/federallearning/inc-project-models-classification-detection-main/src/models/classification_images/dataloaders/dataloader_images.py)**:
   - `Loader.__init__` acepta `normalize_mean=None, normalize_std=None`.
   - Construye `normalize = [T.Normalize(mean=[normalize_mean], std=[normalize_std])] if normalize_mean is not None and normalize_std is not None else []` y lo agrega al final de `transforms_train` y `transforms_test`.

5. **[`src/models/classification_images/training.py`](file:///home/akira/snap/steam/federallearning/inc-project-models-classification-detection-main/src/models/classification_images/training.py)**:
   - Instanciación de `Loader` pasa `normalize_mean=options.normalize_mean, normalize_std=options.normalize_std`.

6. **[`src/models/classification_images/run.sh`](file:///home/akira/snap/steam/federallearning/inc-project-models-classification-detection-main/src/models/classification_images/run.sh)**:
   - Reescrito para ejecución secuencial de las tres corridas (`exp61_resnet50_scratch_inc`, `exp62_resnet18_scratch_inc`, `exp63_resnet18_imagenet_inc`).
   - Soporta variable configurable `NUM_FREEZE_R18` (default: 60 = 100% congelado).

---

### B. Repositorio FedMammoBench (`FedMammoBench/`)

1. **[`src/models/build.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/models/build.py)**:
   - Importación de `resnet18, ResNet18_Weights`.
   - Agregadas las entradas `"resnet50_scratch"`, `"resnet18_scratch"` y `"resnet18_imagenet_v1"` a `_ARCHITECTURES` con `weights_from_factory=True`.
   - Docstrings ampliados explicando la semántica de `weights_from_factory=True` para pesos preentrenados y variantes aleatorias `*_scratch`. `build_model()` permanece intacto.

2. **[`src/models/freeze.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/models/freeze.py)**:
   - Docstrings actualizados de "ResNet50" a "ResNet (18/50)".

3. **[`src/datasets/dataset.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/dataset.py)**:
   - Actualizada la carga al esquema del INC: conversión directa a tensor PyTorch desde array nativo numpy (`torch.from_numpy`) y replicación a 3 canales vía `torch.cat` **antes** del `transform`.
   - `_default_transform` simplificado a `Resize((224, 224))` sin `ToTensor`.

4. **[`src/datasets/transform.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/transform.py)**:
   - Eliminado el paso intermedio `transforms.ToTensor()` (el pipeline recibe directamente tensores `(3, H, W)`).
   - Docstrings actualizados para `normalize_mean`/`std` aclarando compatibilidad con 1 o 3 valores por broadcast.

5. **[`src/config.py`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/config.py)**:
   - `ArchitectureConfig` docstring actualizado con los nombres nuevos (`resnet50_scratch`, `resnet18_scratch`, `resnet18_imagenet_v1`).

6. **Configs Creados**:
   - [`configs/exp61_resnet50_scratch.yaml`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs/exp61_resnet50_scratch.yaml): ResNet50 desde cero (`conv1`, `in_features: 2048`, `norm_neg1_1`, `normalize: null`).
   - [`configs/exp62_resnet18_scratch.yaml`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs/exp62_resnet18_scratch.yaml): ResNet18 desde cero (`conv1`, `in_features: 512`, `norm_neg1_1`, `normalize: null`).
   - [`configs/exp63_resnet18_imagenet.yaml`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs/exp63_resnet18_imagenet.yaml): ResNet18 ImageNet (`none`, `in_features: 512`, `norm_0_1`, `normalize: [0.449] / [0.226]`).

7. **Documentación Actualizada**:
   - [`src/models/DOCS.md`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/models/DOCS.md) y [`.claude/context/code/MODELS.md`](file:///home/akira/snap/steam/federallearning/FedMammoBench/.claude/context/code/MODELS.md): Tabla de arquitecturas y nota de `in_features: 512`.
   - [`src/datasets/DOCS.md`](file:///home/akira/snap/steam/federallearning/FedMammoBench/src/datasets/DOCS.md) y [`.claude/context/code/DATASETS.md`](file:///home/akira/snap/steam/federallearning/FedMammoBench/.claude/context/code/DATASETS.md): Esquema de carga tensor-first y replicación previa a transforms.

---

## 2. Salida Literal de las Verificaciones (§7)

### Verificación 1: Paridad de Carga FMB Antes vs Después
Evaluado con script sobre imágenes sintéticas (TIFF modo "F" en `[-1, 1]`, TIFF modo "F" en `[0, 1]`, PNG RGB uint8):
```text
=== Verificación 1: Paridad de carga FMB antes / después ===
[TIFF mode F [-1, 1]] image_size=None, mean=None, std=None:
  Shape: torch.Size([3, 224, 224]), Dtype: torch.float32
  torch.equal: True, max|Δ|: 0.0
  [image_size=(224,224)] max|Δ| (tensor resize vs PIL resize): 0.0
[TIFF mode F [0, 1]] image_size=None, mean=(0.449,), std=(0.226,):
  Shape: torch.Size([3, 224, 224]), Dtype: torch.float32
  torch.equal: True, max|Δ|: 0.0
  [image_size=(224,224)] max|Δ| (tensor resize vs PIL resize): 0.0
[PNG RGB uint8] image_size=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5):
  Shape: torch.Size([3, 224, 224]), Dtype: torch.float32
  torch.equal: True, max|Δ|: 0.0
  [image_size=(224,224)] max|Δ| (tensor resize vs PIL resize): 0.0
```

### Verificación 2: Paridad INC ↔ FMB
Evaluado sobre el mismo archivo TIFF modo "F" en `[-1, 1]`:
```text
=== Verificación 2: Paridad INC ↔ FMB ===
FMB shape: torch.Size([3, 224, 224]), dtype: torch.float32, label: 1
INC shape: torch.Size([3, 224, 224]), dtype: torch.float32, label: 1
torch.equal(tensor_fmb, tensor_inc): True, max|Δ|: 0.0
```

### Verificación 3: Smoke de Modelos en Ambos Repositorios (3 Variantes)
```text
=== Verificación 3: Smoke de modelos en ambos repos (3 variantes) ===

--- FMB Models ---
[FMB exp61 resnet50_scratch]
  LoadReport: matched=318, missing=0, unexpected=0
  Output shape: torch.Size([2, 2])
  Backbone trainable parameters: 23508032/23508032 (159/159 tensors)
[FMB exp62 resnet18_scratch]
  LoadReport: matched=120, missing=0, unexpected=0
  Output shape: torch.Size([2, 2])
  Backbone trainable parameters: 11176512/11176512 (60/60 tensors)
[FMB exp63 resnet18_imagenet_v1]
  LoadReport: matched=120, missing=0, unexpected=0
  Output shape: torch.Size([2, 2])
  Backbone trainable parameters: 0/11176512 (0/60 tensors)

[FMB Weight check: resnet18_scratch vs resnet18_imagenet_v1 conv1.weight]
  Weights are identical: False
  Max absolute difference: 1.0060406923294067

--- INC Models ---
[INC exp61 ResNet scratch]
  Output shape: torch.Size([2, 2])
  Backbone trainable parameters: 23508032/23508032 (159/159 tensors)
[INC exp62 ResNet18 scratch]
  Output shape: torch.Size([2, 2])
  Backbone trainable parameters: 11176512/11176512 (60/60 tensors)
[INC exp63 ResNet18 imagenet]
  Output shape: torch.Size([2, 2])
  Backbone trainable parameters: 0/11176512 (0/60 tensors)
```

### Verificación 4: Importación de Módulos FMB
```bash
.venv/bin/python -c "import src.cli; import src.evaluate; import src.gradcam"
.venv/bin/python -c "import src.federated.server; import src.federated.client; import src.federated.evaluate_node"
```
Salida:
```text
Imports 1 OK
Imports 2 OK
```

### Verificación 5: Validación de Nuevos Archivos YAML con Pydantic (`extra="forbid"`)
```text
Loaded configs/exp61_resnet50_scratch.yaml: ID=exp61_resnet50_scratch, Arch=resnet50_scratch, in_features=2048, unfreeze=conv1
Loaded configs/exp62_resnet18_scratch.yaml: ID=exp62_resnet18_scratch, Arch=resnet18_scratch, in_features=512, unfreeze=conv1
Loaded configs/exp63_resnet18_imagenet.yaml: ID=exp63_resnet18_imagenet, Arch=resnet18_imagenet_v1, in_features=512, unfreeze=none
```

### Verificación 6: Sintaxis y CLI del INC
- `bash -n src/models/classification_images/run.sh` pasó sin errores (código de salida 0).
- `python main.py --help` muestra correctamente `ResNet18` en `--image_model`, `--from_scratch`, `--normalize_mean` y `--normalize_std`.
- `python main.py --pretrained --from_scratch` levantó la excepción esperada:
  ```text
  ValueError: --pretrained y --from_scratch son excluyentes.
  ```

### Verificación 7: Ejecución Smoke de 1 Época
Ejecutadas sobre GPU local (NVIDIA GTX 1650 con `batch_size=16` debido al límite de 4GB VRAM local; en la workstation con GPU de mayor capacidad aplica el `batch_size=64` default):

| Repositorio | Experimento | Variante | Tiempo 1 Época | Estado |
|---|---|---|---|---|
| **INC** | `exp61_resnet50_scratch_inc` | ResNet50 scratch (todo entrenable) | ~160 s (total con test y val) | ✅ Terminado sin excepción |
| **INC** | `exp62_resnet18_scratch_inc` | ResNet18 scratch (todo entrenable) | 55.4 s | ✅ Terminado sin excepción |
| **INC** | `exp63_resnet18_imagenet_inc` | ResNet18 ImageNet (congelado) | 34.7 s | ✅ Terminado sin excepción |
| **FMB** | `exp61_resnet50_scratch` | ResNet50 scratch (`conv1`) | 165.8 s | ✅ Terminado sin excepción |
| **FMB** | `exp62_resnet18_scratch` | ResNet18 scratch (`conv1`) | 59.3 s (train: 39.8 s) | ✅ Terminado sin excepción |
| **FMB** | `exp63_resnet18_imagenet` | ResNet18 ImageNet (`none`) | 36.0 s (train: 15.2 s) | ✅ Terminado sin excepción |

---

## 3. Estado de Rutas y Desviaciones

- **Rutas de Workstation (§2)**:
  Las rutas de la workstation `imagenesmedicas` (`/media/imagenesmedicas/DATA1/...` y `/home/imagenesmedicas/...`) quedaron configuradas de forma idéntica en [`src/models/classification_images/run.sh`](file:///home/akira/snap/steam/federallearning/inc-project-models-classification-detection-main/src/models/classification_images/run.sh) y en los archivos YAML [`configs/exp61_resnet50_scratch.yaml`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs/exp61_resnet50_scratch.yaml), [`configs/exp62_resnet18_scratch.yaml`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs/exp62_resnet18_scratch.yaml) y [`configs/exp63_resnet18_imagenet.yaml`](file:///home/akira/snap/steam/federallearning/FedMammoBench/configs/exp63_resnet18_imagenet.yaml). En la máquina de desarrollo local `akira` dichas rutas no están montadas, por lo que las pruebas de 1 época se ejecutaron apuntando a los datos locales equivalentes en `/home/akira/snap/steam/preproccesed_julian/`.
- **Desviaciones**: Ninguna respecto al brief.
