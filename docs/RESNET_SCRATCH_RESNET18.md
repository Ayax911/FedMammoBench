# RESNET_SCRATCH_RESNET18 — ResNet50/ResNet18 desde cero + ResNet18 ImageNet, en INC y FedMammoBench

Brief para Antigravity (ciclo `/antigravity`, Fase 1). Redactado 2026-09-23. Informe esperado:
`docs/RESNET_SCRATCH_RESNET18_RESULTADOS.md` (ver §6).

Dos repos involucrados — en este documento:
- **FMB** = este repo, `FedMammoBench/`.
- **INC** = `inc-project-models-classification-detection-main/` (hermano de este repo; en la
  workstation vive bajo el mismo `BASE` que `FedMammoBench/`, ver §2). Todo lo del INC es dentro de
  `src/models/classification_images/` salvo que se diga otra cosa.

---

## 1. Resumen ejecutivo

- Se agregan **tres variantes de backbone** a los dos proyectos: **ResNet50 desde cero** (pesos
  aleatorios, todo entrenable), **ResNet18 desde cero** (ídem) y **ResNet18 con pesos ImageNet**
  (`IMAGENET1K_V1`, congelamiento configurable, **default: backbone 100% congelado**).
- En el INC se respeta su estilo (clase `XModel(nn.Module)` + rama `if/elif` + flag de argparse); en
  FMB, su arquitectura (una entrada por variante en `_ARCHITECTURES`, `build_model()` intacto).
- FMB cambia su carga de imágenes al **esquema del INC**: array → tensor → réplica a 3 canales →
  transforms sobre tensor (hoy FMB transforma el PIL de 1 canal y replica al final).
- El `run.sh` del INC se reescribe para lanzar las 3 corridas en la workstation `imagenesmedicas`;
  FMB recibe los 3 YAML pareados (exp61–63).

---

## 2. Restricción de entorno

- Se ejecuta en la **workstation `imagenesmedicas`** (la que tiene datos y GPU). Rutas verificadas
  en el árbol actual (`INC/.../run_pretrain_ablation_inc.sh`, `configs/exp17|18_*.yaml`):
  ```
  BASE    = /media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/13-PregradoJulian/Federal Learning/infraestructura federada
  REPRO   = $BASE/inc_repro_data              # csvs_norm_neg1_1/, csvs_norm_0_1/ (splits en formato INC)
  PYTHON  = /home/imagenesmedicas/miniconda3/envs/inc-combined/bin/python3   # entorno del INC
  IMGROOT = /media/imagenesmedicas/DATA1/01-ImagenesMedicas-US1/02-Databases/Mammo-Bench/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada_1a9410d8-9769-4064-a064-0160f2fd193d_DATASET-FILE_Mammo_Bench_zip_20241225112148174/Mammo_Data/Mammo-Bench/preproccesed_julian
  ```
  `IMGROOT` es el **padre** de `norm_0_1/` y `norm_neg1_1/`: los CSV ya traen ese prefijo en la ruta.
  Si alguna ruta no existe en la workstation, **detente y repórtalo** en el informe — no la inventes.
- ResNet18 ImageNet descarga `resnet18-f37072fd.pth` a `~/.cache/torch/hub/checkpoints/` la primera
  vez (necesita internet esa vez). Las variantes desde cero no descargan nada.
- FMB: intérprete `.venv/bin/python` (Python 3.12, ver `CLAUDE.md` §Entorno). No hay pytest: la
  verificación son scripts desechables con asserts (convención de `PHASES.md`), **fuera del repo**.
- TIFFs: `mode "F"`, `float32`, `(224, 224)`; `norm_neg1_1` en `[-1, 1]`, `norm_0_1` en `[0, 1]`
  (medido sobre `kau-bcmd`).

---

## 3. Lo que YA está resuelto / cerrado (no re-investigar)

Medido con torchvision en el `.venv` de FMB:

| backbone truncado (`children()[:9]`) | tensores de parámetros | `len(state_dict())` | `fc.in_features` |
|---|---|---|---|
| ResNet18 | **60** | **120** | **512** |
| ResNet50 | 159 | 318 | 2048 |

- **INC**: `models/image_models.py::ResNetModel` es ResNet50; `self.features = base_model.fc.in_features`
  y `get_model.py` construye el MLP con `input_size=image_model.features` → el 512 de ResNet18 se
  propaga solo. `--num_freeze N` congela los primeros N tensores de `model.parameters()`, así que
  **`--num_freeze 60` = ResNet18 100% congelado**, `0` = todo entrenable (159 era el equivalente de
  ResNet50 en la ablación).
- **INC**: hoy **no hay forma de construir un backbone sin pesos**: `get_model.py` hace
  `weigths_file = None if options.pretrained else options.path_image_model`, y `path_image_model`
  tiene un default (`ResNet50.pt`), así que sin `--pretrained` siempre carga un checkpoint.
- **INC**: `options.py` **no declara** `--normalize_mean/--normalize_std`, pero
  `run_pretrain_ablation_inc.sh:89` ya los pasa → esa corrida (exp17_inc) muere en argparse. Este
  brief los agrega (§4.A), lo que además arregla ese script sin tocarlo.
- **FMB**: `ArchitectureSpec.weights_from_factory=True` (`src/models/build.py`) ya salta
  `load_weights()`/`torch.load()` y solo trunca + congela. Por eso una variante "desde cero" es
  simplemente `model_factory=lambda: resnetXX(weights=None)` con ese flag en `True` — **no** hace falta
  otra rama en `build_model()`. Nunca usar el camino de checkpoint para "desde cero": `load_weights()`
  lanza si `matched == 0` a propósito (`CLAUDE.md`, "trampas").
- **FMB**: `ResNetFreezeStrategy.block_order` (`conv1, bn1, relu, maxpool, layer1..4, avgpool`) y el
  `target_layer_index=7` (`layer4`) de `src/interpretability.py` valen tal cual para ResNet18 (mismos
  hijos, mismo orden). No tocar.
- **FMB**: `ConfigurableMLPHead(in_features=2048)` por default → **todo YAML de ResNet18 debe poner
  `in_features: 512`**, o el primer forward falla por forma.
- **FMB, semántica de `unfreeze_from`**: `conv1` = descongelar **todo** (índice 0 de `block_order`);
  `none` = todo congelado. Ojo: el comentario de `configs/exp22_pretrain_ablation_imagenet_all.yaml:41`
  dice "backbone 100% congelado" junto a `unfreeze_from: conv1` — **el comentario está mal, el valor
  está bien** (exp22 es la corrida "_all"). No lo corrijas en esta vuelta; solo no lo copies.
- `torchvision.transforms.Normalize(mean=[m], std=[s])` sobre un tensor `(3, H, W)` funciona por
  broadcast (verificado). `RandomRotation`/flips/`GaussianBlur`/`Resize` aceptan tensores.

---

## 4. El trabajo pedido (en orden)

### 4.A — INC (`src/models/classification_images/`)

**A1. `models/image_models.py`**: importar `resnet18, ResNet18_Weights` junto a los demás, y agregar
una clase calcada de `ResNetModel` (mismo estilo de alineación y docstring):

```python
class ResNet18Model(nn.Module):
    """
    Model class for ResNet18 architecture.
    This class initializes the ResNet18 model without the final fully connected layer.
    It uses the torchvision implementation of ResNet18 (512 features tras el global pooling).
    The model is designed to be used as a backbone for further classification tasks.
    """

    def __init__(self, pretrained: bool = False):
        super(ResNet18Model, self).__init__()

        weights         = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base_model      = resnet18(weights=weights)
        self.features   = base_model.fc.in_features
        encoder_layers  = list(base_model.children())
        self.backbone   = nn.Sequential(*encoder_layers[:9])

    def forward(self, x):
        out = self.backbone(x)
        out = torch.flatten(out, 1)
        return out
```

En `get_image_model`, rama nueva `elif(model_name == "ResNet18"): base_model = ResNet18Model(pretrained=pretrained)`,
y el `ValueError` pasa a listar `'ResNet', 'ResNet18', 'DenseNet', or 'Inception'`. **`"ResNet"`
sigue siendo ResNet50** (no renombrar: lo usan run.sh existentes y los nombres de checkpoints de
`training.py:275`).

**A2. `options.py`**:
- `images_model_choices = ["Inception", "ResNet", "ResNet18", "DenseNet"]`.
- Junto a `--pretrained`:
  ```python
  parser.add_argument("--from_scratch",           help="Inicializar el backbone de imagen con pesos aleatorios (sin preentrenamiento)? Ignora --path_image_model. Incompatible con --pretrained", default=False, action="store_true")
  ```
- En la sección del dataloader:
  ```python
  parser.add_argument("--normalize_mean", type=float, default=None, help="Media para T.Normalize (1 valor, se difunde a los 3 canales replicados). None = sin Normalize. Default = %(default)s")
  parser.add_argument("--normalize_std",  type=float, default=None, help="Desviación estándar para T.Normalize (1 valor). None = sin Normalize. Default = %(default)s")
  ```

**A3. `models/get_model.py`**:
```python
    if options.pretrained and options.from_scratch:
        raise ValueError("--pretrained y --from_scratch son excluyentes.")

    # --pretrained usa los pesos ImageNet de torchvision y --from_scratch deja el backbone con
    # pesos aleatorios: en ambos casos se ignora el checkpoint de --path_image_model.
    weigths_file = None if (options.pretrained or options.from_scratch) else options.path_image_model
```
(El `SimpleNamespace` del `__main__` de ese archivo puede quedar como está.)

**A4. `dataloaders/dataloader_images.py`** (`classification_images`, **no** `classification_final`):
`Loader.__init__(self, images_dir, data_dir, augmentation=False, img_size=(224, 224), normalize_mean=None, normalize_std=None)`.
Construir una lista `normalize = [T.Normalize(mean=[normalize_mean], std=[normalize_std])] if normalize_mean is not None and normalize_std is not None else []`
y añadirla **al final** de `transforms_train` (`T.Compose([...flips/rotación..., *normalize])`) y de
`transforms_test` (`T.Compose([*normalize])`). Comentario corto: se aplica sobre el tensor ya
replicado a 3 canales; el valor único se difunde a los 3. `ImageDataset.__getitem__` **no cambia**.
En `training.py:95`, pasar `normalize_mean=options.normalize_mean, normalize_std=options.normalize_std`.

**A5. `run.sh`** — reescribir entero, estilo `run_pretrain_ablation_inc.sh` (`#!/usr/bin/env bash`,
cabecera de comentarios que explique qué corre y por qué, `set -uo pipefail`, `cd "$(dirname "$0")"`,
variables `BASE/REPRO/PYTHON/IMGROOT` de §2, función `run_exp` con `tee "$REPRO/${exp_name}.log"`).
Las tres corridas **en secuencia** (con backbone entrenable, una sola ya ocupa la GPU; se sigue con la
siguiente aunque una falle, y al final se sale con código ≠ 0 si alguna falló).

Receta compartida (= la de `run_pretrain_ablation_inc.sh`, para que sean comparables con esa
ablación): `--hidden_layers 1024 --output_size 2 --activation LeakyReLU --dropout 0.2 --img_size 224,224 --augmentation --n_epochs 200 --batch_size 64 --lr 1e-3 --b1 0.5 --b2 0.999 --patience_early 100 --min_lr 1e-6 --loss BCE --class_balance --neg_weight 0.7593 --pos_weight 1.4642 --train --result_dir "$REPRO/results" --images_dir "$IMGROOT" --wandb_project fedmammobench2.0 --wandb_group resnet_scratch_vs_resnet18 --tag_exp inc resnet_scratch_vs_resnet18`.

| exp_name | flags propios | CSVs |
|---|---|---|
| `exp61_resnet50_scratch_inc` | `--image_model ResNet --from_scratch --num_freeze 0` | `$REPRO/csvs_norm_neg1_1` |
| `exp62_resnet18_scratch_inc` | `--image_model ResNet18 --from_scratch --num_freeze 0` | `$REPRO/csvs_norm_neg1_1` |
| `exp63_resnet18_imagenet_inc` | `--image_model ResNet18 --pretrained --num_freeze "$NUM_FREEZE_R18" --normalize_mean 0.449 --normalize_std 0.226` | `$REPRO/csvs_norm_0_1` |

`NUM_FREEZE_R18="${NUM_FREEZE_R18:-60}"` arriba del script, con comentario: `60` = backbone ResNet18
100% congelado (default), `0` = todo entrenable, valores intermedios congelan los primeros N tensores
de parámetros (conv1+bn1 = 3; ver tabla de §3). Se sobreescribe con `NUM_FREEZE_R18=0 ./run.sh`.
Justificación de norm/normalize de exp63 en un comentario: ImageNet espera `[0,1]` + estadísticas
ImageNet promediadas a un canal (mismo razonamiento que la cabecera de `FMB/configs/exp17_*.yaml`).

**Criterio de aceptación A**: las 3 variantes hacen forward `(2,3,224,224) → (2,2)`; conteo de
`requires_grad` del backbone = 0 para exp63 con 60, = total para exp61/62; `--pretrained --from_scratch`
lanza `ValueError`; `bash -n run.sh` pasa; sin `--normalize_*` el tensor de salida de `Loader` es
**idéntico** al de antes del cambio.

### 4.B — FMB

**B1. `src/models/build.py`**: importar `resnet18, ResNet18_Weights`. Tres entradas nuevas en
`_ARCHITECTURES` (mismas `valid_prefixes` y `ResNetFreezeStrategy()` que las de ResNet50,
`key_remap={}`, `weights_from_factory=True`), cada una con comentario al estilo de las existentes:
- `"resnet50_scratch"`: `model_factory=lambda: resnet50(weights=None)`
- `"resnet18_scratch"`: `model_factory=lambda: resnet18(weights=None)`
- `"resnet18_imagenet_v1"`: `model_factory=lambda: resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`
  (ResNet18 **no tiene** `IMAGENET1K_V2` en torchvision — por eso solo `_v1`).

Ampliar el docstring de `ArchitectureSpec.weights_from_factory` (y el del módulo/`build_model`) para
que diga que `True` significa "el factory ya entrega los pesos iniciales definitivos —
preentrenados, o **aleatorios a propósito** (`*_scratch`)". Para `*_scratch`, `LoadReport.matched`
= `len(state_dict)` sigue siendo veraz (tensores del backbone, no "pesos preentrenados cargados");
decirlo en el comentario. **`build_model()` no se toca.**

**B2. `src/models/freeze.py`**: solo docstrings ("ResNet50" → "ResNet (18/50)") en
`ResNetFreezeStrategy`. `block_order` igual.

**B3. `src/datasets/dataset.py`** — carga al estilo INC. `__getitem__` queda:

```python
        row = self.df.iloc[idx]
        image = Image.open(row["abs_image_path"])

        if image.mode == "F":
            # <comentario: mismo razonamiento que hoy (PIL .convert() trunca floats en vez de
            # reescalar) + que ahora se sigue el esquema del INC
            # (classification_images/dataloaders/dataloader_images.py): array nativo -> tensor ->
            # réplica a 3 canales ANTES del transform, que opera sobre tensores.>
            array = np.array(image)[np.newaxis, ...]
            image_tensor = torch.from_numpy(array)
            image_tensor = torch.cat([image_tensor, image_tensor, image_tensor], dim=0)
        else:
            image_tensor = TF.to_tensor(image.convert("RGB"))  # [0, 255] -> [0, 1], como el ToTensor de antes

        image_tensor = cast(torch.Tensor, self.transform(image_tensor))
        label = int(row["label_norm"])
        return image_tensor, label
```
(`import torchvision.transforms.functional as TF`; mantener `.iloc`, nunca `.loc`.) Reescribir el
docstring de la clase para el orden nuevo. `_default_transform` = `Resize((224, 224))` sin `ToTensor`.

**B4. `src/datasets/transform.py`**: eliminar el Step 3 (`transforms.ToTensor()`); el pipeline recibe
ya un tensor. Actualizar docstrings de la clase y de `normalize_mean/std`: la réplica a 3 canales
ocurre ahora **antes** de `Normalize`, así que valen tanto 1 valor (broadcast) como 3; los configs
existentes (`[0.449]`, `null`, `(0.5,0.5,0.5)`) siguen siendo válidos **sin editarlos**. Borrar el
comentario del Step 2 que dice que este proyecto mantiene "PIL-first".

**B5. `src/config.py`**: añadir los 3 nombres nuevos al docstring de `ArchitectureConfig`
(y al de `weights_path`: prohibido/ignorado también para `*_scratch`).

**B6. Configs** `configs/exp61_resnet50_scratch.yaml`, `exp62_resnet18_scratch.yaml`,
`exp63_resnet18_imagenet.yaml` — partir de `configs/exp17_pretrain_ablation_imagenet.yaml` /
`exp18_*.yaml` (rutas de workstation, misma receta que el `run.sh` de A5 traducida con el mapa de la
cabecera de `exp57_camilo_centralizado.yaml`: `hidden_layers: [1024]`, `leakyrelu`
+ `negative_slope: 0.2`, `dropout: 0.2`, `num_classes: 2`, `cross_entropy` con
`weight: [0.7593, 1.4642]`, adam `lr 1e-3` betas `[0.5, 0.999]`, cosine `T_max 200 eta_min 1e-6`,
`batch_size 64`, augmentation de exp57, `patience 100`, `min_delta 0.005`). Cabecera de cada YAML =
receta + par INC (`exp6N_*_inc`) + por qué. Diferencias:

| | `architecture.name` | `unfreeze_from` | `in_features` | manifest | normalize |
|---|---|---|---|---|---|
| exp61 | `resnet50_scratch` | `conv1` (todo entrenable) | 2048 | `fedmammobench_norm_neg1_1.csv` | `null` |
| exp62 | `resnet18_scratch` | `conv1` | **512** | `fedmammobench_norm_neg1_1.csv` | `null` |
| exp63 | `resnet18_imagenet_v1` | `none` (default; comentario: `layer4`/`layer3`/…/`conv1` para descongelar) | **512** | `fedmammobench_norm_0_1.csv` | `[0.449]` / `[0.226]` |

Sin `weights_path`. `by_database_manifests` en la variante de normalización que corresponda.
`wandb_project: fedmammobench2.0`, `wandb_group: resnet_scratch_vs_resnet18`,
`metric_name: f1_macro` (**no** `f1`, ver trampa de `CLAUDE.md`), `run_dir`/`checkpoint_dir` en
`runs/exp6N_*`.

**B7. Documentación** (misma PR): tabla de arquitecturas de `.claude/context/code/MODELS.md` y de
`src/models/DOCS.md` (+ nota del 512); `.claude/context/code/DATASETS.md` y `src/datasets/DOCS.md`
(nuevo orden de carga/réplica/transform).

**Criterio de aceptación B**: ver §7.

---

## 5. Qué NO hacer

- No tocar `classification_final/` ni `classification_clinic/` del INC, ni `run_pretrain_ablation_inc.sh`.
- No renombrar `"ResNet"` (INC) ni `resnet50_radimagenet`/`resnet50_imagenet_v*` (FMB).
- No modificar `build_model()`, `load_weights()` ni el `raise` de `matched == 0`.
- No editar configs existentes (exp01–60) — incluido el comentario erróneo de exp22 (§3).
- No lanzar los entrenamientos completos de 200 épocas: solo la verificación de §7. El usuario lanza
  `run.sh` y los YAML después.
- No commitear ni pushear; dejar los cambios en el árbol de trabajo de cada repo.

---

## 6. Entregable

`docs/RESNET_SCRATCH_RESNET18_RESULTADOS.md` (en FMB) con:
1. Lista de archivos tocados por repo, con el diff o un resumen fiel de cada uno.
2. La salida literal (números) de cada verificación de §7.
3. Cualquier desviación de este brief, con su motivo. Cualquier ruta de §2 que no exista.

---

## 7. Verificación

Scripts desechables fuera de ambos repos; pegar salida en el informe.

1. **Paridad de carga FMB antes/después** (TIFF sintético `mode "F"` 224×224 en `[-1,1]` y otro en
   `[0,1]`, más un PNG RGB uint8): con el eval transform (`image_size: null`, sin augmentation), el
   tensor nuevo es **bit a bit igual** al que produce `HEAD` (usa `git stash` o una copia de los
   archivos originales) para `normalize null` y para `[0.449]/[0.226]`. Shape `(3,224,224)`,
   `float32`. Si `image_size` no es `null`, reportar `max|Δ|` (Resize sobre tensor vs PIL) — no es
   bloqueante, solo documentarlo.
2. **Paridad INC ↔ FMB**: el mismo TIFF produce el mismo tensor en `ImageDataset` (INC, sin
   augmentation, sin normalize) y en `MammoBenchDataset` (eval, normalize `null`): `torch.equal` es True.
3. **Smoke de modelos** en los dos repos, las 3 variantes: forward de `torch.randn(2,3,224,224)` →
   `(2,2)`. FMB: `LoadReport.matched` = 120 (R18) / 318 (R50); entrenables del backbone: exp63
   (`none`) = 0, exp61/62 (`conv1`) = total. INC: mismos conteos con `--num_freeze 60` / `0`.
   `resnet18_scratch` y `resnet18_imagenet_v1` tienen pesos distintos (no se colaron los de ImageNet
   en scratch).
4. `.venv/bin/python -c "import src.cli; import src.evaluate; import src.gradcam"` y
   `... import src.federated.server; import src.federated.client; import src.federated.evaluate_node"`.
5. Los 3 YAML nuevos cargan con el loader de `src/config.py` sin error de Pydantic (`extra="forbid"`).
6. INC: `bash -n run.sh`; `"$PYTHON" main.py --help` muestra `ResNet18`, `--from_scratch`,
   `--normalize_*`; `main.py ... --pretrained --from_scratch` falla con el `ValueError`.
7. Si hay GPU y datos: `main.py` con `--n_epochs 1` para cada una de las 3 variantes del INC y
   `src.cli` con `train.epochs: 1` (copia temporal del YAML fuera de `configs/`) para exp61–63 — que
   terminen sin excepción. Reportar tiempo por época.
