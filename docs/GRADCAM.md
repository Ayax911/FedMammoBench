# Grad-CAM: dónde se fija el modelo al clasificar

## Resumen ejecutivo

Nueva funcionalidad, no una auditoría: producir mapas de calor Grad-CAM sobre mamografías reales para
verificar en qué región de la imagen se apoya el modelo al predecir maligno/benigno. Dos archivos
nuevos, ninguno existente se modifica: `src/interpretability.py` (núcleo puro — hooks, cómputo del
mapa, overlay) y `src/gradcam.py` (entrypoint CLI — reconstrucción idéntica a `src/evaluate.py`,
selección de imágenes, guardado). El diseño ya está decidido (capa objetivo, cómo evitar
`@torch.no_grad()`, cómo alinear filas de `predictions.csv` con el manifest) — este brief es
implementación, no investigación.

## Restricción de entorno

Esta sesión (laptop) no tiene imágenes reales ni checkpoints entrenados de verdad, pero **eso no
bloquea la implementación**: todo el mecanismo de Grad-CAM (hooks, forma del mapa, normalización,
overlay) es verificable con un modelo real (`build_model`) sobre un tensor sintético, exactamente
igual que hizo `docs/AUDITORIA_SOBREAJUSTE.md`. La verificación con imágenes/checkpoints reales queda
para cuando el usuario lo corra en la workstation — no es requisito para cerrar esta vuelta.

## Lo que YA está decidido — no rediseñar, implementar tal cual

### Capa objetivo: `model[0][7]` (`layer4`), no `avgpool`

El backbone es `nn.Sequential` de 9 submódulos vía `truncate_backbone()`
(`src/models/weights.py:26-54`): `conv1(0), bn1(1), relu(2), maxpool(3), layer1(4), layer2(5),
layer3(6), layer4(7), avgpool(8)`. El modelo completo es `nn.Sequential(backbone, head)`
(`src/cli.py:116`, mismo contrato documentado en `src/federated/assembly.py`'s docstring de
`NodeAssembly.model`). `avgpool` (índice 8) ya colapsó H×W a 1×1 — ahí no queda nada que localizar.
La capa objetivo es **`model[0][7]`** (`layer4`), que con entrada 224×224 da activaciones
`[2048, 7, 7]`. Parametrizable (`target_layer_index: int = 7`) pero el default no se toca.

### El logit correcto: usar `LossSpec.probs()`, no ramificar BCE/CE a mano

`LossSpec.probs(outputs) -> Tensor[B]` (`src/train/build.py:134-163`) ya unifica sigmoid(1 logit BCE)
y softmax-columna-1(2 logits CE/focal) en "probabilidad de malignant" — es exactamente el escalar que
Grad-CAM necesita para el `.backward()`. **No reimplementar esa rama** dentro de
`src/interpretability.py`; recibir un `LossSpec` ya construido (`build_loss()`) como parámetro, igual
que hace `predict_on_loader()` (`src/train/evaluation.py:66`).

### Por qué no se puede reusar `evaluate()`/`predict_on_loader()`

Ambas llevan `@torch.no_grad()` (`src/train/evaluation.py:65`). Grad-CAM necesita gradientes
habilitados. `compute_gradcam()` (ver abajo) debe envolver su forward+backward en
`torch.enable_grad()` explícito y no debe tocar ni decorar esas dos funciones existentes.

### Alineación `predictions.csv` ↔ fila del manifest — el punto más fácil de arruinar

`save_predictions_csv()` (`src/reporting.py:97-117`) escribe solo `y_true,y_pred,y_prob` — **sin
ninguna columna de ID**. La alineación con una imagen concreta es exclusivamente posicional: la fila
`i` de `predictions.csv` corresponde a la fila `i` de `split.val_df()`/`split.test_df()` (los loaders
de val/test nunca usan `shuffle=True` ni `drop_last=True`, `src/datasets/build.py:81-82`, así que el
orden del `DataFrame` y el orden en que `predict_on_loader()` los recorrió son el mismo). Para que
esa alineación siga siendo válida al reconstruir todo en `src/gradcam.py`, la reconstrucción de
`Manifest`/`Split` debe ser **byte-idéntica** a la que produjo esos `predictions.csv`: mismo
`config.data.manifest_path`, mismo `config.data.image_root`, misma semilla — es decir, cargar
`run_dir/config.yaml` (el que `cli.run()`/`evaluate.py` ya persisten) con `load_config()`, nunca
pedirle al usuario que reescriba manualmente esos campos por CLI. Un join por columna (`ID_image`,
`patient_id`) sería releer mal el contrato — aquí la alineación es y debe seguir siendo por posición.

## El trabajo pedido

### 1. `src/interpretability.py` — núcleo puro

```python
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .train.build import LossSpec


@dataclass(frozen=True)
class GradCAMResult:
    """Resultado de un Grad-CAM sobre una sola imagen.

    Attributes:
        heatmap: array `[H, W]` float32 en `[0, 1]`, ya upsampleado a la
            resolución de `image` (bilinear, `align_corners=False`).
        target_score: el escalar de `loss_spec.probs()` sobre el que se hizo
            `.backward()` -- la probabilidad de malignant que el mapa explica.
    """
    heatmap: np.ndarray
    target_score: float


def compute_gradcam(
    model: nn.Sequential,
    image: torch.Tensor,
    loss_spec: LossSpec,
    device: str = "cpu",
    target_layer_index: int = 7,
) -> GradCAMResult:
    """Grad-CAM de una imagen sobre `model[0][target_layer_index]`.

    Args:
        model: `nn.Sequential(backbone, head)` con pesos ya cargados
            (mismo contrato que `evaluate()`/`predict_on_loader()`) --
            `model[0]` debe tener al menos `target_layer_index + 1` hijos.
        image: tensor `[1, C, H, W]` YA pasado por `eval_transform_builder`
            (sin augmentación) -- una sola imagen, no un batch. `C` es 3
            siempre (`MammoBenchDataset` ya expande TIFFs mode "F" a 3
            canales antes de esto, ver `src/datasets/dataset.py:89-106`).
        loss_spec: de `build_loss()` -- unifica BCE/CE, ver arriba.
        device: debe coincidir con el device de `model`.
        target_layer_index: índice dentro de `model[0]` (el backbone). 7 =
            `layer4`, ver "Lo que YA está decidido" arriba. NO usar 8
            (`avgpool`, sin resolución espacial).

    Returns:
        GradCAMResult.

    Raises:
        RuntimeError: si el gradiente de las activaciones es todo cero
            (ReLU muerta / score saturado) -- en ese caso `heatmap` es un
            array de ceros, NO se lanza excepción; documentar el caso, no
            dividir por `(max - min)` == 0 sin guardia (usar `+ 1e-8` en el
            denominador de la normalización).

    Example:
        >>> result = compute_gradcam(model, image_tensor.unsqueeze(0), loss_spec, "cpu")
        >>> result.heatmap.shape == (224, 224)
        True
    """
    ...


def overlay_heatmap(
    base_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    """Superpone `heatmap` (colormap 'jet') sobre `base_image`.

    Args:
        base_image: imagen PIL YA en modo RGB y YA en un rango visualizable
            (`uint8`, `0-255`) -- responsabilidad de quien llama (ver
            `src/gradcam.py` abajo, sección "imagen para mostrar ≠ imagen
            que ve el modelo"). Esta función NO reescala ni reinterpreta el
            rango de `base_image`.
        heatmap: `[H, W]` float32 en `[0, 1]` (la salida de
            `compute_gradcam().heatmap`), mismo `(H, W)` que `base_image`.
        alpha: peso del heatmap en la mezcla (`0` = solo `base_image`, `1` =
            solo el heatmap).

    Returns:
        Image.Image: composición RGB, mismo tamaño que `base_image`.

    Example:
        >>> composite = overlay_heatmap(original_pil.convert("RGB"), result.heatmap)
        >>> composite.save("gradcam_example.png")
    """
    ...
```

Implementación de `compute_gradcam` (mecánica exacta, no hay margen de diseño aquí):
1. `model.eval()`, pero envuelto en `torch.enable_grad()` explícito (el `@torch.no_grad()` global de
   otras funciones NO aplica acá porque esta función nunca se llama desde dentro de ellas).
2. `activations: torch.Tensor | None = None` capturado por un forward hook en
   `model[0][target_layer_index]`.
3. Gradiente de esas activaciones capturado con `register_full_backward_hook` (no el
   `register_backward_hook` viejo, deprecado y con semántica distinta en módulos con múltiples
   salidas) o, más simple, con `activations.retain_grad()` sobre el tensor capturado por el forward
   hook.
4. `model.zero_grad(set_to_none=True)`, forward pass `image.to(device)` → `logits = model(image)`,
   `score = loss_spec.probs(logits)[0]`, `score.backward()`.
5. `weights = grad.mean(dim=(2, 3), keepdim=True)` (`[1, C, 1, 1]`), `cam =
   F.relu((weights * activations).sum(dim=1))` (`[1, h, w]`).
6. Upsample a `image.shape[-2:]` con `F.interpolate(..., mode="bilinear", align_corners=False)`.
7. Normalizar a `[0, 1]`: `(cam - cam.min()) / (cam.max() - cam.min() + 1e-8)`.
8. `.detach().cpu().numpy()`, `.squeeze()`.

### 2. `src/gradcam.py` — entrypoint CLI

Reconstrucción **idéntica** a `run_evaluation()` (`src/evaluate.py:54-127`) salvo que:
- Solo usa `eval_transform_builder` (nunca `train_transform_builder` — Grad-CAM sobre una imagen
  aumentada/rotada no significa nada, ver `TransformBuilder` — construir `train_transform_builder`
  igual que `evaluate.py` haría porque `builder_dataloader()` lo exige, pero `src/gradcam.py` no
  necesita `builder_dataloader()` en absoluto: opera fila a fila sobre `split.val_df()`/`test_df()`
  directamente vía `MammoBenchDataset(df=..., transform=eval_transform_builder.build())`, sin
  `DataLoader`).
- Carga pesos con `load_checkpoint(model, checkpoint_path, device)` (mismo helper que
  `evaluate_checkpoint()`, `src/checkpoint.py`).

Firma de la función central:

```python
def run_gradcam(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    split_name: str,          # "val" | "test"
    select: str,              # "ids" | "misclassified" | "sample"
    image_ids: list[str] | None = None,     # requerido si select == "ids"
    n_per_class: int = 5,                   # usado si select == "sample"
    output_dir: str | Path | None = None,   # default: config.train.run_dir / "gradcam" / split_name
) -> None:
    ...
```

Selección de imágenes (los tres modos, deterministas):
- **`ids`**: filtrar `df[df["ID_image"].isin(image_ids)]` sobre `split.val_df()`/`test_df()` (la
  columna `ID_image` existe en el manifest, confirmado en la fila de ejemplo de
  `docs/AUDITORIA_IMAGENES.md`).
- **`misclassified`**: leer `run_dir/<split_name>/predictions.csv` con pandas,
  `df = split.val_df().reset_index(drop=True)` (o `test_df()`), verificar `len(df) ==
  len(predictions_df)` (si no calzan, `raise ValueError` — señal de que el checkpoint/config no
  corresponde a ese `predictions.csv`, no intentar adivinar), concatenar por posición
  (`pd.concat([df, predictions_df], axis=1)`), filtrar `y_true != y_pred`, ordenar por
  `abs(y_prob - 0.5)` descendente (los errores más "confiados", los más informativos) y tomar los
  primeros `n_per_class` de cada tipo de error (falso positivo, falso negativo).
- **`sample`**: primeros `n_per_class` de cada `label_norm` en el orden del `DataFrame` — determinista,
  sin `random.sample` (reproducible sin fijar semilla extra).

Por cada fila seleccionada:
1. `MammoBenchDataset` (transform=`eval_transform_builder.build()`) para obtener el tensor que ve el
   modelo → `compute_gradcam(model, tensor.unsqueeze(0), loss_spec, device)`.
2. **Imagen para mostrar ≠ imagen que ve el modelo**: volver a abrir `row["abs_image_path"]` con PIL
   por separado, SIN pasar por `transform` (para no mostrar un tensor normalizado a `[-1,1]`/`[0,1]`
   que se ve casi negro). Si `image.mode == "F"`: convertir a 8 bits con un estiramiento de contraste
   por percentiles (`np.percentile(arr, [1, 99])`, clip, reescalar a `[0, 255]`, `.astype(np.uint8)`),
   luego replicar a 3 canales — igual de simple que lo que ya hace `dataset.py` para el modelo, pero
   independiente. Si no es modo `"F"`: `.convert("RGB")` directo, igual que `dataset.py`.
   `Image.resize(config.data.image_size)` para que calce con `heatmap.shape`.
3. `overlay_heatmap(imagen_display, result.heatmap)` → guardar en
   `output_dir / f"{row['ID_image']}_true{row['label_norm']}_score{result.target_score:.3f}.png"`.

CLI (`argparse`, mismo patrón que `evaluate.py`):
```bash
.venv/bin/python -m src.gradcam --config configs/exp37_*.yaml --checkpoint runs/exp37_*/weights/best_epoch*.pt --split test --select misclassified --n-per-class 5
.venv/bin/python -m src.gradcam --config configs/exp37_*.yaml --checkpoint runs/exp37_*/weights/best_epoch*.pt --split test --select ids --image-ids CM000042,CM000107
```

### 3. Documentación

- Agregar `interpretability.py` y `gradcam.py` al árbol de `src/DOCS.md` (mismo formato que las
  entradas existentes de `config.py`/`evaluate.py`) y una sección "Cómo usar" para cada uno.
- Agregar la línea de import a la "Comprobación barata de que el árbol importa" de `CLAUDE.md`:
  `.venv/bin/python -c "import src.cli; import src.evaluate; import src.gradcam"`.
- **No** crear un `docs/GRADCAM_RESULTADOS.md` con heatmaps reales en esta vuelta — sin imágenes
  reales en esta sesión, no hay nada que mostrar; el informe de esta vuelta se limita a la
  verificación sintética de la sección siguiente.

## Qué NO hacer

- No tocar `evaluate()`/`predict_on_loader()` (`src/train/evaluation.py`) ni ningún `@torch.no_grad()`
  existente — Grad-CAM vive en su propio módulo con su propio `torch.enable_grad()`.
- No procesar batches completos de golpe dentro de `compute_gradcam()` — una imagen a la vez
  (`image.shape[0] == 1`), simplicidad sobre performance; si el usuario quiere Grad-CAM de un split
  completo, `src/gradcam.py` itera fila por fila.
- No implementar variantes (Grad-CAM++, Guided Grad-CAM, Score-CAM) — solo el Grad-CAM vainilla
  descrito arriba.
- No agregar Grad-CAM a `eval_pipeline.py` ni a la corrida normal de `cli.run()`/`evaluate.py` — es
  un análisis post-hoc separado, bajo demanda, nunca automático.
- No tocar `MetricsLogger`, W&B, ni ningún artefacto de `run_dir/` que no sea el nuevo subdirectorio
  `run_dir/gradcam/`.
- No inventar una cuarta forma de seleccionar imágenes fuera de `ids`/`misclassified`/`sample`.

## Entregable esperado

Los dos archivos de código (`src/interpretability.py`, `src/gradcam.py`), las actualizaciones de
`src/DOCS.md`/`CLAUDE.md`, y `docs/GRADCAM_RESULTADOS.md` con: qué se implementó, y el resultado de la
verificación sintética de abajo (no hace falta esperar a imágenes reales para entregar esta vuelta).

## Verificación (sin datos reales — ejecutable en esta laptop)

1. **Forma del heatmap**: construir un `model = nn.Sequential(backbone, head)` real vía `build_model()`
   + `get_head_strategy()` (arquitectura de juguete, ej. `resnet50_imagenet_v2` sin pesos reales) sobre
   un tensor sintético `torch.rand(1, 3, 224, 224)`, correr `compute_gradcam()`, confirmar
   `result.heatmap.shape == (224, 224)` y `0.0 <= heatmap.min()` y `heatmap.max() <= 1.0`.
2. **Backward real, no un mock**: confirmar que `activations.grad` (o el que capture el hook) no es
   `None` ni todo-cero tras `.backward()` — si lo es en el caso sintético, hay un hook mal puesto, no
   un caso legítimo de ReLU muerta.
3. **`overlay_heatmap()` no rompe con tamaños/dtypes reales**: PIL `Image.new("RGB", (224, 224))` +
   heatmap sintético `[224, 224]` con valores en `[0, 1]` (incluyendo un caso con `heatmap.max() ==
   heatmap.min() == 0`, el caso de gradiente muerto) → confirmar que no lanza excepción y devuelve
   `Image.Image` del mismo tamaño.
4. **Alineación posicional**: con un manifest de juguete (igual que la auditoría de sobreajuste,
   `Manifest`/`Split` sintéticos) y un `predictions.csv` de juguete escrito a mano con
   `save_predictions_csv()`, confirmar que el modo `select="misclassified"` de `run_gradcam()`
   selecciona exactamente las filas donde `y_true != y_pred` del `DataFrame` correcto — no filas
   desalineadas por un `merge`/`join` accidental.
5. `.venv/bin/python -c "import src.cli; import src.evaluate; import src.gradcam"` limpio.
