# RADIMAGENET_READINESS_REPORT.md — ¿Está el sistema preparado para integrar RadImageNet?

> **Pregunta**: tras los fixes C1/C2/C3, ¿cuánto avanzó `fedmammobench` hacia la integración de transfer learning con RadImageNet? ¿Qué bloqueadores siguen y cuál es el camino mínimo?
> **Respuesta corta**: **cero progreso técnico**. Los fixes C1/C2/C3 trabajaron exclusivamente en partitioning, training loop y NaN handling. Ningún archivo de `models/`, `configs/schema.ModelConfig`, ni `datasets/transforms.py` fue modificado. Todos los bloqueadores del audit inicial siguen en pie.

---

## 1. Estado actual confirmado (con `file:line`)

### Bloqueador 1 — `ModelConfig` no expone fuente de pesos

**Ubicación**: `src/fedmammobench/configs/schema.py:115-133`.

```python
@dataclass
class ModelConfig:
    name: Literal["resnet18", "efficientnet_b0"] = "resnet18"
    pretrained: bool = True
    dropout: float = 0.2
    num_classes: int = 2
    in_channels: int = 1
```

- Solo flag binario `pretrained`. No hay `weight_source: Literal["imagenet", "radimagenet", "custom"]`.
- No hay `checkpoint_path: str | None`.
- No hay `freeze_backbone: bool`, `freeze_head: bool`, ni `unfreeze_at_epoch: int | None`.

**Estado post-fix**: idéntico al pre-fix. Sin cambios.

### Bloqueador 2 — Builders hardcoded torchvision `Weights.DEFAULT`

**Ubicación**: `src/fedmammobench/models/resnet.py:28`, `src/fedmammobench/models/efficientnet.py:24`.

```python
# resnet.py:28
backbone = torchvision.models.resnet18(
    weights=ResNet18_Weights.DEFAULT if cfg.pretrained else None
)
```

- `ResNet18_Weights.DEFAULT` siempre devuelve los pesos ImageNet de torchvision.
- No hay rama para cargar checkpoint local ni para descargar desde URL externa.
- No hay validación de hash del checkpoint cargado.

**Estado post-fix**: idéntico.

### Bloqueador 3 — `Literal` del backbone limita opciones

**Ubicación**: `src/fedmammobench/configs/schema.py:129`.

```python
name: Literal["resnet18", "efficientnet_b0"] = "resnet18"
```

RadImageNet publica weights para:
- ResNet50 (no en el `Literal` actual)
- DenseNet121 (no soportado)
- Inception-V3 (no soportado)
- InceptionResNet-V2 (no soportado)

De los 4 backbones de RadImageNet, **ninguno** está soportado por el repo. ResNet18 y EfficientNet-B0 (los únicos soportados) **no** tienen weights publicados por RadImageNet.

**Estado post-fix**: idéntico.

### Bloqueador 4 — Channel adapter funciona, pero solo se invoca desde builders torchvision

**Ubicación**: `src/fedmammobench/models/_adapt.py:9-42`, llamado desde `resnet.py:32` y `efficientnet.py:35`.

El `adapt_first_conv()` está bien implementado (promedia los pesos RGB para producir un kernel de N canales). Pero solo se invoca dentro de los builders torchvision-centric. Cualquier checkpoint externo no pasaría por este adapter automáticamente.

**Estado post-fix**: idéntico.

### Bloqueador 5 — Normalize por canal no soportado en transforms

**Ubicación**: `src/fedmammobench/configs/schema.py:163-164` y `src/fedmammobench/datasets/transforms.py:37-42`.

```python
# schema.py
normalize_mean: float = 0.5    # ESCALAR
normalize_std: float = 0.25    # ESCALAR

# transforms.py
if grayscale:
    mean: tuple[float, ...] = (augmentation.normalize_mean,)
    std: tuple[float, ...] = (augmentation.normalize_std,)
else:
    mean = (augmentation.normalize_mean,) * 3   # Expande mismo escalar a 3 canales
    std = (augmentation.normalize_std,) * 3
```

Los escalares se replican uniformemente a N canales. Para ImageNet (R=0.485, G=0.456, B=0.406; std=0.229, 0.224, 0.225) este patrón es **incorrecto**. Para RadImageNet (que publica sus propias stats por canal) también lo es.

Aun para grayscale 1-canal, `(0.5, 0.25)` son valores genéricos arbitrarios — RadImageNet en grayscale tendría sus propios stats derivados de su corpus de entrenamiento.

**Estado post-fix**: idéntico. Bloqueante incluso si se carga el checkpoint correcto, porque la normalización aplicada sería inconsistente con la entrenada.

### Bloqueador 6 — Sin loader para checkpoints externos no-torchvision

**Ubicación**: `src/fedmammobench/utils/checkpoint.py`.

- `load_checkpoint()` (línea 50-71) llama `torch.load(path, map_location=...)` y aplica `model.load_state_dict(payload["state_dict"])`.
- Asume formato `{state_dict: ..., optimizer: ..., epoch: ..., extra: ...}`.
- RadImageNet publica weights originalmente en formato **Keras (.h5)**. Hay forks en formato PyTorch (`.pth`) en GitHub, pero los nombres de capas pueden no coincidir con torchvision.
- No hay conversor Keras→PyTorch en el repo.
- No hay key-renaming pipeline para mapear nombres torchvision↔terceros.

**Estado post-fix**: idéntico.

### Resumen tabla

| Bloqueador | Ubicación | Pre-fix | Post-fix | Δ |
|------------|-----------|---------|----------|---|
| ModelConfig sin weight_source | schema.py:115-133 | ❌ | ❌ | sin cambios |
| Builders hardcoded ImageNet | resnet.py:28, efficientnet.py:24 | ❌ | ❌ | sin cambios |
| Literal limita backbones | schema.py:129 | ❌ | ❌ | sin cambios |
| Channel adapter no parametrizado | _adapt.py | ❌ (funciona pero limitado) | ❌ | sin cambios |
| Normalize escalar | schema.py:163-164, transforms.py:37-42 | ❌ | ❌ | sin cambios |
| Sin loader checkpoints externos | utils/checkpoint.py | ❌ | ❌ | sin cambios |

---

## 2. Por qué los fixes C1/C2/C3 no movieron la aguja

Esto es **esperado y correcto** dado el scope que el usuario explicitó ("Implementa únicamente los hallazgos críticos C1, C2 y C3"). El audit lo nota únicamente para que quede explícito que la integración RadImageNet sigue requiriendo trabajo dedicado:

- **C1** modificó `datasets/partitioning.py` (partitioning logic) y `federated/client.py` (cómo se llama partition_indices). No tocó `datasets/transforms.py` ni `models/`.
- **C2** modificó `training/trainer.py` (proximal term), `federated/client.py` (capturar global_params), `federated/strategies/fedprox.py` (docstring). No tocó `models/`.
- **C3** modificó `federated/client.py` (NaN omitidos en evaluate). No tocó nada relacionado con pesos o normalización.

---

## 3. Cambios mínimos necesarios (estimación: 3-5 horas trabajo)

### Paso 1 — Extender `ModelConfig` (~30 min)

```python
# configs/schema.py
@dataclass
class ModelConfig:
    name: Literal["resnet18", "resnet50", "efficientnet_b0", "densenet121", "inception_v3"] = "resnet18"
    pretrained: bool = True                    # mantener por compatibilidad
    weight_source: Literal["imagenet", "radimagenet", "custom"] = "imagenet"  # NUEVO
    checkpoint_path: str | None = None         # NUEVO; solo si weight_source="custom"
    dropout: float = 0.2
    num_classes: int = 2
    in_channels: int = 1
    freeze_backbone: bool = False              # NUEVO
    freeze_head: bool = False                  # NUEVO
    unfreeze_at_epoch: int | None = None       # NUEVO; progressive unfreezing
```

Reutilizar `_from_dict` de `configs/loader.py` (ya maneja dataclass nesting recursivo).

### Paso 2 — Crear abstracción `WeightLoader` (~1 h)

Nuevo módulo `src/fedmammobench/models/weight_loaders.py`:

```python
class WeightLoader(Protocol):
    def load(self, model: nn.Module, cfg: ModelConfig) -> None: ...

class ImageNetLoader:
    """Default torchvision weights (current behavior)."""
    def load(self, model, cfg):
        # ya implementado en resnet.py / efficientnet.py
        pass

class RadImageNetLoader:
    """Load PyTorch port of RadImageNet weights."""
    def load(self, model, cfg):
        # 1. Resolver URL/path según cfg.name
        # 2. Descargar (con cache local en TORCH_HOME/checkpoints/)
        # 3. Cargar state_dict
        # 4. Renombrar keys si necesario (RadImageNet porting puede tener prefijos distintos)
        # 5. model.load_state_dict(remapped, strict=False) con log de missing/unexpected keys
        pass

class CustomCheckpointLoader:
    """Load arbitrary local checkpoint."""
    def load(self, model, cfg):
        if cfg.checkpoint_path is None:
            raise ValueError("weight_source='custom' requires checkpoint_path")
        # reuse fedmammobench.utils.checkpoint.load_checkpoint

_LOADERS = {
    "imagenet": ImageNetLoader(),
    "radimagenet": RadImageNetLoader(),
    "custom": CustomCheckpointLoader(),
}

def load_weights(model, cfg):
    return _LOADERS[cfg.weight_source].load(model, cfg)
```

Modificar `resnet.py` / `efficientnet.py` para construir el backbone con `weights=None` siempre, luego invocar `load_weights(model, cfg)`. Esto desacopla la fuente de pesos del builder.

### Paso 3 — Soportar normalize por canal (~30 min)

```python
# configs/schema.py
@dataclass
class AugmentationConfig:
    ...
    normalize_mean: float | tuple[float, ...] = 0.5
    normalize_std: float | tuple[float, ...] = 0.25
```

Modificar `transforms.py:37-42` para aceptar tanto float como tupla, validando longitud == `in_channels`.

Añadir presets en `configs/schema.py`:

```python
NORMALIZE_PRESETS = {
    "imagenet_rgb": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "radimagenet_rgb": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},  # verificar stats reales
    "grayscale_0_1": {"mean": (0.5,), "std": (0.5,)},
}
```

Permitir `normalize_preset: str | None` en `AugmentationConfig` que sobrescriba `normalize_mean/std`.

### Paso 4 — Soportar freeze / progressive unfreezing (~1 h)

Nuevo helper en `models/__init__.py`:

```python
def apply_freeze_policy(model: nn.Module, cfg: ModelConfig) -> None:
    if cfg.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("backbone.fc") and not name.startswith("classifier"):
                param.requires_grad = False
    if cfg.freeze_head:
        for name, param in model.named_parameters():
            if name.startswith("backbone.fc") or name.startswith("classifier"):
                param.requires_grad = False
```

Invocar desde el cliente (`FedMammoBenchClient.fit()`) considerando `cfg.training.unfreeze_at_epoch` y la ronda actual.

**Interacción con FedProx (N9)**: si el backbone está frozen, `self.model.parameters()` SIGUE incluyendo todos los params (frozen o no), así que `global_params` clona TODOS. Si queremos optimizar memoria, filtrar por `requires_grad`:

```python
global_params = [p.detach().clone() for p in self.model.parameters() if p.requires_grad]
```

Pero hay que asegurarse de iterar igual en el trainer.

### Paso 5 — Test unitario mínimo (~30 min)

```python
def test_radimagenet_weight_loading_smoke():
    cfg = ModelConfig(name="resnet50", weight_source="radimagenet", in_channels=1)
    model = build_model(cfg)
    # Smoke: forward pass produces correct shape
    x = torch.randn(2, 1, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 2)

def test_freeze_backbone():
    cfg = ModelConfig(name="resnet18", freeze_backbone=True)
    model = build_model(cfg)
    apply_freeze_policy(model, cfg)
    # Solo head debe tener requires_grad=True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    assert trainable < total / 100  # <1% trainable
```

---

## 4. Riesgos específicos RadImageNet + federación

Más allá de los bloqueadores estructurales, hay riesgos metodológicos cuando se combine RadImageNet con el flujo federado actual:

### Riesgo R1 — Catastrophic forgetting con non-IID + transfer learning

Pre-entrenamiento RadImageNet expone al modelo a ~1.35M imágenes de CT/RX/MR de diversa anatomía. Fine-tuning federado en mamografías altamente non-IID (Dirichlet α=0.5) puede causar olvido catastrófico de features útiles.

**Mitigación**: usar `freeze_backbone: true` en las primeras 5-10 rondas, luego `unfreeze_at_epoch` progresivo. Lo más sano: configurar FedProx con mu pequeño para anclar los pesos a la inicialización RadImageNet.

**Interacción con N7**: si se usa FedProx con AMP, el anchoring puede degenerar a FedAvg → catastrophic forgetting más probable. Validar AMP fix de N7 antes de esto.

### Riesgo R2 — BatchNorm statistics drift

Los running_mean/var de BatchNorm en RadImageNet reflejan la distribución de CT/RX/MR. Mamografías tienen distribución muy distinta. Después de la primera ronda federada, los running stats se desplazan; tras N rondas con FedAvg de stats, divergen del modo RadImageNet.

**Mitigación**: usar **FedBN** (que no agrega running stats), o cargar pero "resetear" los stats antes del primer round. Actualmente FedBN está como STUB en `strategies/fedbn.py` (NotImplementedError). Otra TODO.

### Riesgo R3 — Statistics drift por normalización inconsistente

Si se carga checkpoint RadImageNet pero se aplica normalize con `mean=0.5, std=0.25` (default actual), el input al primer conv está en una escala DISTINTA de la que el modelo fue pre-entrenado. Las activaciones iniciales pueden estar saturadas o muy fuera del rango, requiriendo muchas épocas de warm-up.

**Mitigación**: bloqueador 5 debe cerrarse ANTES de cargar weights RadImageNet.

### Riesgo R4 — Channel mismatch en grayscale

RadImageNet pre-entrenó en RGB (3 canales). El repo usa `in_channels: 1` por default para mamografía. El `adapt_first_conv` promedia los 3 canales a 1 — esto se valida en el repo, pero el promediado puede sub-óptimo. Alternativa: replicar mamografía a 3 canales y mantener el conv original (consume 3× memoria de input pero respeta los priors RGB).

**Mitigación**: ambas estrategias son válidas. Recomendar replicación 1→3 para fine-tuning inicial (preserva el adapter de RadImageNet) y luego experimentar con average si se busca eficiencia.

### Riesgo R5 — Interacción FedProx (N7-N10) + transfer learning

Si se usa FedProx para anclar a inicialización RadImageNet:
- Con N7 activo (AMP), el anchoring no funciona → drift no controlado.
- Con N10 (clip clipea task+proximal), grandes mu silencian el task gradient → el modelo no aprende mammografía, solo se queda en RadImageNet.

**Mitigación**: cerrar Fase 1.5 antes de cualquier experimento RadImageNet.

---

## 5. Plan de validación experimental para RadImageNet

Una vez los bloqueadores estructurales estén resueltos, el plan mínimo de validación científica:

### Experimento E1 — Baseline centralizado ImageNet

- `centralized_cbis_ddsm.yaml` (a crear), `weight_source: imagenet`, `freeze_backbone: false`, 50 epochs, seed fijo.
- Reporta: best val AUC, test AUC, train loss curves.

### Experimento E2 — Baseline centralizado RadImageNet

- Mismo config que E1 pero `weight_source: radimagenet`, `freeze_backbone: false`.
- Comparar test AUC vs E1. Si RadImageNet > ImageNet → señal positiva.

### Experimento E3 — Baseline centralizado RadImageNet con freeze progresivo

- Mismo que E2 pero `freeze_backbone: true` por 10 epochs, luego unfreezing.
- Esperar mejor convergencia early, similar final test AUC.

### Experimento E4 — Federado FedAvg con RadImageNet (sin FedProx)

- `fedavg_cbis_ddsm.yaml` con `weight_source: radimagenet`, mismo seed que E2.
- Comparar test AUC final vs E2. Δ esperado: degradación marginal por non-IID.

### Experimento E5 — Federado FedProx con RadImageNet

- Mismo que E4 con `strategy.name: fedprox`, `strategy.params: {proximal_mu: 0.01}`.
- **Pre-requisito**: N7 (AMP) y N8 (loss logging) cerrados.
- Comparar test AUC vs E4. FedProx debería mejorar en non-IID alto.

### Experimento E6 — Federado FedBN con RadImageNet

- Mismo que E4 con `strategy.name: fedbn`.
- **Pre-requisito**: implementar FedBN (actualmente STUB).
- Validar hipótesis de R2 (BN drift).

### Matriz de comparación

| Exp | Setup | Loss | Test AUC | Δ vs E2 |
|-----|-------|------|----------|---------|
| E1 | Cent ImageNet | ... | ... | baseline |
| E2 | Cent RadImageNet | ... | ... | upper bound |
| E3 | Cent RadImageNet + freeze | ... | ... | sanity check |
| E4 | Fed FedAvg RadImageNet | ... | ... | degradación non-IID |
| E5 | Fed FedProx RadImageNet | ... | ... | mejora vs E4? |
| E6 | Fed FedBN RadImageNet | ... | ... | mejora vs E4? |

Reportar todos los runs con mismo seed, mismo manifest hash, mismo commit SHA del repo.

---

## Conclusión

**El sistema NO está preparado para RadImageNet hoy. Cero progreso vs pre-fix.**

Los 6 bloqueadores estructurales (config schema, builders, backbone literal, channel adapter, normalize, checkpoint loader) siguen abiertos. Esto es esperado dado que los fixes C1/C2/C3 trabajaron en otras capas del sistema.

**Trabajo mínimo para habilitar RadImageNet experimental**: 3-5 horas según secciones 3.1-3.5.

**Trabajo mínimo para experimentos publicables con RadImageNet**: 3-5 horas + Fase 1.5 (cerrar N1, N7) + Implementar FedBN si se quiere comparar contra ese baseline (otras 2-4 horas).

**Estimación realista hasta paper-quality RadImageNet results**: 1-2 semanas de trabajo dedicado más experimentación.
