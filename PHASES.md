# Integración INC → FedMammoBench: bitácora por fase

Este documento registra, fase por fase, qué se portó del proyecto INC
(`inc-project-models-classification-detection-main/src/models/classification_images/`)
hacia `src/` de FedMammoBench, qué se modificó y qué se añadió. Análisis previo
completo (componente → impacto → propuesta) en el historial de la sesión que
originó esta rama; este archivo documenta la *implementación*, no la
justificación completa de cada decisión.

Convención de ramas: `develop` es la base de integración; cada fase vive en
`phaseN-<nombre>`, se implementa ahí y se mergea a `develop` con
`--no-ff` para dejar el límite de cada fase visible en `git log --graph`.
`main` no se toca hasta que `develop` esté validado.

---

## Fase 1 — Quick fixes: métricas, `drop_last`, determinismo cuDNN

Rama: `phase1-quick-fixes`. Sin dependencias de las fases siguientes — todo
lo demás se apoya en estos tres cambios.

### Modificado

- **[`src/metrics.py`](src/metrics.py)** — `build_metric_collection()` gana
  `"f1": BinaryF1Score()` y `"precision": BinaryPrecision()`. El proyecto INC
  selecciona su mejor checkpoint por F1 de validación
  (`classification_images/early_stopping.py`, ver Fase 3); sin F1 en la
  colección, `metric_name: f1` en el YAML no era ni siquiera expresable.
  `precision` es el equivalente de torchmetrics al `vpp()` que el INC calcula
  a mano (`TP/(TP+FP)`) — se reutiliza la métrica ya provista por la
  librería en vez de portar la función.
- **[`src/datasets/build.py`](src/datasets/build.py)** — `builder_dataloader()`
  añade `drop_last=True` al `DataLoader` de `"train"` (val/test sin cambios).
  `REFACTOR.md` §6 ya marcaba esto como obligatorio y nunca se implementó:
  la cabeza (`StandardMLPHead`) usa `BatchNorm1d`, que lanza excepción con un
  batch de una sola muestra.
- **[`src/seed.py`](src/seed.py)** — `set_global_seed()` gana un parámetro
  `cudnn_deterministic: bool = True` que fija
  `torch.backends.cudnn.deterministic = True` y `benchmark = False`. Sin
  esto, dos corridas con la misma semilla podían dar resultados distintos en
  GPU — necesario para poder atribuir cualquier diferencia numérica frente
  al baseline del INC a un cambio de código real, no a ruido de cuDNN.

### Añadido

Nada nuevo en esta fase — son extensiones puntuales de módulos existentes.

### Compatibilidad

Todos los cambios son aditivos con default que preserva el comportamiento
previo excepto `drop_last` (que corrige un bug de omisión, no introduce uno).
Ningún YAML existente deja de validar.

