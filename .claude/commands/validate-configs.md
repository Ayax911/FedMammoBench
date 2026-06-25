# /validate-configs — Valida todos los configs YAML del proyecto

Carga y valida cada archivo de configuración del proyecto antes de disparar un experimento en GitHub Actions.

## Uso
```
/validate-configs [<glob_o_nombre>]
```

**Ejemplos:**
- `/validate-configs` → valida todos los configs (excepto `legacy/`)
- `/validate-configs exp12` → valida solo los archivos en `configs/exp12/`
- `/validate-configs configs/exp07/server.yaml`

## Instrucciones

### 1. Encontrar los archivos a validar

Si no se pasa argumento:
- Todos los `configs/**/*.yaml` **excepto** `configs/legacy/` y `configs/reference.yaml`

Si se pasa un argumento:
- Si es un nombre de experimento (ej: `exp12`), valida `configs/exp12/*.yaml`
- Si es una ruta completa a un `.yaml`, valida solo ese archivo

### 2. Validar cada archivo

Para cada YAML, ejecuta este script en el venv:

```python
import sys; sys.path.insert(0, 'src')
from fedmammobench.configs import load_config

try:
    cfg = load_config('<path>')
    cfg.validate()
    print(f"OK  {path}")
except Exception as e:
    print(f"FAIL {path}: {e}")
```

Ejecuta validando todos en un solo script (no uno por uno) para ser eficiente.

### 3. Verificaciones de coherencia servidor↔cliente

Para cada par `configs/<exp>/server.yaml` + `configs/<exp>/client.yaml`, verifica que estos campos coincidan:
- `federated.rounds`
- `training.local_epochs`
- `training.scheduler.t_max`
- `model.local_unfreeze_at_epoch`
- `model.unfreeze_layers`
- `model.freeze_backbone`
- `model.in_channels`
- `model.num_classes`
- `training.mixed_precision`

Si difieren, reporta como **WARN** (no es error fatal pero probablemente es un bug de configuración).

### 4. Verificaciones adicionales

Para cada config de modo federado (`mode: federated`):
- Si `model.weight_source == 'custom'`, verifica que `model.checkpoint_path` no está vacío
- Si `model.weight_source == 'custom'` y el checkpoint es `runs/*/final.pt`, informa si ese archivo no existe en disco (significa que el pretrain todavía no se ha ejecutado)
- Si `training.scheduler.t_max` está definido, verifica que coincide con `training.local_epochs`

### 5. Reporte final

Presenta un resumen:

```
VALIDACIÓN DE CONFIGS
═══════════════════════════════════════════════════════
  OK    configs/exp07/server.yaml
  OK    configs/exp07/client.yaml
  WARN  configs/exp07/server.yaml ↔ client.yaml: t_max difiere (server=15, client=8)
  FAIL  configs/exp09/server.yaml: Unknown weight_source 'custom' — checkpoint_path not set
  INFO  configs/exp12/server.yaml: checkpoint_path runs/exp07_pretrain_ddsm/final.pt no encontrado
───────────────────────────────────────────────────────
  Total: 8 configs — 6 OK, 1 WARN, 1 FAIL
```

Si todo pasa, termina con: ✓ Todos los configs son válidos. Listo para disparar GitHub Actions.

Si hay FAILs, lista exactamente qué hay que corregir antes de disparar el workflow.
