# /new-exp — Crea un nuevo experimento a partir de uno existente

Copia la estructura de configuración de un experimento base, actualiza los campos de identidad (`name`, `output_dir`) y aplica los cambios que el usuario indique.

## Uso
```
/new-exp <id_nuevo> [--base <id_base>] [cambios adicionales en lenguaje natural]
```

**Ejemplos:**
- `/new-exp 13 --base exp12` → crea `configs/exp13/` copiando exp12
- `/new-exp 13 --base exp07 rounds=10 local_epochs=20`
- `/new-exp 13` → pregunta el base si no se especifica

## Instrucciones

### 1. Resolver el experimento base
Si `--base` no se pasa, lista los directorios en `configs/` (excluyendo `legacy/` y `base.yaml`) y pregunta cuál usar.

El id del nuevo experimento puede venir como `13`, `exp13` o `exp13_fedavg_resnet50`; normaliza a `expNN` para el directorio y completa el nombre si el base lo tiene (ej: base `exp12_fedavg_resnet50` → nuevo `exp13_fedavg_resnet50`).

### 2. Leer los YAML del base
Lee los archivos presentes en `configs/<base>/`:
- `server.yaml` (siempre)
- `client.yaml` (si existe)
- `pretrain.yaml` (si existe)

### 3. Identificar cambios necesarios
Siempre actualiza:
- `name:` → reemplaza el prefijo `expNN` por el nuevo número (preserva el sufijo, ej: `_fedavg_resnet50`)
- `output_dir:` → mismo patrón: `runs/expNN_...`

Aplica los cambios adicionales que el usuario especificó en lenguaje natural (ej: `rounds=10`, `local_epochs=20`, `weight_source: none`). Los aplica en **todos** los YAMLs donde el campo existe.

Verifica coherencia entre server.yaml y client.yaml: `rounds`, `local_epochs`, `t_max` del scheduler, y `local_unfreeze_at_epoch` deben ser iguales en ambos archivos (o el valor correcto propagado a los dos).

### 4. Crear los archivos
- Crea `configs/exp<NN>/` si no existe
- Escribe cada YAML con los campos actualizados. **No modifiques los comentarios** del base salvo que sean directamente sobre el campo cambiado; actualiza los comentarios relevantes (ej: si cambia `local_epochs`, actualiza el comentario que dice `8 épocas locales`).

### 5. Confirmar antes de escribir
Antes de crear los archivos, muestra un resumen de los cambios:
```
Creando configs/exp13_fedavg_resnet50/ desde configs/exp12_fedavg_resnet50/
  server.yaml:
    name: exp12_fedavg_resnet50 → exp13_fedavg_resnet50
    output_dir: runs/exp12_fedavg_resnet50 → runs/exp13_fedavg_resnet50
    training.local_epochs: 15 → 20
  client.yaml:
    (mismos campos)
```
Y pregunta confirmación.

### 6. Verificación final
Después de crear, ejecuta:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from fedmammobench.configs import load_config
cfg = load_config('configs/exp<NN>/server.yaml')
cfg.validate()
print('OK:', cfg.name)
"
```
Si `validate()` lanza error, muéstralo y corrige.

**Contexto del proyecto:** Los YAMLs del servidor tienen `data.name: none` (sin datos en el servidor). Los YAMLs del cliente apuntan a `manifests/node0_manifest.csv` como default. No modifiques estas rutas de datos salvo que el usuario lo pida explícitamente. El campo `checkpoint_path` en warm-start apunta a `runs/exp07_pretrain_ddsm/exp07_pretrain_ddsm/final.pt` — si el nuevo experimento también lo necesita, mantenerlo; si cambias `weight_source`, elimina o comenta `checkpoint_path`.

**Nota sobre `output_dir` anidado:** el código resuelve la ruta de salida como `<output_dir>/<name>`, y la convención del proyecto es fijar `output_dir: runs/<name_completo>` (idéntico a `name:`), así que las carpetas reales quedan anidadas — `runs/exp13_fedavg_resnet50/exp13_fedavg_resnet50/`, no `runs/exp13_fedavg_resnet50/` — como ya ocurre en todos los runs existentes (`exp07_pretrain_ddsm`, `exp12_fedavg_resnet50`, etc.). Es el comportamiento esperado, no un bug a corregir: al copiar el patrón del base y solo reemplazar el número/sufijo, el nuevo experimento queda anidado igual. Ten esto en cuenta al construir o verificar rutas de `checkpoint_path` que apunten a runs previos (siempre con el nombre duplicado al final).
