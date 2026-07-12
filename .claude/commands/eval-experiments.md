# /eval-experiments — Evalúa checkpoints de experimentos (por nodos o en mammobench)

Ejecuta post-hoc evaluation de checkpoints entrenados, con dos modos: evaluación local por particiones de nodos (útil para debugging), o evaluación global en todo mammobench (producción).

## Uso

```
/eval-experiments <exp> <checkpoint> [--mode nodes|mammobench] [--dataset ddsm|no-ddsm|all] [--output-base runs] [--predictions]
```

**Ejemplos:**
- `/eval-experiments exp17 runs/exp17_fedavg/exp17_fedavg/global_model.pt` → mammobench, todas las configs
- `/eval-experiments exp17 runs/exp17_fedavg/exp17_fedavg/global_model.pt --mode nodes` → por nodos (5 particiones)
- `/eval-experiments exp16 runs/exp16_centralized/exp16_centralized/final.pt --dataset no-ddsm --predictions` → solo mammo (sin DDSM), con predictions.csv
- `/eval-experiments exp15 --mode mammobench --dataset ddsm` → pregunta por el checkpoint si no se especifica

## Variables de entorno

Iguales a `/docker-run`:

| Variable | Qué contiene | Default |
|---|---|---|
| `REPO` | Raíz del repo | `pwd` |
| `MAMMO_DATA` | Carpeta con imágenes (`Preprocessed_Dataset/`) | **Pregunta al usuario** |
| `WEIGHTS_DIR` | Checkpoints preentrenados | `$REPO/weights` |
| `IMAGE_TAG` | Imagen Docker | `ayax911/federal-learning:latest` |

## Instrucciones

### 1. Detectar configs de eval disponibles

Para el experimento `<exp>` indicado:
- Si existe `configs/<exp>/eval/`, listar los YAML dentro (típicamente `mammo_bench.yaml`, `node0_partition.yaml`...`node4_partition.yaml`).
- Si no existe, avisar que no hay evaluaciones pre-definidas para este experimento — `configs/<exp>/eval/` es requerido.

Las configs pueden ser:
- **Por nodos:** `node<N>_partition.yaml` o `node<N>_<dataset>.yaml` (evalúa contra la partición local del nodo N)
- **Global:** `mammo_bench.yaml` o variantes (`mammo_bench_ddsm.yaml`, `mammo_bench_no_ddsm.yaml`)

### 2. Interpretar `--mode` y `--dataset`

**Modos de evaluación:**

| Flag | Significado | Configs a ejecutar |
|---|---|---|
| `--mode nodes` | Evaluación por partición local (debugging) | Todos los `node<N>_*.yaml` presentes |
| `--mode mammobench` (default) | Evaluación global en todo mammobench | Solo `mammo_bench*.yaml` |

**Filtrado por dataset (solo si `--mode mammobench`):**

| Flag | Qué checkpoints buscar / qué configs usar |
|---|---|
| `--dataset ddsm` | Incluye DDSM en la evaluación (si existe `mammo_bench_ddsm.yaml`, usar esa; si no, usar `mammo_bench.yaml`) |
| `--dataset no-ddsm` (default) | Excluye DDSM (buscar `mammo_bench_no_ddsm.yaml`; fallback a `mammo_bench.yaml` si no existe) |
| `--dataset all` | Ejecuta todos los `mammo_bench*.yaml` presentes (DDSM, no-DDSM, etc. en secuencia) |

Si `--mode nodes` se pasa, `--dataset` se ignora (cada nodo tiene su propia partición).

### 3. Validar el checkpoint

Confirma que el archivo `<checkpoint>` existe (ruta relativa a `$REPO`). Si no se pasa checkpoint, pregunta cuál usar — sugiere opciones basadas en el tipo de experimento (pretrain: `final.pt`, federado: `global_model.pt`).

### 4. Preparar directorios de salida

Crea `<output-base>/<exp>/eval/<config-name>/` para cada evaluación:
- `run.log` — log de ejecución
- `metrics.json` — métricas (accuracy, precision, recall, auc, etc.)
- `predictions.csv` (solo si `--predictions` se pasó)

Ejemplo: `--output-base runs --mode nodes` → `runs/exp17/eval/node0_partition/{run.log, metrics.json}`

### 5. Construir comandos `fedmammobench-evaluate`

Para cada config a ejecutar:

```bash
fedmammobench-evaluate \
  --config "configs/<exp>/eval/<config_name>.yaml" \
  --checkpoint "<ruta_checkpoint>" \
  --output-dir "runs/<exp>/eval/<config_name>" \
  [--predictions-out "runs/<exp>/eval/<config_name>/predictions.csv"]
```

Ejecutar en contenedor Docker:

```bash
docker run --rm --gpus all --network host \
  -v "$REPO/configs:/app/configs:ro" \
  -v "$REPO/manifests:/app/manifests:ro" \
  -v "$WEIGHTS_DIR:/app/weights:ro" \
  -v "$MAMMO_DATA:/app/data:ro" \
  -v "$REPO/runs:/app/runs" \
  "$IMAGE_TAG" \
  fedmammobench-evaluate \
    --config "configs/<exp>/eval/<config_name>.yaml" \
    --checkpoint "<checkpoint>" \
    --output-dir "runs/<exp>/eval/<config_name>" \
    [--predictions-out "runs/<exp>/eval/<config_name>/predictions.csv"]
```

En orden: nodos primero (si `--mode nodes`), luego configs de mammobench en el orden encontrado en el directorio.

### 6. Ejecutar en secuencia

Cada evaluación espera a completar antes de la siguiente (no paralelizar — GPU ocupada). Loguea resumen de éxitos/fallos al final.

### 7. Mostrar resumen

Al terminar, listar los archivos generados:
```
✓ runs/exp17/eval/node0_partition/ — 1.2k eval metrics
✓ runs/exp17/eval/mammo_bench_no_ddsm/ — 2.8k eval metrics (predictions omitted)
Log: runs/_logs/eval/eval_20260712_150330.log
```

## Casos de uso

**Caso 1: Evaluación federada global (producción)**
```bash
/eval-experiments exp17 runs/exp17_fedavg/exp17_fedavg/global_model.pt --mode mammobench --dataset no-ddsm
```
→ Evalúa el modelo agregado contra todo mammo sin DDSM.

**Caso 2: Debugging por nodos (¿por qué un cliente está atrasado?)**
```bash
/eval-experiments exp17 runs/exp17_fedavg/exp17_fedavg/global_model.pt --mode nodes
```
→ Evalúa el modelo contra cada partición de nodo por separado, ve dónde están los problemas.

**Caso 3: Pretrain centralizado con predictions**
```bash
/eval-experiments exp16 runs/exp16_centralized/exp16_centralized/final.pt --dataset no-ddsm --predictions
```
→ Evalúa, genera metrics.json + predictions.csv por imagen.

**Caso 4: Exploración completa de todas las configs de eval**
```bash
/eval-experiments exp07 --mode mammobench --dataset all
```
→ Ejecuta mammo_bench.yaml, mammo_bench_ddsm.yaml, mammo_bench_no_ddsm.yaml (todas las que existan).

## Notas

- Los comandos Docker reutilizan las mismas variables (`MAMMO_DATA`, `WEIGHTS_DIR`, `IMAGE_TAG`) que `/docker-run`, garantizando coherencia.
- Los YAMLs de eval en `configs/<exp>/eval/` ya especifican qué datasets y qué split (train/val/test) usar — este comando no sobreescribe eso, solo elige qué configs ejecutar.
- Chequeo de coherencia: si una config de eval no puede cargarse, se reporta y se sigue con la siguiente (no detiene).
- El output siempre se vuelca a `runs/_logs/eval/eval_<timestamp>.log` (master log compartido con `run-eval-queue.sh`) más los logs individuales en cada `<config_name>/run.log`.
