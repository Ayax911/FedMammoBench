# /compare — Tabla comparativa de métricas entre experimentos

Extrae las métricas finales de dos o más experimentos y las presenta en una tabla de comparación lista para papers.

## Uso
```
/compare <exp1> <exp2> [<exp3> ...]
```

**Ejemplos:**
- `/compare exp07_fedavg_resnet50 exp09_fedavg_resnet50`
- `/compare exp08_centralized_resnet50 exp07_fedavg_resnet50 exp12_fedavg_resnet50`

Los nombres se resuelven bajo `runs/` automáticamente. También acepta rutas completas.

## Instrucciones

Para cada experimento en la lista:

1. Resuelve la ruta bajo `runs/` si no es absoluta. Si no existe, informa y excluye.

2. Detecta el tipo de run leyendo qué CSVs existen:
   - `metrics.csv` → **centralizado** (lee la última fila = última época)
   - `server_federated_metrics.csv` → **federado** (lee la última fila = última ronda)
   - Ambos → muestra centralizado y la línea federada por separado

3. Para runs **federados** lee `server_federated_metrics.csv`:
   - Columnas de interés: `round`, `roc_auc`, `f1`, `sensitivity`, `specificity`, `accuracy`, `loss`
   - Toma la **última fila** (ronda final) como resultado reportable
   - También extrae la **mejor ronda** (mayor `roc_auc`)
   - Lee `timing_summary.csv` para el tiempo total si existe

4. Para runs **centralizados** lee `metrics.csv`:
   - Columnas de interés: `epoch`, `val_roc_auc`, `val_f1`, `val_sensitivity`, `val_specificity`, `val_loss`
   - Toma la **última fila**

5. Presenta una **tabla Markdown** con:

   | Experimento | Tipo | Rondas/Épocas | AUC-ROC | F1 | Sensitividad | Especificidad | Accuracy | Loss | Mejor ronda |
   |-------------|------|---------------|---------|-----|-------------|--------------|---------|------|-------------|

   Valores con 4 decimales. Si un valor no está disponible, muestra `—`.

6. Presenta una segunda tabla con **timing** (si `timing_summary.csv` o `timing_summary` existe):

   | Experimento | Tiempo total | Tiempo/ronda | Nodos |

7. Al final, añade un párrafo corto con las observaciones más relevantes: cuál experimento tiene mejor AUC, si la sensitividad mejora con más épocas, etc.

**Nota metodológica:** Para comparaciones FedAvg vs FedProx, usa `task_loss` en vez de `loss` si está disponible (ver METHODOLOGY.md §6). Indica esto si detectas estrategias mixtas.
