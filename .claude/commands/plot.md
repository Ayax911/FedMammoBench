# /plot — Genera gráficas para un experimento completado

Genera todas las gráficas de curvas de entrenamiento para uno o más directorios de ejecución.

## Uso
```
/plot [<nombre_exp_o_ruta>] [--dpi N]
```

**Ejemplos:**
- `/plot exp07_fedavg_resnet50` → busca `runs/exp07_fedavg_resnet50/`
- `/plot runs/exp12_fedavg_resnet50`
- `/plot exp07_fedavg_resnet50 exp08_centralized_resnet50`
- `/plot exp07_fedavg_resnet50 --dpi 150`

## Instrucciones

El argumento puede ser un nombre de experimento (se busca automáticamente bajo `runs/`) o una ruta completa. Si no se pasan argumentos, lista los directorios disponibles en `runs/` y pregunta cuál graficar.

Para **cada** directorio de run dado:

1. Resuelve la ruta: si el argumento no empieza por `/` ni `./`, prepend `runs/`. Verifica que el directorio existe; si no, informa y salta.

2. Ejecuta:
   ```bash
   venv/bin/python scripts/plot_experiment.py --run-dir <ruta> [--dpi <N>]
   ```
   El `--dpi` por defecto es 120. Si el usuario pasó `--dpi N`, úsalo.

3. Después de ejecutar, lista los archivos PNG generados en `<ruta>/plots/` con sus tamaños (usa `ls -lh`).

4. Informa qué tipo de run se detectó (centralizado / federado) basado en los archivos CSV presentes.

**Contexto del proyecto:** `scripts/plot_experiment.py` auto-detecta el modo (centralizado vs federado) según los CSV presentes. Para runs federados genera: `server_federated.png`, `nodes_train_loss.png`, `nodes_val_auc.png`, `nodes_val_f1.png`, `node_<N>_curves.png`. Para centralizados: `loss_curves.png`, `metric_curves.png`. Los plots van a `<run-dir>/plots/`.
