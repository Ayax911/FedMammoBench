# Experimentos federados — resultados objetivos

Registro de lo que corrió en `src/federated/` (Flower, gRPC real, sin simulación ni Ray). Los números
salen de `runs/<exp>/server/best.json` y `runs/<exp>/nodes/<node>/test/metrics.json`. El diseño y el
porqué de cada decisión están en `docs/FEDERATED_DESIGN.md`; cómo lanzar y qué contrato cumple cada
módulo, en [../code/FEDERATED.md](../code/FEDERATED.md).

---

## Antes de leer cualquier número: el AUC del servidor no es el AUC del modelo

`best.json` guarda `metric_value` = el **promedio de AUCs por nodo ponderado por muestras**, que
produce `strategies.weighted_average()`. **No es el AUC del pool de predicciones.** Media de métricas
≠ métrica de la unión, y acá la diferencia no es cosmética:

| | AUC | f1_macro |
|---|---:|---:|
| exp37 centralizado (test real) | 0,9016 | 0,8071 |
| exp46 ronda 12, **agregado ponderado** | 0,8647 | — |
| exp46 ronda 12, **pooled real** (`src.evaluate`) | **0,8471** | 0,7531 |

La brecha federado-vs-centralizado es de **~0,055 de AUC**, no de ~0,037. El promedio por nodo va
inflado porque kau-bcmd (4,3 % de malignos) aporta specificity casi perfecta de forma trivial.

Para obtener el número pooled: el checkpoint global se guarda con el mismo
`src.checkpoint.save_checkpoint()` que usan las corridas centralizadas, así que `src.evaluate` lo
carga directo contra el manifest completo, sin conversión. El YAML de esa evaluación es descartable y
no se commitea (ver `ae8c390`).

**Regla:** nunca cites una brecha federado-vs-centralizado desde `best.json`. Re-evalúa pooled.

---

## Topología

4 nodos, uno por base de datos, cada uno con su `configs/federated/<exp>/node_<base>.yaml`,
su propio `class weight` calculado sobre su distribución local, y su propio
`manifests/by_database/<base>_norm_0_1.csv`. El servidor nunca ve imágenes. Los tamaños locales son
los de la tabla de [CENTRALIZED.md](CENTRALIZED.md) — cmmd es ~57 % del total y kau-bcmd ~26 %, así
que la agregación ponderada por muestras la dominan esos dos.

`aggregation_scope: full` en todo el grid (se agrega el modelo entero, no solo el backbone).
`local_epochs: 2`, receta de arquitectura/head/optimizer heredada de exp37.

---

## Grid principal: estrategia × rondas (exp41–52)

`metric_value` = AUC agregado ponderado sobre val, en la ronda que `round_tracking` marcó como mejor.

| exp | estrategia | rondas | hparams | mejor ronda | AUC agregado |
|---|---|---:|---|---:|---:|
| **exp46** | fedprox | 30 | `proximal_mu: 0.1` | 12 | **0,8647** |
| exp43 | fedavg | 30 | — | 3 | 0,8610 |
| exp45 | fedprox | 20 | `proximal_mu: 0.1` | 9 | 0,8608 |
| exp44 | fedprox | 10 | `proximal_mu: 0.1` | 9 | 0,8544 |
| exp41 | fedavg | 10 | — | 7 | 0,8471 |
| exp42 | fedavg | 20 | — | 3 | 0,8460 |
| exp47 | fedadam | 10 | `eta: 0.01, tau: 1e-3` | 4 | 0,7328 |
| exp52 | fedyogi | 30 | `eta: 1e-3, tau: 0.01` | 30 | 0,7016 |
| exp48 | fedadam | 20 | `eta: 0.01, tau: 1e-3` | 6 | 0,6813 |
| exp49 | fedadam | 30 | `eta: 0.01, tau: 1e-3` | 5 | 0,6467 |
| exp51 | fedyogi | 20 | `eta: 1e-3, tau: 0.01` | 20 | 0,6431 |
| exp50 | fedyogi | 10 | `eta: 1e-3, tau: 0.01` | 10 | 0,5838 |

**fedprox > fedavg > fedadam > fedyogi, sin ambigüedad.** Los adaptativos (fedadam/fedyogi) pierden
entre 0,13 y 0,28 de AUC frente a fedprox incluso con eta/tau ya ajustados, y encima *empeoran* con
más rondas en fedadam (0,7328 → 0,6467 de r10 a r30). Este es un fine-tuning con backbone-lr ~2e-4:
el régimen para el que FedAdam/FedYogi fueron diseñados (entrenar desde cero, pseudo-gradientes
grandes) no es este.

### Las estrategias adaptativas con los defaults de flwr no dan "peor", dan muerto

Documentado en `6caafb6` y `93f472d`, y verificado en vivo antes de fijar los hparams de la tabla:

- **fedadam con `tau=1e-9` (default stock)** → `val_loss=NaN` en el modelo agregado desde la ronda 1,
  idéntico en cmmd *y* cdd-cesm (no es ruido de un nodo). `train_loss` salta de ~0,56 a 3,6–6,2 en
  cuanto se aplica el update del servidor. Causa: con pseudo-gradientes chicos, el denominador
  `sqrt(v) + tau` de Adam amplifica ruido de punto flotante en vez de normalizar señal.
  **Fix: `eta=0.01` (10× menor que el default 0,1) + `tau=1e-3`.**
- **fedyogi con `eta=0.01/tau=1e-3`** — los valores que *sí* estabilizaron fedadam — igual colapsa:
  las 4 bases llegan a `val_auc=0.5000` **exacto** en la ronda 6, con `val_loss` escalando a 30–52.
  Salida degenerada, no NaN, pero igual de inútil. La regla de Yogi
  (`v_t = v_{t-1} - (1-β₂)·sign(v_{t-1}-Δ²)·Δ²`, en vez del EMA de Adam) amplifica más ese ruido.
  **Fix: `eta=1e-3` + `tau=0.01`.**

Si ves NaN o un 0,5000 exacto en las primeras rondas: **mata la corrida y ajusta eta/tau**, no gastes
el presupuesto de rondas. Los valores buenos ya están en `configs/federated/exp47-52/server.yaml`.

---

## Grid de `proximal_mu` (exp53–56)

Motivado por exp46 siendo la mejor celda del grid anterior con el `mu=0.1` por defecto. Barrido
alrededor de ese valor, rondas fijas en 30 (0,1 ya medido en exp46, no se repite):

| exp | `proximal_mu` | mejor ronda | AUC agregado |
|---|---:|---:|---:|
| exp46 (referencia) | 0,1 | 12 | **0,8647** |
| exp55 | 0,2 | 25 | 0,8585 |
| exp56 | 0,5 | 29 | 0,8558 |
| exp54 | 0,05 | 7 | 0,8513 |
| exp53 | 0,01 | 7 | 0,8484 |

**El barrido no encontró nada mejor que el default.** `mu=0.1` se queda como valor de trabajo. Se ve
además que mu más alto (0,2/0,5, que atan más el modelo local al global) empuja la mejor ronda hacia
el final del presupuesto (25 y 29 de 30), mientras que mu bajo converge y se estanca temprano
(ronda 7). `mu=0` sería FedAvg exacto y no se probó: para eso ya está exp43.

---

## exp46 por nodo (test, mejor checkpoint global)

| nodo | AUC | f1_macro | acc | sens | spec |
|---|---:|---:|---:|---:|---:|
| cdd-cesm | 0,9109 | 0,8119 | 0,8300 | — | — |
| inbreast | 0,8164 | 0,8039 | 0,9000 | 0,5000 | — |
| cmmd | 0,8030 | 0,7064 | 0,7161 | — | — |
| kau-bcmd | 0,7608 | 0,4837 | 0,9369 | **0,0000** | — |

kau-bcmd otra vez: accuracy 0,937 con sensibilidad 0,000 y `f1 = 0.0`. Ese 0,937 es el modelo no
prediciendo ningún maligno, y su AUC 0,7608 es el peor de los cuatro nodos — es decir, en la base
donde el número ingenuo se ve mejor, el modelo está de hecho peor. Recalibrar umbral
(`scripts/calibrate_threshold.py --by-database`) antes de reportar cualquier cosa por nodo.

---

## exp40

`configs/federated/exp40_fedavg_full/` es la configuración plantilla de fedavg con `scope: full` que
documenta el formato; no tiene `runs/` propio. El grid ejecutado arranca en exp41.

---

Ver también: [CENTRALIZED.md](CENTRALIZED.md) para los números contra los que se compara esto, y
`docs/FEDERATED_DESIGN.md` para por qué Flower, por qué gRPC real y no simulación, y el mapeo de cada
decisión contra el dolor que arregla del paquete federado legacy ya borrado.
