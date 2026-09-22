# Experimentos centralizados — resultados objetivos

Registro de lo que **efectivamente corrió**, con los números leídos de `runs/<exp>/test/metrics.json`
y `runs/<exp>/val/metrics.json`. Es un archivo de *resultados*, no de recetas: la receta de cada
experimento vive en el encabezado de su `configs/expNN_*.yaml`, que es el log experimental real.

Cómo se produce cada fila: `.venv/bin/python -m src.cli --config configs/expNN_*.yaml` reentrena y
re-evalúa el **mejor** checkpoint sobre val y test. Para re-evaluar sin reentrenar, `src.evaluate`.
Ver [../code/ARCHITECTURE.md](../code/ARCHITECTURE.md).

---

## El dataset detrás de todos los números

`manifests/fedmammobench_norm_{0_1,neg1_1}.csv` — 8.341 filas, split **6.671 train / 836 val / 834 test**,
paciente-disjunto (lo verifica `Split`, no lo genera).

| base | train | val | test | benignos | malignos | % maligno |
|---|---:|---:|---:|---:|---:|---:|
| cmmd | 3.776 | 474 | 472 | 2.414 | 2.308 | 48,9 % |
| kau-bcmd | 1.764 | 220 | 222 | 2.112 | 94 | **4,3 %** |
| cdd-cesm | 803 | 100 | 100 | 669 | 334 | 33,3 % |
| inbreast | 328 | 42 | 40 | 310 | 100 | 24,4 % |

Global 66/34 benigno/maligno — de ahí los `weight: [0.7593, 1.4642]` de las corridas ponderadas.

**kau-bcmd domina cualquier lectura por base.** Con 4,3 % de malignos, un modelo que nunca predice
maligno saca *accuracy* 0,94 y *specificity* 1,00 en esa base. Por eso su `f1`/`sensitivity` aparecen
en 0,0000 en casi todas las tablas de abajo pese a AUC alto: **el ranking es bueno, el umbral fijo de
0,5 es el que no sirve ahí**. Es exactamente el caso de uso de `scripts/calibrate_threshold.py
--by-database`. Cualquier conclusión sacada de accuracy/f1 por base, sin recalibrar, está mal leída.

---

## Tabla maestra (test, mejor checkpoint)

| exp | AUC | f1_macro | acc | sens | spec | loss | val AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| **exp37** hpsearch_v1 ganador | **0,9016** | **0,8071** | 0,8309 | 0,7220 | 0,8851 | 0,3887 | 0,9212 |
| exp38 hpsearch (head más profundo) | 0,8959 | 0,7861 | 0,7998 | 0,8231 | 0,7882 | 0,3788 | 0,9152 |
| exp31 antioverfit sin input-drop | 0,8889 | 0,7950 | 0,8201 | 0,7076 | 0,8761 | 0,4554 | 0,9083 |
| exp22 imagenet_all | 0,8872 | 0,7780 | 0,8118 | 0,6354 | 0,8995 | 0,4654 | 0,8992 |
| exp39 hpsearch (head más ancho) | 0,8732 | 0,7411 | 0,7506 | 0,8412 | 0,7056 | 0,4252 | 0,9066 |
| exp29 antioverfit wd=1e-2 | 0,8830 | 0,7853 | 0,8153 | 0,6643 | 0,8905 | 0,3824 | 0,9092 |
| exp30 antioverfit label-smooth | 0,8814 | 0,7701 | 0,8153 | 0,5596 | 0,9425 | 0,5565 | 0,9139 |
| exp28 antioverfit base | 0,8801 | 0,7931 | 0,8189 | 0,7004 | 0,8779 | 0,4171 | 0,9079 |
| exp04 inc_strict_replica | 0,8698 | 0,7307 | 0,8546 | 0,9062 | 0,5714 | 0,4093 | — |
| exp20 imagenet_layer4 | 0,8596 | 0,7600 | 0,7974 | 0,6065 | 0,8923 | 1,0970 | 0,8833 |
| exp32 antioverfit radimagenet | 0,8531 | 0,7464 | 0,7662 | 0,7329 | 0,7828 | 0,4400 | 0,8651 |
| exp03 frozen_backbone | 0,8501 | 0,7471 | 0,8546 | 0,8906 | 0,6571 | 0,4177 | — |
| exp59 pretrain_ablation_imagenet_512 | 0,8415 | 0,7351 | 0,7650 | 0,6462 | 0,8241 | 0,4603 | 0,8411 |
| exp60 pretrain_ablation_imagenet_256 | 0,8399 | 0,7385 | 0,7686 | 0,6462 | 0,8294 | 0,4611 | 0,8438 |
| exp17 imagenet (frozen) | 0,8396 | 0,7301 | 0,7674 | 0,5957 | 0,8528 | 0,4684 | 0,8438 |
| exp23 radimagenet_all | 0,8376 | 0,7390 | 0,7770 | 0,5957 | 0,8671 | 1,4802 | 0,8801 |
| exp11 layer4_bce_h1024 | 0,8374 | 0,7287 | 0,7602 | 0,6318 | 0,8241 | 0,7627 | 0,8308 |
| exp12 layer4_bce_h1024_512 | 0,8361 | 0,7113 | 0,7482 | 0,5884 | 0,8276 | 0,7735 | 0,8350 |
| exp13 layer4_bce_h1024_512_256 | 0,8361 | 0,7386 | 0,7638 | 0,6823 | 0,8043 | 0,6198 | 0,8369 |
| exp08 layer4_bce_posweight | 0,8356 | 0,7422 | 0,7686 | 0,6751 | 0,8151 | 0,7347 | 0,8416 |
| exp16 layer4_bce_h1024_dr05 | 0,8316 | 0,7234 | 0,7446 | 0,7040 | 0,7648 | 0,6096 | 0,8341 |
| exp21 radimagenet_layer4 | 0,8287 | 0,7212 | 0,7554 | 0,6101 | 0,8276 | 0,4899 | 0,8398 |
| exp58 pretrain_ablation_imagenet_2048_1024_1024_512_512 | 0,8287 | 0,7220 | 0,7518 | 0,6390 | 0,8079 | 0,4804 | 0,8469 |
| exp06 full_unweighted | 0,8279 | 0,7113 | 0,7482 | 0,5884 | 0,8276 | 0,4376 | 0,8148 |
| exp15 layer4_bce_h1024_dr04 | 0,8278 | 0,7271 | 0,7518 | 0,6787 | 0,7882 | 0,6238 | 0,8348 |
| exp33 imagenet_mismatched_norm | 0,8277 | 0,6804 | 0,6823 | 0,9134 | 0,5673 | 1,0488 | 0,8389 |
| exp14 layer4_bce_h1024_dr03 | 0,8274 | 0,7251 | 0,7506 | 0,6715 | 0,7899 | 0,6254 | 0,8349 |
| exp10 layer4_bce_2048_2 | 0,8253 | 0,7255 | 0,7578 | 0,6245 | 0,8241 | 0,5478 | 0,8388 |
| exp57 camilo_centralizado | 0,8213 | 0,7284 | 0,7398 | 0,8051 | 0,7074 | 0,4451 | 0,8192 |
| exp19 imagenet_mismatched_norm | 0,8192 | 0,7219 | 0,7434 | 0,7004 | 0,7648 | 0,4619 | 0,8408 |
| exp05 full_weighted | 0,8177 | 0,7220 | 0,7398 | 0,7329 | 0,7433 | 0,4567 | 0,8199 |
| exp18 radimagenet (frozen) | 0,7942 | 0,6584 | 0,7026 | 0,5162 | 0,7953 | 0,5245 | 0,8012 |
| exp07 fullfreeze_bce_posweight | 0,7921 | 0,6812 | 0,7146 | 0,5884 | 0,7774 | 0,7592 | 0,7963 |
| exp09 fullfreeze_bce_2048_2 | 0,7916 | 0,6444 | 0,7110 | 0,4188 | 0,8564 | 0,5223 | 0,7915 |

**No tienen `run_dir`** (nunca se ejecutaron en este árbol): exp02, exp34–36.
`exp01_resnet_layer4_test` y `Classification_Images_HL1024-512_lr1e-4_dr0.2` tienen carpeta pero no
`test/metrics.json`: son corridas de humo previas al pipeline de evaluación actual.

**exp58–60 ya corrieron** (ablación de tamaño de head sobre ImageNet pretraining, con el manifest
`fedmammobench_norm_0_1_local.csv`, ver §El fix del gotcha de rutas en
[CONFIG.md](../code/CONFIG.md)) — resultados en la tabla maestra de arriba. `labmirp` **no es otra
máquina**: es la misma workstation, solo un disco/ruta distinto (confirmado en
`docs/AUDITORIA_IMAGENES_FASE2_RESULTADOS.md`, 2026-09-22).

`exp03`/`exp04` no tienen `val/metrics.json`: son anteriores a que `cli.run()` re-evaluara val. Su
`accuracy` alta (0,8546) con `specificity` baja (0,57–0,66) no es comparable con el resto — corren
sobre el manifest del INC, 84 % maligno en train, no sobre `fedmammobench.csv`.

---

## Lecturas que ya están cerradas

**Barrido de hiperparámetros (exp37–39).** `sweeps/hpsearch_v1.yaml`, sampler bayesiano sobre
`val_auc`. exp37 es el ganador promovido a config permanente y tiene el mejor AUC de test medido por
imagen del proyecto: **0,9016 / f1_macro 0,8071**. exp38 (head más profundo) y exp39 (más ancho) son
el grid manual encima de exp37 y los dos empeoran — el head de exp37 ya está en su punto.

⚠️ **"Mejor" no es estadísticamente concluyente frente a exp28/exp31.** `docs/AUDITORIA_SOBREAJUSTE.md`
+ `docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md` (2026-09-20, H1 y H3) miden que el AUC por imagen infla
la certeza: al hacer bootstrap clusterizado por paciente (no por imagen) el ancho del IC del AUC de
exp37 crece **+42,7 %**, y en la comparación pareada por paciente **exp37 vs exp28 da p=0,933** — el
IC del delta `[-0,0073, +0,0625]` cruza el cero, es decir, **no hay evidencia estadística de que exp37
sea mejor que exp28**. Causa probable: exp37 se eligió maximizando `val_auc` sobre ~300 trials del
sweep contra un único split de val de 836 imágenes (~254 pacientes) — selección de modelo repetida
sobre la misma muestra, con un ruido de muestreo estimado σ≈0,019 que cubre buena parte del margen
entre exp37 (0,9212) y exp28 (0,9079) en val. Tratar el ranking de exp37/exp28/exp31 como empatado
hasta que la workstation corra multisemilla o k-fold (ver `docs/AUDITORIA_SOBREAJUSTE.md` §8).

**exp37 por base de datos** (`test/metrics_by_database.json`):

| base | AUC | f1_macro | sens | spec |
|---|---:|---:|---:|---:|
| kau-bcmd | 0,9414 | 0,4932 | **0,0000** | 1,0000 |
| cdd-cesm | 0,9127 | 0,8317 | 0,7879 | 0,8806 |
| inbreast | 0,8555 | 0,7949 | 0,6250 | 0,9375 |
| cmmd | 0,8306 | 0,7560 | 0,7348 | 0,7769 |

El mejor AUC por base es el de kau-bcmd y a la vez su sensibilidad es cero: el caso testigo de por
qué hay que recalibrar el umbral antes de reportar nada por base.

**Ablación de pretraining (exp17–23, 32, 33).** ImageNet gana a RadImageNet en cada par comparable
con este head y esta normalización: exp17 (0,8396) > exp18 (0,7942) congelados; exp20 (0,8596) >
exp21 (0,8287) con layer4; exp22 (0,8872) > exp23 (0,8376) descongelado entero. Cuánto se
descongela pesa más que qué pesos trae el backbone. exp19/exp33 son el control de normalización
cruzada — alimentar el backbone de ImageNet con píxeles en `[-1,1]` cuesta ~0,02 de AUC frente a
exp17, y exp33 además desplaza el modelo a sens 0,91 / spec 0,57.

**Anti-sobreajuste (exp28–32).** Las cuatro variantes quedan en 0,88–0,89 de AUC, empatadas dentro
del ruido; ninguna regularización extra bate a exp28 de forma clara. exp32 confirma otra vez que
cambiar a RadImageNet cuesta (~0,03 de AUC). El ensemble de las cuatro
(`runs/ensemble_exp28_29_30_31/`, generado por `scripts/ensemble_eval.py`) da **test AUC 0,9015** —
esencialmente igual a exp37 en solitario, y ese directorio solo escribe `auc`, no el resto de
métricas.

**Por base de datos en solitario (exp24–27).** Entrenar un modelo por base y no sobre el pool:
cdd-cesm 0,8227, cmmd 0,7202, kau-bcmd 0,9090 (con f1_macro 0,5768 — el mismo artefacto de umbral),
inbreast 0,5781 con val AUC 0,4906. inbreast y kau-bcmd **no tienen poder estadístico propio**
(40 y 222 muestras de test, 100 y 94 malignos en total); sus números en solitario no sostienen una
conclusión.

**exp57 (`camilo_centralizado`).** Réplica de los hiperparámetros del INC sobre `fedmammobench.csv`,
con `metric_name: f1` y `norm_neg1_1` para igualar el criterio del INC. Da 0,8213 — bastante por
debajo de exp37, que es el punto: la receta del INC no transfiere a este manifest.

**exp58–60** replican exp17 (ablación ImageNet, distintos tamaños de head) con `*_local.csv` y
`[0.449]`/`[0.226]`; ya corrieron, resultados en la tabla maestra de arriba. Ninguno bate a exp37 ni
a exp17 — ampliar/reducir el head sobre este pretraining no mueve la aguja.

---

Ver también: [FEDERATED.md](FEDERATED.md) para el contraste federado-vs-centralizado (y por qué el
AUC agregado del servidor no se compara contra esta tabla), y `docs/EXPERIMENTOS_CENTRALIZADOS.md` /
`docs/INFORME_EXP01_22.md` para la serie de notebooks borrada, incluida la falla de propagación de
etiquetas a nivel paciente en CMMD que deja un piso de ~0,44 de val-loss bajo todas aquellas.
