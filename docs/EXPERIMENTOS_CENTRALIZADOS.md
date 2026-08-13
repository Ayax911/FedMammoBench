# Experimentos centralizados exp01–exp19 (notebooks, `feature/radimagenet`)

Estado a 2026-08-09. Los 19 están ejecutados y sus resultados commiteados en `runs/`.

> **Lee primero la sección "El techo de 0.44".** Los resultados de la tabla no se pueden interpretar
> por su valor nominal: un problema de etiquetado en el manifest impone un piso común a las 19
> configuraciones y aplasta las diferencias entre ellas.

---

## 1. Qué es esta línea de trabajo

Ablaciones **centralizadas** de transfer learning que sirven de base para decidir la configuración del
modelo antes de llevarlo al escenario federado. Buscan responder tres preguntas:

1. ¿Cuántos bloques de ResNet50 conviene descongelar sobre pesos RadImageNet?
2. ¿Cabeza lineal o MLP con dropout?
3. ¿Ayuda un learning rate diferencial (backbone más lento que la cabeza)?

**No usan el paquete `fedmammobench`.** Son notebooks Jupyter autónomos: `torchvision.resnet50` + una
clase `Classifier` escrita a mano, envueltos en un `FullModel(backbone, classifier)` local. Nada del
sistema de configs, registries, estrategias ni `Trainer` interviene. Se ejecutan celda a celda en la
GPU de la workstation del laboratorio, fuera de Docker.

Consecuencia práctica: **las rutas están hardcodeadas a la workstation**
(`/media/imagenesmedicas/DATA1/...`) para `PROJECT_ROOT`, `DATASET_ROOT` y el checkpoint RadImageNet.
No resuelven en otra máquina y no respetan `MAMMO_DATA` / `WEIGHTS_DIR` /
`FEDMAMMOBENCH_RADIMAGENET_DIR`.

### Procedencia de los notebooks

| | exp01–exp09 | exp10–exp19 |
|---|---|---|
| Origen | escritos a mano | **generados** por `scripts/gen_lr_backbone_grid.py` |
| Semilla | ninguna | `SEED = 42`, `seed_everything()` |
| Learning rate | único (`LR`) | discriminativo (`LR` cabeza + `LR_BACKBONE`) |

Para modificar exp10–exp19 hay que **editar el generador y re-ejecutarlo con `--force`**, no tocar los
notebooks a mano; es la única forma de que el bloque siga siendo internamente consistente. exp01–exp09
son anteriores al generador y no deben regenerarse.

`configs/exp71`–`exp77` son un borrador anterior y **superado** de este mismo trabajo (sin celda de
parámetros, sin semilla, sin `metrics.json`, sin resultados en `runs/`). No forman parte de la serie.

---

## 2. Datos

`manifests/fedmammobench.csv` — 8821 imágenes, split por paciente **sin fuga** (0 pacientes
compartidos entre splits).

| split | imágenes | pacientes | Benign | Malignant |
|---|---|---|---|---|
| train | 7057 | 2149 | 3356 | 3701 |
| val | 884 | 270 | 421 | 463 |
| test | 880 | 268 | 425 | 455 |

Cuatro fuentes agregadas (Mammo-Bench): `cmmd` 5202 (59%), `kau-bcmd` 2206, `cdd-cesm` 1003,
`inbreast` 410. Vistas CC/MLO y lateralidades L/R balanceadas; 3.28 imágenes por paciente.

Lectura vía un `CSVDataset` local (no `datasets/`) sobre las columnas `split` / `classification` /
`preprocessed_image_path`.

**Columnas del manifest sin explotar:** `mask_path` (poblada al 100%), `abnormality` (59%), `BIRADS`
(41%), `density` (40%), `subject_age` (95%). `ROI_path`, `x`, `y`, `radius` están vacías.

---

## 3. Configuración común a los 19

| | |
|---|---|
| Backbone | ResNet50, pesos **RadImageNet** (`RadImageNet-resnet50.pth`) |
| Entrada | 224×224 RGB, `Normalize(mean=std=[0.5]*3)` — correcto para RadImageNet (= preset `radimagenet_rgb`) |
| Augmentación | `RandomRotation(10)` **y nada más** (sin flip horizontal) |
| Optimizador | AdamW, `weight_decay=1e-4` |
| Scheduler | `CosineAnnealingLR` |
| LR cabeza | 1e-3 |
| Batch | 32 |
| Épocas | 100 máx., early stopping con `PATIENCE = 20` sobre val_loss |
| Umbral | fijo en 0.5 (`logits.argmax`), sin calibrar |

Dos detalles que el paquete resuelve de forma centralizada y aquí están reimplementados a mano en cada
notebook (si se corrige uno, hay que corregirlo en toda la serie):

- `load_radimagenet_backbone` — remapea las claves `backbone.<N>.*` del checkpoint a los nombres de
  torchvision. Sin esto `load_state_dict(strict=False)` no casa ningún tensor y el modelo entrena
  desde init aleatorio.
- `freeze_bn_running_stats` — vuelve a poner en `eval()` las capas BatchNorm cuyos parámetros afines
  están congelados, para que no sigan actualizando `running_mean`/`running_var`.

---

## 4. Las 19 configuraciones

Sólo varían tres ejes: qué se descongela, el tipo de cabeza, y el LR del backbone.

| exp | blq | descongelado | cabeza | LR backbone | propósito |
|---|---|---|---|---|---|
| exp01 | 1 | frozen | MLP d=0.5 | – | profundidad × cabeza MLP 0.5 |
| exp02 | 1 | layer4 | MLP d=0.5 | – | " |
| exp03 | 1 | layer4+layer3 | MLP d=0.5 | – | " |
| exp04 | 2 | frozen | MLP d=0.3 | – | profundidad × cabeza MLP 0.3 |
| exp05 | 2 | layer4 | MLP d=0.3 | – | " |
| exp06 | 2 | layer4+layer3 | MLP d=0.3 | – | " |
| exp07 | 3 | frozen | lineal | – | profundidad × cabeza lineal |
| exp08 | 3 | layer4 | lineal | – | " |
| exp09 | 3 | layer4+layer3 | lineal | – | " |
| exp10 | 4 | frozen | lineal | — (congelado) | ancla sembrada de exp07 |
| exp11 | 4 | layer4+layer3 | lineal | 1e-3 | ancla sembrada de exp09 (LR único) |
| exp12 | 4 | layer4+layer3 | lineal | 3e-4 | barrido LR_BACKBONE (ratio 3×) |
| exp13 | 4 | layer4+layer3 | lineal | 1e-4 | " (ratio 10×) |
| exp14 | 4 | layer4+layer3 | lineal | 3e-5 | " (ratio 33×) |
| exp15 | 4 | layer4+layer3 | lineal | 1e-5 | " (ratio 100×) |
| exp16 | 5 | layer4 | lineal | 1e-4 | profundidad × cabeza con LR sano |
| exp17 | 5 | layer4 | MLP d=0.3 | 1e-4 | " |
| exp18 | 5 | layer4+layer3 | MLP d=0.3 | 1e-4 | " |
| exp19 | 5 | layer4+layer3+layer2 | lineal | 1e-4 | extiende profundidad un bloque más |

La cuarta celda de la rejilla 2×2 del bloque 5 (layer4+layer3 × lineal) es exp13.

---

## 5. Resultados

Métricas de test, evaluadas sobre el checkpoint de early stopping (`best_model.pth`).

| exp | ép. | best ép. | **min val_loss** | min train_loss | AUC | Acc | F1 | Prec | Rec |
|---|---|---|---|---|---|---|---|---|---|
| exp01 | 29 | 9 | 0.4423 | 0.3666 | .8501 | .7955 | .8185 | .756 | .892 |
| exp02 | 21 | 1 | 0.4430 | 0.0225 | .8548 | .8102 | .8318 | .768 | .908 |
| exp03 | 23 | 3 | 0.4365 | 0.0246 | .8524 | .7989 | .8188 | .766 | .879 |
| exp04 | 27 | 7 | 0.4438 | 0.3205 | .8411 | .7841 | .8085 | .747 | .881 |
| exp05 | 22 | 2 | 0.4295 | 0.0229 | .8581 | .8057 | .8295 | .759 | .914 |
| exp06 | 21 | 1 | 0.4488 | 0.0260 | .8504 | .7852 | .7930 | .790 | .796 |
| exp07 | 90 | 70 | 0.4441 | 0.4302 | .8355 | .7943 | .8120 | .770 | .859 |
| exp08 | 22 | 2 | 0.4322 | 0.0231 | .8630 | .7977 | .8130 | .779 | .851 |
| exp09 | 22 | 2 | 0.4199 | 0.0253 | .8626 | .8114 | .8296 | .778 | .888 |
| exp10 | 72 | 52 | 0.4438 | 0.4328 | .8357 | .7920 | .8119 | .763 | .868 |
| **exp11** | 21 | **1** | **0.4179** | 0.0244 | **.8658** | **.8148** | **.8372** | .767 | .921 |
| exp12 | 22 | 2 | 0.4233 | 0.0159 | .8539 | .8148 | .8362 | .770 | .914 |
| exp13 | 24 | 4 | 0.4370 | 0.0353 | .8482 | .8068 | .8244 | .778 | .877 |
| exp14 | 26 | 6 | 0.4382 | 0.2267 | .8454 | .7966 | .8137 | .773 | .859 |
| exp15 | 37 | 17 | 0.4390 | 0.3591 | .8428 | .8034 | .8226 | .771 | .881 |
| exp16 | 26 | 6 | 0.4436 | 0.0507 | .8482 | .8011 | .8190 | .773 | .870 |
| exp17 | 23 | 3 | 0.4398 | 0.0822 | .8494 | .8045 | .8277 | .761 | .908 |
| exp18 | 24 | 4 | 0.4381 | 0.0496 | .8493 | .8000 | .8170 | .775 | .864 |
| exp19 | 25 | 5 | 0.4406 | 0.0295 | .8529 | .8023 | .8157 | .787 | .846 |

### Lectura por eje

- **Rango total de AUC en 19 configuraciones: 0.8355 – 0.8658.** Amplitud de 0.03.
- **Profundidad de descongelado:** descongelar aporta ~+0.02 AUC frente a congelado, y ahí se acaba.
  Ni `layer4` vs `layer4+layer3` ni añadir `layer2` (exp19) se separan.
- **Tipo de cabeza:** sin efecto discernible. El bloque 5 entero cabe en .848–.853.
- **LR diferencial (bloque 4):** AUC monótonamente decreciente al bajar el LR del backbone
  (1e-3: .866 → 1e-5: .843). Aparentemente el LR diferencial *perjudica*. **Esta lectura no se
  sostiene** — ver abajo.
- **El seeding no alteró los resultados:** exp10 vs exp07 (misma config, sembrada vs no) da .8357 vs
  .8355; exp11 vs exp09 da .8658 vs .8626. Los bloques 1–3 y 4–5 son más comparables de lo esperado.
- **Sesgo sistemático hacia maligno en los 19:** precision .75–.79 frente a recall .85–.92. Del orden
  de 110–135 falsos positivos sobre 425 benignos, en todas las configuraciones.

---

## 6. El techo de 0.44

### El síntoma

| | rango entre los 19 |
|---|---|
| min **train**_loss | 0.016 → 0.433 (**27×**) |
| min **val**_loss | 0.4179 → 0.4488 (**0.4369 ± 0.0084**) |

El train_loss recorre dos órdenes de magnitud mientras el val_loss no se mueve. exp11 memoriza el
conjunto de entrenamiento (train 0.024) y exp07 apenas lo ajusta (train 0.430), y **ambos terminan en
el mismo val_loss**. Un techo insensible a la capacidad efectiva del modelo no es un problema de
optimización ni de regularización.

Además, `best_epoch` cae en 1–6 en todas las configuraciones con bloques descongelados. En exp11 —el
mejor del grid— el mínimo de validación está en la **época 1**, y a partir de ahí sube de forma
monótona (0.418 → 0.423 → 0.472 → 0.538 → 0.611 → 0.764) mientras el train_loss se desploma a 0.09.

### La causa: etiquetas propagadas a nivel de paciente en CMMD

La fuente `cmmd` (59% del dataset) asigna la etiqueta **por paciente, no por mama**:

- 826 pacientes CMMD tienen ambas lateralidades presentes. De ellos, 751 están etiquetados todo-maligno
  y **0 son mixtos**. En `kau-bcmd`, por contraste, hay 22 mixtos de 475 bilaterales: allí la etiqueta
  sí discrimina por mama.
- El cáncer bilateral sincrónico real ronda el 1–3% de los casos. Aquí sería el 91% de los bilaterales.
- Prueba definitiva: **745 de 745 de esos pacientes tienen un único `molecular_subtype` compartido por
  ambas mamas.** Un subtipo molecular caracteriza un tumor concreto; no puede ser el mismo en dos
  mamas independientes.

De ahí que **~1502 imágenes (17.0% del dataset) sean mamas contralaterales sanas etiquetadas
`Malignant`** — y, al ser el split por paciente y estratificado, exactamente el 17.0% de cada split.

### Cuadra con lo observado

Si esas contralaterales son visualmente indistinguibles de una benigna, el mejor clasificador posible
no puede hacer más que predecir la proporción de la mezcla:

```
piso de BCE estimado:      0.408 – 0.444   (según el supuesto sobre las malignas reales)
piso de val_loss observado:  0.4369 ± 0.0084   (mejor: 0.4179, exp11)
```

Explica los cuatro síntomas a la vez:

1. **El piso de 0.44** — es la entropía irreducible del ruido de etiqueta.
2. **El sobreajuste inmediato** — la señal generalizable se agota en 1–2 épocas; lo único que queda por
   aprender después es memorizar qué imágenes sanas concretas están marcadas como malignas, y eso por
   definición no transfiere.
3. **El sesgo hacia maligno** — al modelo se le enseñó en ~1200 imágenes de entrenamiento que una mama
   sana es maligna.
4. **Que ningún eje del grid se separe** — los tres chocan contra el mismo muro.

**Corolario favorable:** el test está igual de contaminado, así que las AUC de ~0.85 **subestiman** el
rendimiento real. Cuando el modelo acierta que una contralateral está sana, la métrica lo cuenta como
error.

### Lo que esto invalida

- Las conclusiones de los tres ejes del grid, incluida la del bloque 4 sobre el LR diferencial: lo que
  ese barrido mide en realidad es *en qué época toca cada régimen su primer mínimo antes de
  sobreajustar*, no la calidad del régimen.
- Cualquier barrido adicional de hiperparámetros sobre el manifest actual: no puede bajar del piso.
- Por extensión, la línea federada (exp32–49 y siguientes) arrastra el mismo sesgo, porque `cmmd` es un
  nodo entero del reparto — las conclusiones sobre heterogeneidad entre nodos están afectadas.

---

## 7. Siguientes pasos

1. **Prueba diagnóstica (barata, decisiva).** Re-correr exp11 excluyendo los 751 pacientes CMMD
   sospechosos de train y test. Si el piso de val_loss baja claramente de 0.44, el diagnóstico queda
   cerrado. Es un filtro de una línea sobre el manifest.
2. **Corrección real.** Reconstruir la etiqueta por mama desde el CMMD original, que publica la
   lateralidad de la lesión — información que existe en la fuente y se perdió al construir
   `fedmammobench.csv`. Recupera ~1500 imágenes bien etiquetadas en lugar de descartarlas.
3. **Re-correr el grid** una vez corregidas las etiquetas. Los tres ejes vuelven a ser medibles y es
   previsible que las diferencias, hoy aplastadas en 0.03 de AUC, se abran.
4. **Pendientes menores**, útiles pero que no rompen el techo por sí solos:
   - Flip horizontal en la augmentación (natural en mamografía; el manifest tiene `laterality`).
   - Umbral calibrado sobre validación y curva precision-recall, en vez del 0.5 fijo.
   - Baseline ImageNet y/o init aleatorio: hoy **no hay control** que justifique RadImageNet.
   - Réplicas multi-semilla (3 semillas sobre 2 configuraciones) para estimar la barra de ruido.
   - Explotar `mask_path`, y resolución mayor que 224px (a 224 las microcalcificaciones son
     sub-píxel).

---

## Referencias

- Notebooks: `configs/exp01/` … `configs/exp19/`
- Generador del bloque 4–5: `scripts/gen_lr_backbone_grid.py`
- Resultados: `runs/<RUN_NAME>/` → `metrics.json`, `loss_history.csv`, `plots/`
- Manifest: `manifests/fedmammobench.csv`
- Contexto del repo: `CLAUDE.md` (sección "Notebook experiments"), `configs/README.md`
