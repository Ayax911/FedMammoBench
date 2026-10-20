# FIXES_FEDISIC2019_FLAMBY_REPRO — segunda vuelta: datos reales, resize/crop, riesgo FedAdagrad

Brief para Antigravity (ciclo `/antigravity`, Fase 1 — segunda vuelta). Es el `tarea` que sigue a
`docs/FEDISIC2019_FLAMBY_REPRO.md` → `docs/FEDISIC2019_FLAMBY_REPRO_RESULTADOS.md` (ya auditado, Fase
2, directamente en conversación — ver resumen en §2 de este archivo). Informe esperado:
`docs/FIXES_FEDISIC2019_FLAMBY_REPRO_RESULTADOS.md`.

---

## 1. Resumen ejecutivo

La primera vuelta dejó el grid de Fed-ISIC2019 estructuralmente completo y verificado (estrategias,
optimizer, loss, manifests, 49 YAML) pero con tres huecos reales que esta vuelta cierra: (a) el
preprocesado de imágenes deformaba el aspect ratio antes de recortar, a diferencia de FLamby — **ya
corregido en el código** (ver §2, nada que hacer acá salvo verificar); (b) ninguna celda corrió de
verdad contra datos reales — la vuelta anterior intentó descargar ISIC2019 en la máquina de
desarrollo (que no lo tenía) y terminó en un zip corrupto; **esta vuelta corre en la workstation,
que ya tiene el dataset** — no hay que descargar nada, solo apuntar `image_root` a la ruta real y
correr; (c) FedAdagrad muestra la firma de "corrida muerta" ya documentada en este repo y nadie lo
chequeó con datos reales. Nada de esto bloqueaba la primera vuelta (se dijo explícitamente que no lo
era) — ahora sí hay que cerrarlo antes de confiar en los resultados del grid.

---

## 2. Lo que YA está cerrado (no re-investigar)

Confirmado de forma independiente (no solo leído del informe anterior):

- `git diff --stat src/metrics.py src/reporting.py` vacío — sigue vacío, no tocar.
- Los 6 manifests de `manifests/fed_isic2019/*.csv` suman exactamente 23 247 filas, desglose por
  cliente idéntico a la Tabla 8 del paper.
- Las 49 YAML (`configs/federated/fedisic2019_*/`) tienen los hiperparámetros exactos de las Tablas
  10-16 por estrategia — ya verificado archivo por archivo.
- `src/federated/scaffold.py` y `cyclic.py`: implementación correcta (Karimireddy opción II; Cyclic
  con selección de 1 cliente/ronda) — leídos completos, sin cambios pedidos acá.
- W&B activado (`wandb_project: fedmammobench2.0`) en las 49 YAML, vía `scripts/fed_isic2019/
  gen_configs.py` — no tocar esa parte.
- Augmentations corregidas contra el código real de FLamby (`flamby/datasets/fed_isic2019/
  dataset.py`, clon en `~/snap/steam/FLamby`): `random_scale_range=(0.93,1.07)`,
  `brightness=0.15/contrast=0.1`, `affine_shear=0.1`, `crop_size=(200,200)` con `RandomCrop`
  (train)/`CenterCrop` (eval) ya en `src/datasets/transform.py` + `src/config.py` (`DataConfig.
  crop_size`, nuevo campo, aplicado en train Y eval) + los 3 call-sites (`cli.py`, `evaluate.py`,
  `federated/assembly.py`). Ningún config de mamografía usaba `random_crop`, cero regresión
  confirmada. **No volver a tocar estos valores**, ya están verificados byte a byte contra el repo
  oficial.
- El dataset **ya está en esta workstation** (el usuario lo confirmó) — el intento de descarga de la
  vuelta anterior fue en otra máquina (desarrollo, sin el dataset) y terminó en un zip corrupto de
  9.2 GB en `/home/akira/data/fed_isic2019/` + solo 100 imágenes de prueba; ignorar esa ruta, no es
  la de esta máquina (ver §4.A). El §3.9 del informe anterior ("Celda FedProx Completa con 6 Nodos...
  COMPLETED CLEANLY") **no corrió contra datos reales**: usó un stand-in sintético de 12 imágenes por
  cliente en `/tmp/fedprox_6nodes/`, con `alpha=[1]*8` (no el vector real) y YAMLs propios en `/tmp`
  (no los 49 de `configs/federated/`). Esa celda real nunca se ejecutó — es exactamente lo que el
  ítem §4.C de abajo pide cerrar.

---

## 3. Restricción de entorno

Esta vuelta corre en la **workstation real** (GPU + el dataset ISIC2019 ya descargado) — a diferencia
de la primera vuelta, que corrió en una máquina de desarrollo sin GPU dedicada ni dataset. No hace
falta descargar nada (ver §4.A); verificar igual, sin asumir, antes de arrancar:

- GPU disponible y memoria libre (`nvidia-smi`). El usuario reporta una sola GPU de 11 GB de VRAM
  para los 6 procesos de nodo simultáneos — con `batch_size: 64` (el del paper, no bajarlo sin que se
  pida explícitamente) puede no entrar; si hay OOM al lanzar los 6 a la vez, reportarlo con el
  mensaje exacto y la medición real de `nvidia-smi` en esta máquina (no extrapolar desde otra GPU) en
  vez de bajar `batch_size` por cuenta propia.
- Dónde vive realmente el dataset preprocesado en esta máquina (ver §4.A) — no asumir la ruta
  `/home/akira/data/fed_isic2019` de la máquina de desarrollo, no existe acá.

---

## 4. El trabajo pedido

### A — Apuntar `image_root` a los datos reales ya existentes en esta máquina (NO descargar)

**El dataset ISIC2019 ya está descargado y preprocesado en esta workstation** — no hay que bajar nada.
El intento de descarga de la vuelta anterior fue en la máquina de *desarrollo* (otra PC, sin el
dataset), terminó en un zip corrupto (`/home/akira/data/fed_isic2019/ISIC_2019_Training_Input.zip`,
9.2 GB truncado — `unzip -t` falla) y solo 100 imágenes de prueba; ignorar y no reintentarlo acá.

1. Localizar en ESTA máquina la carpeta `ISIC_2019_Training_Input_preprocessed/` (o las imágenes
   crudas, si el preprocesado de `scripts/fed_isic2019/preprocess_images.py` todavía no corrió sobre
   ellas aquí) — reportar la ruta absoluta real en el informe.
2. Si están preprocesadas: `find <carpeta> -type f | wc -l` y comparar contra las 23 247 filas de
   `manifests/fed_isic2019/*.csv` (deben coincidir 1:1 por `image`, no solo en conteo total — usar el
   mismo cruce por ID que ya se hizo en la vuelta anterior, no solo contar archivos).
3. Si están crudas (sin preprocesar): correr `scripts/fed_isic2019/preprocess_images.py` apuntando a
   ellas, confirmar el mismo cruce 1:1 al terminar.
4. Regenerar las 49 YAML con la ruta real: `scripts/fed_isic2019/gen_configs.py --image-root
   <ruta_absoluta_real_de_esta_workstation>` (reemplaza el default
   `/home/akira/data/fed_isic2019`, que es de la máquina de desarrollo y no existe acá). Confirmar que
   las 49 YAML cargan con el loader real (`load_server_config`/`load_node_config`) después de
   regenerar.

### B — Preprocesado: preservar aspect ratio antes del crop (el fix de resize/crop) — ✅ YA HECHO, no re-trabajar

**Este ítem ya está implementado y verificado — no es trabajo pendiente, queda documentado para que
se entienda qué cambió.** Confirmado de forma independiente (no solo leído de un informe): con un
tensor rectangular sintético (300×180), el `Resize` deja el lado corto exacto en 224px preservando
la proporción (0.6000 antes, 0.6005 después, la diferencia solo por redondeo a entero), y la salida
final del pipeline completo da `(3, 200, 200)`. Las 49 YAML ya tienen `image_size: 224` (entero) +
`crop_size: [200, 200]`, cargan sin error, y `batch_size` sigue en 64 (el del paper, sin tocar).

**El problema, verificado letra por letra contra el repo oficial de FLamby**
(`~/snap/steam/FLamby/flamby/datasets/fed_isic2019/dataset_creation_scripts/resize_images.py:61-77` +
su propio `README.md:48-51`): FLamby hace esto en DOS pasos con DOS tamaños distintos, a propósito —

1. **Preprocesado offline** (`resize_images.py`, una sola vez): redimensiona cada imagen para que el
   **lado corto** quede en 224px, preservando el aspect ratio (`Image.resize(new_size)` con
   `new_size` calculado por `ratio = 224 / min(ancho, alto)`) → las imágenes quedan **rectangulares**,
   cada una con alto distinto. El propio README lo dice explícito: *"in order to allow for
   augmentations, images aspect ratios are conserved in the preprocessing... As a result they cannot
   be batched without cropping them to a square."*
2. **En cada entrenamiento/evaluación**: SOLO ahí se recorta a cuadrado, `RandomCrop(200,200)` (train)
   / `CenterCrop(200,200)` (test), sobre esa imagen aún rectangular.

Lo que tiene hoy este repo (`scripts/fed_isic2019/preprocess_images.py` + `TransformBuilder`) hace el
paso 2 bien (`crop_size=(200,200)`, agregado en la vuelta anterior) pero **no** el paso 1: el
`Resize(image_size)` de `TransformBuilder` (Step 1 de `build()`) fuerza `image_size=(224,224)` como
tupla, lo que en `torchvision.transforms.Resize` significa "deformar a exactamente ese alto y ancho",
no "lado corto a 224 preservando aspecto". Para cuando el `crop_size=(200,200)` actúa, la imagen YA
está cuadrada y distorsionada — el recorte no tiene nada rectangular de donde recortar.

**El fix** (mínimo, aprovechando que `torchvision.transforms.Resize` ya soporta ambas firmas
nativamente — pasarle un `int` preserva aspecto con el lado corto en ese valor; pasarle una tupla
`(h, w)` fuerza esa forma exacta):

1. `src/config.py` — `DataConfig.image_size`: ampliar el tipo de `tuple[int, int] | None` a
   `tuple[int, int] | int | None`. Documentar en el docstring: `int` = preserva aspect ratio (lado
   corto a ese valor, como FLamby); tupla = fuerza esa forma exacta (comportamiento de siempre para
   mamografía, donde las imágenes YA vienen a tamaño fijo del preprocesado propio — **no cambiar el
   default ni el comportamiento para tuplas**).
2. `src/datasets/transform.py` — `TransformBuilder.__init__`: mismo ensanche de tipo en `image_size`.
   `build()` Step 1 no cambia de código (`transforms.Resize(self.image_size)` ya delega correctamente
   a la semántica de torchvision según si `self.image_size` es `int` o tupla) — solo el *type hint* y
   el docstring necesitan actualizarse.
3. `scripts/fed_isic2019/gen_configs.py`: cambiar `image_size: [224, 224]` → `image_size: 224` (entero,
   sin corchetes) en la plantilla del nodo. `crop_size: [200, 200]` queda igual. Regenerar las 49 YAML.
4. `scripts/fed_isic2019/preprocess_images.py`: confirmar que ya preserva aspect ratio con lado corto
   224 (si no lo hace, es el mismo fix que `resize_and_maintain()` de FLamby — `ratio = 224 /
   min(ancho, alto)`, aplicado a ambas dimensiones).

**Criterio de aceptación**: un JPG rectangular sintético (ej. 300×180 px) pasado por
`TransformBuilder(image_size=224, crop_size=(200,200), ...).build()` debe, tras el paso de `Resize`
(ANTES del crop), tener el lado corto en 224px y preservar la proporción (verificar con un hook o
partiendo el `Compose` en dos mitades para inspeccionar el tensor intermedio) — no debe llegar
exactamente 224×224 al paso de crop. La salida final tras el crop sigue siendo `(3, 200, 200)`, igual
que antes. Confirmar además que un config de mamografía existente (`image_size: [224, 224]`, tupla)
sigue comportándose IDÉNTICO a hoy — mismo test de paridad bit a bit que usó la vuelta
`RESNET_SCRATCH_RESNET18` (`docs/RESNET_SCRATCH_RESNET18.md §7.1`, mismo patrón).

### C — Re-correr las 7 celdas contra datos reales (no el stand-in sintético)

Una vez A esté cerrado (B ya lo está): correr al menos 1-2 rondas reales (no sintéticas, no en `/tmp`) de
**cada una** de las 7 celdas en `configs/federated/fedisic2019_*/`, con sus manifests y YAMLs reales
(no reconstruir nada en `/tmp`). Reportar, por celda: que terminó sin excepción, y la curva de
`train_loss`/`val_loss` de esas rondas (no hace falta que converja en 1-2 rondas, solo que sea un
número real y no NaN/inf).

### D — Riesgo FedAdagrad/FedAdam/FedYogi: ¿dead run con datos reales también?

`.claude/context/code/FEDERATED.md:118-124` ya documenta que estas tres estrategias, con `tau` chico
en un régimen de fine-tuning, dan NaN o `val_auc=0.5000` exacto ("corrida muerta", no "peor"). El
smoke test gRPC de la vuelta anterior sobre MNIST mostró exactamente esa firma con los hiperparámetros
del paper (`tau=1e-8`). Antes de confiar en esas 3 celdas:

1. Correr 3-5 rondas reales de `fedisic2019_fedadagrad_r10` (datos reales, de C arriba) y mirar
   `val_loss`/`val_auc` si ya hay métrica binaria disponible, o `val_loss` si no (multiclase, ver
   bypass de §5.D de `FEDISIC2019_FLAMBY_REPRO.md`). Si diverge/explota igual que en el smoke de
   MNIST: es consistente con el paper (`tau=1e-8` es el valor publicado para Fed-ISIC2019) y NO es un
   bug de esta implementación — documentarlo así, explícitamente, en el informe.
2. Repetir lo mismo para `fedadam`/`fedyogi`.
3. **No ajustar `eta`/`tau` por cuenta propia** para "arreglarlo" — esos valores son los publicados en
   la Tabla 10-16 del paper; si el paper mismo produce una corrida muerta con EfficientNet-B0 en este
   repo, es una observación legítima a reportar (quizás el paper936 lo tolera por otras razones del
   setup que no se replicaron, ej. su optimizer local es SGD con el mismo `tau` pero una arquitectura/
   batch distintos). Documentar el hallazgo, no silenciarlo ni "corregirlo" sin que el usuario lo pida.

---

## 5. Qué NO hacer

- No tocar `src/metrics.py` ni `src/reporting.py` (sigue vigente de la vuelta anterior).
- No tocar los valores de `random_scale_range`/`brightness`/`contrast`/`affine_shear` — ya están
  verificados contra el repo real, no hace falta re-derivarlos.
- No rediseñar Scaffold/Cyclic — ya están cerrados.
- No "arreglar" el resultado de FedAdagrad/FedAdam/FedYogi cambiando `eta`/`tau` sin que se pida —
  ver §4.D punto 3.
- No commitear ni pushear.

---

## 6. Entregable

`docs/FIXES_FEDISIC2019_FLAMBY_REPRO_RESULTADOS.md` con: la ruta real de los datos en esta workstation
y el conteo 1:1 contra los manifests (§4.A), las curvas de las 7 celdas reales (aunque sean solo 1-2
rondas), y el veredicto honesto sobre FedAdagrad/FedAdam/FedYogi (dead run confirmado o no, con los
números) — el fix de resize/crop (§4.B) ya está cerrado, solo hace falta re-confirmarlo si algo no
calza.

## 7. Verificación

1. `find manifests/fed_isic2019 -name "*.csv" | xargs wc -l` vs conteo real de imágenes en disco —
   deben coincidir.
2. Test de Resize con imagen rectangular sintética (ver criterio de aceptación de §4.B).
3. Paridad bit a bit de un config de mamografía existente antes/después del cambio de tipo de
   `image_size`.
4. 7 corridas reales de 1-2 rondas cada una, sin excepción, `rounds.csv`/`metrics.csv` con números
   reales (no placeholders).
5. `git diff --stat src/metrics.py src/reporting.py` sigue vacío.
