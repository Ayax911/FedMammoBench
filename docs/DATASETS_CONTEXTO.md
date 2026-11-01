# Contexto de los conjuntos de datos (Mammo-Bench / FedMammoBench)

Conteos calculados directamente de `manifests/fedmammobench_norm_0_1.csv` (2026-09-30). Las cifras de
formato, rango y resolución provienen de `docs/AUDITORIA_IMAGENES_RESULTADOS.md` y
`docs/AUDITORIA_IMAGENES_FASE2_RESULTADOS.md` (medidas en la workstation con imágenes reales; esta
máquina no tiene los datos, así que no se re-verificaron aquí).

Tarea: clasificación binaria **benigno / maligno** de mamografías, a nivel de imagen.

---

## 1. Resumen por base de datos

| Base | Imágenes | Pacientes | Benignas | Malignas | % maligno | CC / MLO |
|---|---:|---:|---:|---:|---:|---|
| CMMD | 4.722 | 1.622 | 2.414 | 2.308 | 48,9 % | 2.361 / 2.361 |
| KAU-BCMD | 2.206 | 478 | 2.112 | 94 | **4,3 %** | 1.104 / 1.102 |
| CDD-CESM | 1.003 | 326 | 669 | 334 | 33,3 % | 500 / 503 |
| INbreast | 410 | 108 | 310 | 100 | 24,4 % | 203 / 206 (+1 sin vista) |
| **Total** | **8.341** | **2.534** | **5.505** | **2.836** | 34,0 % | |

Lateralidad (L / R): CMMD 2.432/2.290 · KAU-BCMD 1.101/1.105 · CDD-CESM 510/493 · INbreast 209/201.

## 2. Splits (la columna `split` del manifest; disjuntos por paciente)

Imágenes (benigna / maligna):

| Base | train | val | test |
|---|---|---|---|
| CMMD | 3.776 (1.930 / 1.846) | 474 (242 / 232) | 472 (242 / 230) |
| KAU-BCMD | 1.764 (1.682 / 82) | 220 (214 / 6) | 222 (216 / 6) |
| CDD-CESM | 803 (535 / 268) | 100 (67 / 33) | 100 (67 / 33) |
| INbreast | 328 (246 / 82) | 42 (32 / 10) | 40 (32 / 8) |
| **Total** | **6.671** | **836** | **834** |

Pacientes por split (train / val / test): CMMD 1.298 / 162 / 162 · KAU-BCMD 382 / 48 / 48 ·
CDD-CESM 261 / 33 / 32 · INbreast 86 / 11 / 11.

Consecuencia: **KAU-BCMD tiene solo 6 malignas en val y 6 en test; INbreast 10 y 8.** No hay poder
estadístico para métricas por base en esas dos (ver la memoria de diagnóstico de sobreajuste).

## 3. Metadatos clínicos disponibles (columnas del manifest)

Columnas: `ID_image, source_dataset, laterality, view, preprocessed_image_path, classification, density,
BIRADS, abnormality, molecular_subtype, raw_image_path, mask_path, ROI_path, x, y, radius, subject_age,
source_subjectID, original_source_path, patient_id, split`.

| Campo | CMMD | KAU-BCMD | CDD-CESM | INbreast |
|---|---|---|---|---|
| `subject_age` (media ± std, rango) | 48,3 ± 11,1 (17–87) | 48,9 ± 7,2 (35–84) | 50,3 ± 11,7 (18–90) | **no disponible** |
| `density` (A/B/C/D) | no disponible | 672/1.038/375/121 | 8/329/515/70 (81 NA) | 136/146/99/28 (1 NA) |
| `BIRADS` | no disponible | 1: 1.847 · 3: 265 · 4: 72 · 5: 22 | 1: 430 · 2: 160 · 3: 49 · 4: 123 · 5: 224 · 6: 4 (13 NA) | 1: 67 · 2: 220 · 3: 23 · 4: 43 · 5: 49 · 6: 8 |
| `abnormality` | mass 2.780 · calcification 648 · both 1.294 | no disponible | no disponible | no disponible |
| `molecular_subtype` | Luminal B 664 · Luminal A 278 · HER2 250 · triple neg. 140 (3.390 NA) | no disponible | no disponible | no disponible |

`mask_path` / `ROI_path` existen en el manifest pero **no se usan como fuente de imagen** en ningún
punto del código (auditado). `x, y, radius` (coordenadas de lesión) solo aparecen en algunas bases.

## 4. Formato de las imágenes

Hay tres variantes del mismo conjunto, con las mismas 8.341 filas y el mismo split:

| Manifest | Archivo | Modo PIL | Tamaño | Rango |
|---|---|---|---|---|
| `fedmammobench.csv` | `Preprocessed_Dataset/<base>/<base>_N.jpg` | `"L"` (8 bits, gris) | **resolución nativa, variable** | [0, 255] |
| `fedmammobench_norm_0_1.csv` | `norm_0_1/<base>/<base>_N.tiff` | `"F"` (float32, 1 canal) | **224×224** | [0, 1] |
| `fedmammobench_norm_neg1_1.csv` | `norm_neg1_1/<base>/<base>_N.tiff` | `"F"` (float32, 1 canal) | **224×224** | [-1, 1] |

- Los TIFF los genera `Preproccesed/preprocess_images.py` (redimensionados a 224×224 y normalizados en
  disco). Por eso sus configs usan `image_size: null`.
- Verificado en las 8.341 imágenes de cada variante: 0 archivos faltantes, 0 corruptos, 100 % en modo
  y rango esperados.
- En el `DataLoader` el tensor sale siempre como `(3, 224, 224)` `float32`: el canal único se replica a
  3 **antes** del transform (`src/datasets/dataset.py`). Etiquetas `int64`.
- Variantes `*_local.csv` (y `manifests/by_database/*_local.csv`): mismos datos, solo cambia
  `preprocessed_image_path` a la ruta absoluta de `labmirp`. El diff columna a columna dio 0 filas
  distintas.
- Otros manifests: `manifests/by_database/*` (un CSV por base) y
  `manifests/{cdd-cesm,cmmd,inbreast,kau-bcmd}-*-grouped.csv` (manifests originales por base, mismo
  número de filas).

### Resolución nativa (JPG, muestra de ~250 por base; W × H medios)

| Base | CC | MLO | Aspect ratio medio (W/H) |
|---|---|---|---|
| CDD-CESM | 1341 × 2269 | 1394 × 2394 | 0,58–0,59 |
| CMMD | 732 × 1780 | 815 × 2061 | 0,40–0,41 |
| INbreast | 1459 × 2933 | 1526 × 3256 | 0,47–0,49 |
| KAU-BCMD | 1143 × 1942 | 1246 × 2277 | 0,55–0,60 |

Ninguna es cuadrada; el resize a 224×224 **no preserva el aspect ratio** (en CMMD estira ~2,5× en
horizontal). Es una decisión conocida, no un bug. El rango completo de AR observado es 0,21–0,998.

### Estadísticas de píxel (muestra de 250 por base, `norm_0_1`)

| Base | Media | Std media |
|---|---|---|
| CDD-CESM | 0,212 | 0,183 |
| CMMD | 0,173 | 0,180 |
| INbreast | 0,371 | 0,261 |
| KAU-BCMD | 0,246 | 0,180 |

INbreast es más brillante (menos fondo negro por imagen), no un error de escalado. Esta diferencia de
distribución entre bases es lo que permite a un modelo "reconocer el hospital": ver la distinción AUC
pooled vs. intra-base en `docs/DECISION_PASAR_A_FEDERADO.md`.

## 5. Consideraciones para entrenar/evaluar

- **Desbalance por base**: KAU-BCMD (4,3 % maligno) vs. CMMD (48,9 %). Con umbral 0,5 fijo KAU-BCMD
  queda en ~0 de sensibilidad pese a AUC ~0,9; recalibrar umbral antes de reportar por base.
- **Predominio de CMMD**: 56,6 % de las imágenes y 64 % de los pacientes; domina el AUC pooled.
- **Varias imágenes por paciente** (≈3,3 de media): el split es paciente-disjunto, pero medir AUC por
  imagen infla la significancia; comparar a nivel paciente.
- **Reparto federado** (exp40–56): un nodo por base (ver `.claude/context/code/FEDERATED.md`).
- **Smoke test**: `manifests/mnist_smoketest/` (MNIST 64×64, paridad par/impar) usa el mismo esquema,
  sin relación clínica.
