# Informe: configuraciones y resultados — exp01 a exp22

Ablation centralizado de transferencia RadImageNet → Mammo-Bench sobre ResNet-50, previo al
pipeline federado del proyecto. Cubre los tres primeros bloques de la serie de notebooks
(`configs/exp01`–`configs/exp22`), todos con resultados commiteados en `runs/`.

## Metodología común a las 22 corridas

- **Arquitectura:** ResNet-50 con warm-start RadImageNet (backbone) + cabeza clasificadora
  (lineal o MLP según corrida).
- **Dataset / manifest:** `manifests/fedmammobench.csv`.
- **Test set:** 880 imágenes, **fijo e idéntico** en las 22 corridas (verificado por la suma de
  cada matriz de confusión).
- **Optimizador / scheduler:** AdamW + CosineAnnealingLR en todas.
- **Loss:** CrossEntropyLoss en 21 de 22 corridas — la única excepción es **exp22**
  (BCEWithLogitsLoss, cabeza de una sola salida).
- **Selección de checkpoint:** mínimo val_loss (early stopping con `patience=20`,
  `num_epochs` máximo = 100) en todas.
- Ninguna corrida usa seed fija salvo **exp10** (seed=42, repetición de control de exp07) y
  el bloque 4/5/6 posteriores (fuera de este informe).

---

## Bloque 1 (exp01–exp10) — profundidad de descongelamiento × cabeza × dropout

LR único = 1e-3 para toda capa entrenable (sin diferenciar backbone de cabeza). Batch size 32,
image size 224px, `RandomRotation(10)`.

| Exp | Descongelado | Cabeza | Dropout | AUC | F1 | Acc | Precisión | Sensibilidad | Época mejor / máx |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp01 | Congelado | MLP | 0.5 | 0.8501 | 0.8185 | 0.7955 | 0.7561 | 0.8923 | 9 / 29 |
| exp02 | layer4 | MLP | 0.5 | 0.8548 | 0.8318 | 0.8102 | 0.7677 | 0.9077 | 1 / 21 |
| exp03 | layer4+layer3 | MLP | 0.5 | 0.8524 | 0.8188 | 0.7989 | 0.7663 | 0.8791 | 3 / 23 |
| exp04 | Congelado | MLP | 0.3 | 0.8411 | 0.8085 | 0.7841 | 0.7467 | 0.8813 | 7 / 27 |
| exp05 | layer4 | MLP | 0.3 | 0.8581 | 0.8295 | 0.8057 | 0.7591 | 0.9143 | 2 / 22 |
| exp06 | layer4+layer3 | MLP | 0.3 | 0.8504 | 0.7930 | 0.7852 | 0.7904 | 0.7956 | 1 / 21 |
| exp07 | Congelado | Lineal | — | 0.8355 | 0.8120 | 0.7943 | 0.7697 | 0.8593 | 70 / 90 |
| exp08 | layer4 | Lineal | — | 0.8630 | 0.8130 | 0.7977 | 0.7787 | 0.8505 | 2 / 22 |
| exp09 | layer4+layer3 | Lineal | — | 0.8626 | 0.8296 | 0.8114 | 0.7784 | 0.8879 | 2 / 22 |
| exp10 | Congelado (seed=42) | Lineal | — | 0.8357 | 0.8119 | 0.7920 | 0.7625 | 0.8681 | 52 / 72 |

**Lectura del bloque:** congelar el backbone es la peor opción con margen claro (exp01, 04, 07,
10 ocupan las posiciones más bajas del bloque); descongelar `layer4` ya basta para la mayor
parte de la ganancia, y sumar `layer3` no aporta de forma consistente a este LR.

---

## Bloque 2 (exp11–exp19) — LR de backbone × extensión de capas (cabeza fija)

Punto de partida: `layer4+layer3`, cabeza lineal. LR de cabeza fijo en 1e-3; el LR de
backbone se desacopla y se vuelve la variable principal. Batch size 32, image size 224px.

| Exp | Descongelado | Cabeza | LR backbone | AUC | F1 | Acc | Precisión | Sensibilidad | Época mejor / máx |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| exp11 | layer4+layer3 | Lineal | 1e-3 | 0.8658 | 0.8372 | 0.8148 | 0.7674 | 0.9209 | 1 / 21 |
| exp12 | layer4+layer3 | Lineal | 3e-4 | 0.8539 | 0.8362 | 0.8148 | 0.7704 | 0.9143 | 2 / 22 |
| exp13 | layer4+layer3 | Lineal | 1e-4 | 0.8482 | 0.8244 | 0.8068 | 0.7778 | 0.8769 | 4 / 24 |
| exp14 | layer4+layer3 | Lineal | 3e-5 | 0.8454 | 0.8137 | 0.7966 | 0.7727 | 0.8593 | 6 / 26 |
| exp15 | layer4+layer3 | Lineal | 1e-5 | 0.8428 | 0.8226 | 0.8034 | 0.7712 | 0.8813 | 17 / 37 |
| exp16 | layer4 | Lineal | 1e-4 | 0.8482 | 0.8190 | 0.8011 | 0.7734 | 0.8703 | 6 / 26 |
| exp17 | layer4 | MLP (0.3) | 1e-4 | 0.8494 | 0.8277 | 0.8045 | 0.7606 | 0.9077 | 3 / 23 |
| exp18 | layer4+layer3 | MLP (0.3) | 1e-4 | 0.8493 | 0.8170 | 0.8000 | 0.7751 | 0.8637 | 4 / 24 |
| exp19 | layer4+layer3+layer2 | Lineal | 1e-4 | 0.8529 | 0.8157 | 0.8023 | 0.7873 | 0.8462 | 5 / 25 |

**Lectura del bloque:** el AUC decrece de forma monótona al bajar el LR del backbone
(0.8658 → 0.8539 → 0.8482 → 0.8454 → 0.8428 en exp11→15) — bajar el LR del backbone nunca
ayudó en este grid. La mejor corrida de todo el bloque (exp11) es, de hecho, la que **no**
diferencia LR de backbone y cabeza.

---

## Bloque 3 (exp20–exp22) — resolución de imagen y régimen de optimización

Parte de la config ganadora de profundidad (`layer4`, cabeza lineal), pero cambia resolución,
batch size y LR como paquete. exp22 además introduce loss BCE y LR discriminativo real
(optimizador con grupos de parámetros separados).

| Exp | Descongelado | Cabeza | Imagen | Batch | LR cabeza | LR backbone | Loss | AUC | F1 | Acc | Precisión | Sensibilidad | Época mejor / máx |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| exp20 | layer4 | Lineal | 384px | 16 | 5e-4 | 5e-4 (único) | CE | 0.8679 | 0.8215 | 0.8034 | 0.7743 | 0.8747 | 1 / 21 |
| exp21 | layer4 | Lineal | 512px | 16 | 5e-4 | 5e-4 (único) | CE | 0.8655 | 0.8105 | 0.8023 | 0.8035 | 0.8176 | 4 / 24 |
| exp22 | layer4 | MLP (0.3) | 512px | 16 | 1e-3 | 1e-5 | BCE | 0.8675 | 0.8227 | 0.8080 | 0.7871 | 0.8615 | 21 / 41 |

**Lectura del bloque:** subir la resolución sola (exp20→exp21, 384→512px) no mejora el AUC
(0.8679→0.8655) y de hecho baja la sensibilidad de forma notoria (0.8747→0.8176). exp22
recupera parte del terreno con LR discriminativo real y BCE a la misma resolución de exp21,
pero sin superar a exp20.

---

## Ranking global — las 22 corridas por AUC-ROC en test

| # | Exp | Bloque | Descongelado | Cabeza | AUC | F1 |
|---:|---|---:|---|---|---:|---:|
| 1 | exp20 | 3 | layer4 | Lineal | 0.8679 | 0.8215 |
| 2 | exp22 | 3 | layer4 | MLP | 0.8675 | 0.8227 |
| 3 | exp11 | 2 | layer4+layer3 | Lineal | 0.8658 | 0.8372 |
| 4 | exp21 | 3 | layer4 | Lineal | 0.8655 | 0.8105 |
| 5 | exp08 | 1 | layer4 | Lineal | 0.8630 | 0.8130 |
| 6 | exp09 | 1 | layer4+layer3 | Lineal | 0.8626 | 0.8296 |
| 7 | exp05 | 1 | layer4 | MLP | 0.8581 | 0.8295 |
| 8 | exp02 | 1 | layer4 | MLP | 0.8548 | 0.8318 |
| 9 | exp12 | 2 | layer4+layer3 | Lineal | 0.8539 | 0.8362 |
| 10 | exp19 | 2 | layer4+layer3+layer2 | Lineal | 0.8529 | 0.8157 |
| 11 | exp03 | 1 | layer4+layer3 | MLP | 0.8524 | 0.8188 |
| 12 | exp06 | 1 | layer4+layer3 | MLP | 0.8504 | 0.7930 |
| 13 | exp01 | 1 | Congelado | MLP | 0.8501 | 0.8185 |
| 14 | exp17 | 2 | layer4 | MLP | 0.8494 | 0.8277 |
| 15 | exp18 | 2 | layer4+layer3 | MLP | 0.8493 | 0.8170 |
| 16 | exp16 | 2 | layer4 | Lineal | 0.8482 | 0.8190 |
| 17 | exp13 | 2 | layer4+layer3 | Lineal | 0.8482 | 0.8244 |
| 18 | exp14 | 2 | layer4+layer3 | Lineal | 0.8454 | 0.8137 |
| 19 | exp15 | 2 | layer4+layer3 | Lineal | 0.8428 | 0.8226 |
| 20 | exp04 | 1 | Congelado | MLP | 0.8411 | 0.8085 |
| 21 | exp10 | 1 | Congelado (seed42) | Lineal | 0.8357 | 0.8119 |
| 22 | exp07 | 1 | Congelado | Lineal | 0.8355 | 0.8120 |

## Conclusiones

1. **Congelar el backbone es sistemáticamente la peor opción** — las 4 corridas con backbone
   totalmente congelado (exp01, 04, 07, 10) ocupan 3 de las 4 últimas posiciones del ranking
   global, con la única excepción de exp01 (posición 13) gracias a la cabeza MLP+dropout 0.5.
2. **El mejor AUC global (exp20, 0.8679) y el mejor F1 (exp11, 0.8372) no coinciden** —
   exp20 gana en discriminación agregada pero exp11 tiene mejor equilibrio precisión/recall al
   umbral por defecto.
3. **Bajar el LR del backbone nunca ayuda** dentro de un mismo régimen (Bloque 2,
   exp11→exp15 monótono decreciente).
4. **Subir la resolución de imagen no es una mejora automática** — exp21 (512px) no supera a
   exp20 (384px) con el mismo régimen de LR/batch; hace falta ajustar el LR discriminativo
   (exp22) para no perder terreno a esa resolución.
5. Los tres bloques comparten el mismo test set (880 imágenes), lo que hace las 22 corridas
   directamente comparables entre sí sin ajuste adicional.
