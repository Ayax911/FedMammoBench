# Métricas exp65–69 — ResNet18 ImageNet con descongelamiento progresivo

Fuente: `runs/centralizado/exp6{5..9}_*/{test,val}/metrics.json` (mejor checkpoint) y `metrics.csv` (épocas ejecutadas).

| exp | capas descongeladas | épocas |
|---|---|---|
| 65 | ninguna (frozen) | 200 |
| 66 | layer4 | 113 |
| 67 | layer4–3 | 147 |
| 68 | layer4–2 | 200 |
| 69 | layer4–1 | 200 |

## Test

| exp | AUC | Accuracy | F1 | F1 macro | Precision | Sensibilidad | Especificidad | Loss |
|---|---|---|---|---|---|---|---|---|
| 65 | 0.8220 | 0.7482 | 0.5833 | 0.7015 | 0.6476 | 0.5307 | 0.8564 | 0.7147 |
| 66 | 0.8874 | 0.8094 | **0.7244** | **0.7893** | 0.6967 | **0.7545** | 0.8366 | **0.5403** |
| 67 | **0.8895** | **0.8118** | 0.7065 | 0.7840 | 0.7326 | 0.6823 | 0.8761 | 0.6546 |
| 68 | 0.8625 | 0.8046 | 0.6747 | 0.7675 | **0.7545** | 0.6101 | **0.9013** | 1.3790 |
| 69 | 0.8534 | 0.7914 | 0.6742 | 0.7604 | 0.7004 | 0.6498 | 0.8618 | 1.1003 |

## Validación

| exp | AUC | Accuracy | F1 | F1 macro | Precision | Sensibilidad | Especificidad | Loss |
|---|---|---|---|---|---|---|---|---|
| 65 | 0.8232 | 0.7763 | 0.6355 | 0.7371 | 0.7026 | 0.5801 | 0.8757 | 0.7151 |
| 66 | 0.8925 | 0.8278 | 0.7483 | 0.8087 | 0.7354 | 0.7616 | 0.8613 | 0.5409 |
| 67 | 0.9042 | 0.8457 | 0.7717 | 0.8276 | 0.7676 | 0.7758 | 0.8811 | 0.6444 |
| 68 | 0.8888 | 0.8481 | 0.7562 | 0.8229 | 0.8208 | 0.7011 | 0.9225 | 1.3407 |
| 69 | 0.8846 | 0.8373 | 0.7472 | 0.8136 | 0.7821 | 0.7153 | 0.8991 | 0.9260 |

## Lectura

- Descongelar mejora mucho sobre el backbone congelado (AUC test 0.822 → 0.887–0.890 con layer4/layer3).
- Mejor AUC en test: exp67 (0.8895), prácticamente empatado con exp66 (0.8874); exp66 tiene mejor F1/sensibilidad y menor loss.
- Descongelar layer2 y layer1 empeora AUC (0.863 / 0.853) y la loss se infla (1.38 / 1.10), señal de sobreajuste/descalibración.
- Las diferencias entre exp66 y exp67 son pequeñas; no se ha evaluado significancia estadística (ver advertencia sobre AUC por imagen vs. por paciente en `docs/AUDITORIA_SOBREAJUSTE_RESULTADOS.md`).
- Son métricas pooled; no se recalibró el umbral (0.5 fijo) ni se desglosó por base de datos (ver `test/metrics_by_database.json`).
