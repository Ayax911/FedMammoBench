"""Constantes de Fed-ISIC2019 (8 clases) para el pipeline multiclase.

Orden de clases = el de FLamby/ISIC2019 (MEL, NV, BCC, AK, BKL, DF, VASC, SCC -> 0..7).
Ver docs/FEDISIC2019_FLAMBY_REPRO.md §5.D1/§5.H.
"""

ISIC2019_LABEL_MAP: dict[str, int] = {
    "melanoma": 0,
    "melanocytic_nevus": 1,
    "basal_cell_carcinoma": 2,
    "actinic_keratosis": 3,
    "benign_keratosis": 4,
    "dermatofibroma": 5,
    "vascular_lesion": 6,
    "squamous_cell_carcinoma": 7,
}
