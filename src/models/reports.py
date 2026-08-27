from dataclasses import dataclass


@dataclass
class LoadReport:
    """Resultado de cargar un state_dict remapeado en un modelo.

    Raises:
        RuntimeError: si matched == 0 al construirse — 0 tensores
            coincidieron, aunque el state_dict no estaba vacío.
    """
    matched: int
    missing: list[str]
    unexpected: list[str]

    def __post_init__(self) -> None:
        if self.matched == 0:
            raise RuntimeError(
                "0 tensores coincidieron al cargar los pesos, aunque el "
                "state_dict remapeado no estaba vacío. Revisa el remapeo de "
                "claves contra los nombres reales del modelo."
            )