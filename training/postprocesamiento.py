
import numpy as np


def postprocesar_curva(predicciones):
    """
    Corrige las predicciones granulometricas del modelo.

    Reglas:
    1. Valores entre 0 y 100.
    2. Curva no decreciente.
    3. Ultimo tamiz (200 mm) igual a 100%.
    """
    curva = np.asarray(predicciones, dtype=float)

    if curva.ndim not in (1, 2) or curva.shape[-1] != 11:
        raise ValueError("Se esperan 11 valores por curva.")

    if not np.all(np.isfinite(curva)):
        raise ValueError("Las predicciones contienen NaN o infinitos.")

    corregida = np.clip(curva, 0, 100)
    corregida = np.maximum.accumulate(corregida, axis=-1)
    corregida[..., -1] = 100.0

    return corregida


def validar_curva(curva):
    curva = np.asarray(curva, dtype=float)

    return (
        curva.shape == (11,)
        and np.all(np.isfinite(curva))
        and np.all((curva >= 0) & (curva <= 100))
        and np.all(np.diff(curva) >= 0)
        and curva[-1] == 100.0
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    tamanos = np.array([
        0.002, 0.0063, 0.02, 0.063, 0.2,
        0.63, 2, 6.3, 20, 63, 200
    ])

    # Prediccion artificial para comprobar las tres reglas
    original = np.array([
        -2, 5, 3, 12, 25, 22, 45, 68, 104, 83, 91
    ], dtype=float)

    corregida = postprocesar_curva(original)

    print("\nPRUEBA DE POST-PROCESAMIENTO")
    print("Original: ", original)
    print("Corregida:", corregida)

    print("\nVerificacion:")
    print("Valores entre 0 y 100:",
          np.all((corregida >= 0) & (corregida <= 100)))
    print("Curva no decreciente:",
          np.all(np.diff(corregida) >= 0))
    print("Ultimo valor igual a 100:",
          corregida[-1] == 100)
    print("Curva valida:", validar_curva(corregida))

    plt.figure(figsize=(10, 5))
    plt.plot(tamanos, original, "o--",
             label="Prediccion original")
    plt.plot(tamanos, corregida, "o-",
             label="Prediccion corregida")

    plt.xscale("log")
    plt.xlabel("Tamano de particula (mm)")
    plt.ylabel("Porcentaje que pasa (%)")
    plt.title("Post-procesamiento de curva granulometrica")
    plt.ylim(-10, 115)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        "postprocesamiento_ejemplo.png",
        dpi=150
    )