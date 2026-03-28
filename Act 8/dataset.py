import numpy as np

CLASES         = ["Setosa", "Versicolor", "Virginica"]
COLORES_CLASE  = ["#f97316", "#22d3ee", "#a78bfa"]
COLORES_KMEANS = ["#fb7185", "#4ade80", "#facc15"]

def generar_dataset(n_por_clase: int = 40, seed: int = 42) -> tuple:
    np.random.seed(seed)
    centros = np.array([[2.0, 1.0],
                        [5.5, 3.5],
                        [7.5, 6.0]])
    X_list, y_list = [], []
    for i, centro in enumerate(centros):
        puntos = np.random.randn(n_por_clase, 2) * 0.55 + centro
        X_list.append(puntos)
        y_list.extend([i] * n_por_clase)

    return np.vstack(X_list), np.array(y_list)

def train_test_split(X: np.ndarray, y: np.ndarray,
                     test_ratio: float = 0.25, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    idx    = rng.permutation(len(X))
    n_test = int(len(X) * test_ratio)
    return (X[idx[n_test:]], y[idx[n_test:]],
            X[idx[:n_test]],  y[idx[:n_test]])
