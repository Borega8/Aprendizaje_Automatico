import numpy as np

class KNN:
    def __init__(self, k: int = 5):
        self.k        = k
        self.X_train  = None
        self.y_train  = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _predecir_uno(self, x: np.ndarray) -> int:
        distancias = np.linalg.norm(self.X_train - x, axis=1)
        vecinos    = np.argsort(distancias)[: self.k]
        etiquetas  = self.y_train[vecinos]
        clases, conteos = np.unique(etiquetas, return_counts=True)
        return int(clases[np.argmax(conteos)])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predecir_uno(x) for x in X])

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))


# Métricas
def matriz_confusion(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     n_clases: int) -> np.ndarray:
    M = np.zeros((n_clases, n_clases), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[t][p] += 1
    return M


def recall_por_clase(conf_mat: np.ndarray, clases: list) -> dict:
    resultados = {}
    for i, cls in enumerate(clases):
        total = conf_mat[i].sum()
        resultados[cls] = conf_mat[i][i] / total if total else 0.0
    return resultados
