import numpy as np


class KMeans:
    def __init__(self, k: int = 3, max_iter: int = 100,
                 tol: float = 1e-6, seed: int = 42):
        self.k           = k
        self.max_iter    = max_iter
        self.tol         = tol
        self.seed        = seed
        self.centroides  = None
        self.asignaciones = None
        self.historia    = []
        self.inercia_hist = []

    # Asignar al centroide más cercano
    def _asignar(self, X: np.ndarray) -> np.ndarray:
        dists = np.linalg.norm(X[:, None] - self.centroides[None, :], axis=2)
        return np.argmin(dists, axis=1)

    # Recalcular centroides
    def _actualizar(self, X: np.ndarray, asign: np.ndarray) -> np.ndarray:
        nuevos = np.zeros_like(self.centroides)
        for k in range(self.k):
            pts = X[asign == k]
            nuevos[k] = pts.mean(axis=0) if len(pts) > 0 else self.centroides[k]
        return nuevos

    def _inercia(self, X: np.ndarray, asign: np.ndarray) -> float:
        return float(sum(
            np.sum((X[asign == k] - self.centroides[k]) ** 2)
            for k in range(self.k)
        ))

    def fit(self, X: np.ndarray) -> "KMeans":
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), self.k, replace=False)
        self.centroides = X[idx].copy().astype(float)

        for i in range(self.max_iter):
            asign   = self._asignar(X)
            nuevos  = self._actualizar(X, asign)
            inercia = self._inercia(X, asign)

            self.historia.append({
                "iter"        : i + 1,
                "centroides"  : self.centroides.copy(),
                "asignaciones": asign.copy(),
                "inercia"     : inercia,
            })
            self.inercia_hist.append(inercia)

            desplaz = np.linalg.norm(nuevos - self.centroides)
            self.centroides   = nuevos
            self.asignaciones = asign

            if desplaz < self.tol:
                break

        return self

    def pureza(self, y_true: np.ndarray) -> float:
        total = 0
        for k in range(self.k):
            mask = self.asignaciones == k
            if not mask.any():
                continue
            _, counts = np.unique(y_true[mask], return_counts=True)
            total += counts.max()
        return total / len(y_true)

    def resumen(self) -> dict:
        return {
            "iteraciones" : len(self.historia),
            "inercia"     : self.inercia_hist[-1],
            "centroides"  : self.centroides,
            "tamanios"    : [(self.asignaciones == k).sum() for k in range(self.k)],
        }
