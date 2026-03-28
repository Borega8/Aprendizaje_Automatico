from dataset import generar_dataset, train_test_split, CLASES
from knn import KNN, matriz_confusion, recall_por_clase
from kmeans import KMeans

X, y = generar_dataset(n_por_clase=40)
X_train, y_train, X_test, y_test = train_test_split(X, y, test_ratio=0.25)

print("\n\tDataset")
print(f"Total={len(X)}  Train={len(X_train)}  Test={len(X_test)}")


print("\n\tSupervisado: KNN")
knn   = KNN(k=5).fit(X_train, y_train)
y_pred = knn.predict(X_test)
acc    = knn.score(X_test, y_test)
cm     = matriz_confusion(y_test, y_pred, len(CLASES))
recall = recall_por_clase(cm, CLASES)

print(f"Exactitud global : {acc*100:.1f}%")
for cls, val in recall.items():
    print(f"Recall {cls:12}: {val*100:.1f}%")


print("\n\tNo Supervisado: K-Means")
km  = KMeans(k=3).fit(X)
res = km.resumen()
pur = km.pureza(y)

print(f"Iteraciones      : {res['iteraciones']}")
print(f"Inercia (WCSS)   : {res['inercia']:.2f}")
print(f"Pureza           : {pur*100:.1f}%")
for k, (c, t) in enumerate(zip(res["centroides"], res["tamanios"])):
    print(f"Clúster {k+1}: centroide=({c[0]:.2f}, {c[1]:.2f})  puntos={t}")
