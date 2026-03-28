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

# ─────────────────────────────────────────────────────────────
# 5. Resumen comparativo
# ─────────────────────────────────────────────────────────────
# print("\n" + "=" * 60)
# print("  HALLAZGOS")
# print("=" * 60)
# print(f"""
#   1. KNN ({acc*100:.1f}% acc) clasifica con alta precisión gracias
#      a las etiquetas: aprende la frontera de decisión
#      implícitamente por similitud geométrica.

#   2. K-Means ({pur*100:.1f}% pureza) descubre la misma estructura
#      sin ver ninguna etiqueta, minimizando la varianza
#      intra-clúster (WCSS={res['inercia']:.1f}).

#   3. Diferencia clave: supervisado produce etiquetas
#      accionables; no supervisado revela agrupaciones
#      naturales desconocidas a priori.

#   4. La alta pureza de K-Means confirma que los clústeres
#      geométricos coinciden con las clases reales del dataset.
# """)
