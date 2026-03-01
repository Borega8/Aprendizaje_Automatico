import numpy as np

X = np.array([
  [0,0],
  [0,1],
  [1,0],
  [1,1],
])

y = np.array([0,1,1,0])

w = np.zeros(2)
b = 0
lr = 0.1

def step(z):
  return 1 if z >= 0 else 0

for _ in range(10):
  for i in range(4):
    z = np.dot(w, X[i]) + b
    y_pred = step(z)
    error = y[i] - y_pred
    
    w += lr * error * X[i]
    b += lr * error

print("Pesos:", w)
print("Bias:", b)

for x in X:
  print(x, step(np.dot(w, x) + b))