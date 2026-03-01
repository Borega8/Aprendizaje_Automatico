import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([
  [0.0,0.0],
  [0.0,1.0],
  [1.0,0.0],
  [1.0,1.0]
])

y = torch.tensor([
  [0.0],
  [1.0],
  [1.0],
  [0.0]
])

model = nn.Sequential(
  nn.Linear(2, 4),
  nn.ReLU(),
  nn.Linear(4, 1),
  nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
  optimizer.zero_grad()
  output = model(X)
  loss = criterion(output, y)
  loss.backward()
  optimizer.step()

with torch.no_grad():
  print(torch.round(model(X)))