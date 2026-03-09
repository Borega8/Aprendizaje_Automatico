import numpy as np
import random

grid_size = 4
goal = (3, 3)
obstacles = [(0,2), (1,0), (2,2), (3,0)]

actions = 4

# Genera un índice para la tabla Q
def state_to_index(state):
    return state[0] * grid_size + state[1]

def step(state, action):
    row, col = state

    if action == 0: row -= 1 # Ir hacia arriba
    elif action == 1: row += 1 # Ir hacia abajo
    elif action == 2: col -= 1 # Ir hacia la izquierda
    elif action == 3: col += 1 # Ir hacia la derecha

    if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
        return state, -0.1

    if (row, col) in obstacles:
        return state, -0.1

    next_state = (row, col)

    if next_state == goal:
        return next_state, 1.0

    return next_state, -0.04


# Inicializar la tabla Q
num_states = grid_size * grid_size
Q = np.zeros((num_states, actions))

alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 2000

# Entrenamiento
for episode in range(episodes):
    state = (0, 0)
    
    while state != goal:
        s = state_to_index(state)

        if random.uniform(0,1) < epsilon:
            action = random.randint(0, actions-1)
        else:
            action = np.argmax(Q[s])

        next_state, reward = step(state, action)
        s_next = state_to_index(next_state)

        Q[s, action] += alpha * (
            reward + gamma * np.max(Q[s_next]) - Q[s, action]
        )

        state = next_state

print("Q-table final:")
print(Q)

print("\nPolítica aprendida:")
symbols = ["^", "v", "<", ">"]

for row in range(grid_size):
    for col in range(grid_size):
        if (row, col) == goal:
            print(" G ", end="")
        elif (row, col) in obstacles:
            print(" X ", end="")
        else:
            s = state_to_index((row, col))
            best_action = np.argmax(Q[s])
            print(f" {symbols[best_action]} ", end="")
    print()