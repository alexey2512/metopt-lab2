import numpy as np

# sin x cos y
sinxcosy = lambda x: np.sin(x[0]) * np.cos(x[1])

# rosenbrock function
rosenbrock = lambda x: 0.1 * (x[1] - x[0] ** 2) ** 2 + (x[0] - 1) ** 2

# himmelblau function
himmelblau = lambda x: (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2

# stibenski-tang function
stibenski_tang = lambda x: ((x[0] ** 4 + x[1] ** 4) - 16 * (x[0] ** 2 + x[1] ** 2) + 5 * (x[0] + x[1])) / 2
