import numpy as np
import functions as fn
import utils as ut
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow
from matplotlib.colors import LinearSegmentedColormap

calc_count = 0
def wrap(f):
    def inner(*args, **kwargs):
        global calc_count
        calc_count += 1
        return f(*args, **kwargs)
    return inner


# params ===
function = fn.himmelblau
bounds = [[-5, 5], [-5, 5]]
start = np.array([1, 1])
strategy = ut.wolfe_strategy
points = ut.bfgs_descent(wrap(function), start, strategy, bounds)
# params ===


# log results
print('point:     ', points[-1])
print('value:     ', function(points[-1]))
print('method its:', len(points) - 1)
print('function:  ', calc_count - 2 * ut.der_count)
print('derivative:', ut.der_count - 2 * ut.grad_count - 6 * ut.hess_count)
print('gradient:  ', ut.grad_count)
print('hessian:   ', ut.hess_count)

# build image
__count = 400
eps = 0.5
bounds = bounds if bounds is not None else \
    [
        [min([x for x, y in points]), max([x for x, y in points])],
        [min([y for x, y in points]), max([y for x, y in points])]
    ]
__x = np.linspace(bounds[0][0] - eps, bounds[0][1] + eps, __count)
__y = np.linspace(bounds[1][0] - eps, bounds[1][1] + eps, __count)
X, Y = np.meshgrid(__x, __y)
Z = function(np.array([X, Y]))

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, levels=25)
ax.clabel(CS, fontsize=7)
ax.grid()
ax.set_title('2D Gradient Descent Visualization')
ax.set_xlabel('X')
ax.set_ylabel('Y')

cmap = LinearSegmentedColormap.from_list('gradient', ['red', 'blue'])
n_points = len(points)
color_norm = plt.Normalize(0, n_points-2)

for i in range(n_points - 1):
    dx = points[i + 1][0] - points[i][0]
    dy = points[i + 1][1] - points[i][1]
    arrow = FancyArrow(points[i][0], points[i][1], dx, dy,
                       width=0.05, length_includes_head=True,
                       head_width=0.05, head_length=0.05,
                       color=cmap(color_norm(i)), alpha=1)
    ax.add_patch(arrow)

plt.show()
