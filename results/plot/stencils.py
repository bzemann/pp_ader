import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Cell boundaries and cell‐average values
x_edges = [-2.5, -1.5, -0.5, 0.5, 1.5]
values  = [
    -0.5495327926151055,  # ū_{i-2}
    -0.8891627363838879,  # ū_{i-1}
     0.0,                 # ū_i
     0.8891627363838879,  # ū_{i+1}
     0.5495327926151055   # ū_{i+2}
]
labels = ['i-2', 'i-1', 'i', 'i+1', 'i+2']

# (x0, x1, û_x,       û_xx,                ū_i)
segments = [
    (-2.5, -1.5, -0.7199509203162717,  0.37949453134331995, values[0]),
    (-1.5, -0.5,  0.278014963617219,   0.40635119166782824, values[1]),
    (-0.5,  0.5,  0.889204390049466,   0.0,                 values[2]),
    ( 0.5,  1.5,  0.27801496361721906, -0.4063511916678282,  values[3]),
    ( 1.5,  2.5, -0.7199509203162717, -0.37949453134331995, values[4])
]

fig, ax = plt.subplots()

# 1) Major ticks and labels at cell centers
center_ticks = [-2.0, -1.0, 0.0, 1.0, 2.0]
ax.set_xticks(center_ticks)
ax.set_xticklabels(labels)

# 2) Minor ticks at cell boundaries (no labels)
ax.set_xticks(x_edges + [2.5], minor=True)

# 3) Vertical grid lines at minor ticks
ax.grid(which='minor', axis='x', zorder=1)

# 4) Dashed blue rectangles for the cell means
for x0, μ in zip(x_edges, values):
    bottom = min(0, μ)
    height = abs(μ)
    ax.add_patch(Rectangle(
        (x0, bottom), 1, height,
        fill=False, linestyle='--', linewidth=1,
        edgecolor='blue', zorder=2
    ))

# 5) Black dotted plot of y = sin(2π/5 x)
x_sine = np.linspace(-2.5, 2.5, 400)
ax.plot(
    x_sine,
    np.sin(2 * np.pi / 5 * x_sine),
    color='black', linestyle=':', linewidth=1, zorder=3
)

# 6) Solid black line at y=0
ax.axhline(0, color='black', linestyle='-', linewidth=1, zorder=4)

# 7) Piecewise reconstructions using eq. (19):
for x0, x1, ux, uxx, μ in segments:
    dx = x1 - x0
    xc = 0.5 * (x0 + x1)
    x_seg = np.linspace(x0, x1, 200)
    x_ref = (x_seg - xc) / dx   # local coordinate ∈ [-½, +½]
    y_seg = μ + ux * x_ref + uxx * (x_ref**2 - 1/12)
    ax.plot(
        x_seg, y_seg,
        color='green', linestyle='-', linewidth=1, zorder=5
    )

# 8) Final axis limits and labels
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-1, 1)
ax.set_xlabel('Interval')
ax.set_ylabel('Value')

# 9) Save
fig.savefig('plot.pdf', format='pdf')
