import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np

# Number of rows and columns
n =15

# Creating a figure and axis
fig, ax = plt.subplots(figsize=(8, 8))

# Setting the ticks at the edges of each grid square
ax.set_xticks(np.arange(0.5, n+2, 1))
ax.set_yticks(np.arange(0.5, n+2, 1))

# Setting the minor ticks at the center of each grid square for labeling
ax.set_xticks(np.arange(1, n+1, 1), minor=True,)
ax.set_yticks(np.arange(1, n+1, 1), minor=True)

# Labeling the axes
ax.set_xlabel("Investment Foresight [yr]", fontsize=14)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_ylabel("Operation Foresight [yr]", fontsize=14)

# Adding a grid
ax.grid(True, which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.8)

# Setting axis limits
ax.set_xlim(0, n+1)
ax.set_ylim(0, n+1)

# Generate labels [2, 4, 6, ..., 2*n] for both axes
labels = [str(2*i) for i in range(1, n+1)]

# Setting labels for the minor ticks (centered on the squares)
ax.set_xticklabels(labels, minor=True)
ax.set_yticklabels(labels, minor=True)
ax.tick_params(axis='x', which='minor', length=0)  # Hiding minor tick marks for X-axis
ax.tick_params(axis='y', which='minor', length=0)  # Hiding minor tick marks for Y-axis

# Moving Y-axis numbering to the right-hand side of the plot
ax.tick_params(axis='y', which='minor', labelright=True, labelleft=False)

# Removing labels from the major ticks
ax.tick_params(axis='both', which='major', labelbottom=False, labelright=False)

# Setting aspect of the plot to be equal
ax.set_aspect('equal', adjustable='box')

# Sample matrix to color the squares
matrix = []

# Extending the pattern to the rest of the matrix
for i in range(n):
    row = [1]*(n-i-1) + [0]*(i+1)
    matrix.append(row)


# Create a red colormap with a gradient from strong red for high values to light red for low values
cmap = sns.light_palette("red", as_cmap=True)

# Coloring the squares based on the matrix
for i in range(-1, n+1):
    for j in range(-1, n+1):
        if i == -1 or i == n or j == -1 or j == n:
            border_square = patches.Rectangle((j+0.5, n -i -.5), 1, 1, color='grey', alpha=0.5)
            ax.add_patch(border_square)
        if 0 <= i < n and 0 <= j < n:
            if j < n - 1 - i:
                # Adding a colored square at the specified location
                square = patches.Rectangle((j+0.5, n-i-.5), 1, 1, color='grey', alpha=0.9)  # light red color
                ax.add_patch(square)
            elif matrix[i][j] != 0:
                # Get color from color map using the normalized value
                color = cmap(matrix[i][j])
                # Adding a colored square at the specified location
                square = patches.Rectangle((j+0.5, n-i-.5), 1, 1, facecolor=color)
                ax.add_patch(square)



# Display the plot
plt.show()
