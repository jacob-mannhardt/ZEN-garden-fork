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
ax.set_xlabel("Investment Foresight Horizon", fontsize=14)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.set_ylabel("Operation Foresight Horizon", fontsize=14)

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

dataset = {
    "7_1": 5174664.218,
    "7_2": 5131090.897,
    "7_3": 5095107.793,
    "7_4": 5068570.022,
    "7_5": 5032261.87,
    "7_6": 5012892.733,
    "7_7": 4999259.033,
    "6_1": 5200118.477,
    "6_2": 5143060.586,
    "6_3": 5112577.513,
    "6_4": 5076870.63,
    "6_5": 5044001.39,
    "6_6": 5017435.078,
    "5_1": 5220537.145,
    "5_2": 5161717.18,
    "5_3": 5121825.109,
    "5_4": 5084999.631,
    "5_5": 5046589.242,
    "4_1": 5308352.315,
    "4_2": 5234457.854,
    "4_3": 5179130.997,
    "4_4": 5121601.903,
    "3_1": 5556494.992,
    "3_2": 5455296.862,
    "3_3": 5363860.962,
    "2_1": 5703807.956,
    "2_2": 5589961.884,
    "1_1": 5873989.197
}

# Normalize the dataset values to a range of 0 to 1 for color mapping
min_value = min(dataset.values())
max_value = max(dataset.values())
normalized_dataset = {key: (value - min_value) / (max_value - min_value) for key, value in dataset.items()}

# Function to add data to the matrix, filling from the bottom
def add_data_from_bottom(matrix, dataset, n):
    for key, value in dataset.items():
        col, row = map(int, key.split('_'))
        # Adjust indices for 0-based indexing and start from the bottom row
        adjusted_row = n - row
        adjusted_col = col - 1
        matrix[adjusted_row][adjusted_col] = value

# Add the dataset to the matrix, filling from the bottom
add_data_from_bottom(matrix, normalized_dataset, n)

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

# Create a ScalarMappable with our red color map and the normalization
sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=((min_value/min_value)-1)*100, vmax=((max_value/min_value)-1)*100))

# Create the colorbar
cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.1)
cbar.set_label('Total CAPEX + OPEX [% more expensive]', fontsize=12)
# Reverse the direction of the colorbar
cbar.ax.invert_xaxis()

# Display the plot
plt.show()
