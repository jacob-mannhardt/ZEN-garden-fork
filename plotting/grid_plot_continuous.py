import matplotlib.pyplot as plt
import numpy as np


# Create a figure with square aspect ratio
fig, ax = plt.subplots(figsize=(6, 6))

# Set labels for the axes
ax.set_xlabel("Investment Horizon")
ax.set_ylabel("Operation Horizon")

# Set the limits for the axes
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)

# Add ticks on both axes
ax.set_xticks(range(0, 16))
ax.set_yticks(range(0, 16))

# Plot the line x = y
x = np.linspace(0, 15, 100)
y = x
ax.plot(x, y, label="x = y", color='black', alpha=0.2)

# Fill the area where y > x
ax.fill_between(x, y, 15, color='red', alpha=0.1)

# Show the plot (you can omit this line if you are adding this code to a larger project)
plt.show()
