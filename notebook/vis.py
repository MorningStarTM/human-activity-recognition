import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("E:\\github_clone\\human-activity-recognition\\data\\copy\\person_1\\downstairs\\acc.csv")

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the accelerometer data
ax.scatter(df['x'], df['y'], df['z'], c='b', marker='o')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set a title
plt.title('3D Accelerometer Data')

# Show the plot
plt.show()
"""act1 = pd.read_csv("E:\\github_clone\\human-activity-recognition\\data\\custom_activity_old\\person_1\\Activity_1\\Accelerometer.csv")
act1['mag'] = np.sqrt((act1['x']**2 + act1['y']**2 + act1['z']**2))
def plot_flux(path, title):
    act1 = pd.read_csv(path)
    act1['mag'] = np.sqrt((act1['x']**2 + act1['y']**2 + act1['z']**2))
    plt.figure(figsize=(16, 6))  # Set the figure size
    plt.plot(act1.index, act1['mag'], linestyle='-')
    plt.title(title)
    plt.show()

plot_flux("E:\\github_clone\\human-activity-recognition\\data\\custom_activity_old\\person_1\\Activity_1\\Accelerometer.csv", "Activity 2")

"""