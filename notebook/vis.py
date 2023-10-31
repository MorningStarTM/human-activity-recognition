import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

act1 = pd.read_csv("E:\\github_clone\\human-activity-recognition\\data\\custom_activity_old\\person_1\\Activity_1\\Accelerometer.csv")
act1['mag'] = np.sqrt((act1['x']**2 + act1['y']**2 + act1['z']**2))
def plot_flux(path, title):
    act1 = pd.read_csv(path)
    act1['mag'] = np.sqrt((act1['x']**2 + act1['y']**2 + act1['z']**2))
    plt.figure(figsize=(16, 6))  # Set the figure size
    plt.plot(act1.index, act1['mag'], linestyle='-')
    plt.title(title)
    plt.show()

plot_flux("E:\\github_clone\\human-activity-recognition\\data\\custom_activity_old\\person_1\\Activity_1\\Accelerometer.csv", "Activity 2")

