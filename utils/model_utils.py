import matplotlib.pyplot as plt
import numpy as np


def visualize_importances(feature_names,
                          importances,
                          title="Average Feature Importances",
                          plot=True,
                          axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(f"Electrode {feature_names[i]}", ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12, 6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
