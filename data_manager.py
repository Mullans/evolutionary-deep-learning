import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_accuracy_over_gen(filenames, labels=[], colors=[], lower_cutoff=0.5):
    for i in range(len(filenames)):
        data = pd.read_csv(filenames[i], sep=',').values.astype(np.float32)
        generations = data[:, 0]
        fitness = data[:, 1]
        mask = np.ones(len(generations), dtype=bool)
        mask[fitness < lower_cutoff] = False
        fitness = fitness[mask]
        generations = generations[mask]
        if len(labels) > i:
            label = labels[i]
        else:
            label = "file {}".format(i+1)
        if len(colors) > i:
            color = colors[i]
        else:
            color = np.random.rand(3, 1)
        plt.scatter(generations, fitness, color=color, label=label)
    plt.title('Evolved Networks')
    plt.xlabel('Generation')
    plt.ylabel('Test Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()
