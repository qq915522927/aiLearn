import numpy as np
import matplotlib.pyplot as plt

def plot_digits(data):
    fig, axes = plt.subplots(
        10, 10, figsize=(10, 10),
        subplot_kw={'xticks':[], 'yticks':[]}
    )
    gridspec_kw = dict(hspace=0.1, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        ax.imshow(
            data[i].reshape(8, 8),
            cmap='binary', interpolation='nearest',
            clim=(0, 16)
        )
    plt.show()


def plot_faces(faces):
    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')
    plt.show()

def generate_random_faces(components):
    w = np.random.normal(0, 1, size=len(components))
    return w.dot(components).reshape((-1, components.shape[1]))
