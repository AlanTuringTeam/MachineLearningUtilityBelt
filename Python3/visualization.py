import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from skimage import measure
from sklearn.datasets import make_classification
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV


def plot_hyperplane(clf, ax: Axes3D = None, interval: float = .05) -> Line3DCollection:
    """
Plots the hyperplane of the model in an axes
    :param clf: the classifier to use to find the hyperplane
    :param ax: the axes to plot the hyperplane into
    :param interval: the precision of the the hyperplane rendering.
    :return: the mesh of the created hyperplane that was added to the axes
    """
    if ax is None:
        ax = plt.gca(projection="3d")
        
    interval = int(1 / interval)

    # get the separating hyperplane
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    z_min, z_max = ax.get_zlim()

    # create grid to evaluate model
    xx = np.linspace(x_min, x_max, interval)
    yy = np.linspace(y_min, y_max, interval)
    zz = np.linspace(z_min, z_max, interval)

    yy, xx, zz = np.meshgrid(yy, xx, zz)

    if hasattr(clf, "decision_function"):
        z = clf.decision_function(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])
    elif hasattr(clf, "predict_proba"):
        z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])[:, 1]
    z = z.reshape(xx.shape)

    vertices, faces, _, _ = measure.marching_cubes(z, 0)
    # Scale and transform to actual size of the interesting volume
    vertices = vertices * [x_max - x_min, y_max - y_min, z_max - z_min] / interval
    vertices += [x_min, y_min, z_min]
    # and create a mesh to display
    # mesh = Poly3DCollection(vertices[faces],
    #                         facecolor='orange', alpha=0.3)
    mesh = Line3DCollection(vertices[faces],
                            facecolor='orange', alpha=0.3)

    ax.add_collection3d(mesh)

    return mesh

def plot_fitting_plane(clf, ax: Axes3D):
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    yy, xx = numpy.meshgrid(numpy.linspace(y_min, y_max), numpy.linspace(x_min, x_max))

    z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    ax.plot_wireframe(xx, yy, z)

def test_classification_data(n_samples=100, random_state=None) -> (np.ndarray, np.ndarray):
    return make_classification(n_samples, n_features=3, n_informative=3, n_redundant=0, random_state=random_state)


def create_color_array(labels, colors_array):
    return np.array(list(map(lambda x: colors_array[x], labels)))


def retrieve_svm_features_manipulated_by_kernel(svm, features):
    return features * svm._compute_kernel(features)


def deep_grid_search_cv_svm(clf, features, labels, c_min=1, c_max=1e3, c_precision=1000, gamma_min=1e-2, gamma_max=1e2,
                            gamma_precision=1000, kernels=("linear", "rbf", "sigmoid"),
                            class_weights=("balanced", None)):
    gamma_test = list(np.linspace(gamma_min, gamma_max, gamma_precision))
    gamma_test.append("auto")

    param_grid = {"C": np.linspace(c_min, c_max, c_precision),
                  "gamma": gamma_test,
                  "kernel": kernels,
                  "class_weight": class_weights}

    grid_search = GridSearchCV(clf, param_grid, n_jobs=-1, verbose=10)
    grid_search.fit(features, labels)


def save_sklearn_model(clf, filename):
    return joblib.dump(clf, filename)


def load_sklean_model(filename):
    return joblib.load(filename)
