import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from itertools import combinations


def compute_eer(far, tar, threshold):
    """Compute EER for FAR and TAR."""
    values = np.abs(1 - tar - far)
    idx = np.argmin(values)
    eer = far[idx]
    thr = threshold[idx]
    return eer, thr


def cos_sim(a, b):
    """ Cosine similarity between vector a and b
    """
    a, b = a.reshape(-1), b.reshape(-1)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

