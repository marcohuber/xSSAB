import fnmatch
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils_os import get_pairs_genimp

##############################################################################
# Specify replacement percentages, strategies, metrics and modelname, datapaths
replace_percentages = np.arange(0, 100, 5)
strategies = ['best_worst', 'random']
APPROACHES = ['gradient']
MODELNAMES = ['ElasticCosface', 'CurricularFace']

PATH_THR = "./methodology/cos_sim/ds_lfw_mag_mtcnn"
PATH_COS = "./methodology/cos_sim/patch"

#############################################################################


def compute_FNMR(cos, genuines, threshold):
    """
    Computes FNMR for given threshold and cosine similarity values.
    """
    match = len(
        np.argwhere(np.isin(np.argwhere(genuines == 1), np.argwhere(cos >= float(threshold)))))
    false_non_match = len(
        np.argwhere(np.isin(np.argwhere(genuines == 1), np.argwhere(cos < float(threshold)))))

    fnmr = false_non_match / (false_non_match + match)
    print(f'False Non Matches: {false_non_match}/{false_non_match + match}')
    return fnmr


def compute_FMR(cos, genuines, threshold):
    """
    Computes FMR for given threshold and cosine similarity values.
    """
    non_match = len(
        np.argwhere(np.isin(np.argwhere(genuines == -1), np.argwhere(cos < float(threshold)))))
    false_match = len(
        np.argwhere(np.isin(np.argwhere(genuines == -1), np.argwhere(cos >= float(threshold)))))

    fmr = false_match / (false_match + non_match)
    print(f"False Matches: {false_match}/{(false_match + non_match)} ")
    return fmr



def plot_all(rate_function, approaches, modelnames, metric):
    """Plot FNMR/FMR of given explainability approaches for given models."""
    for approach in approaches:

        '''Plot our approach performance.'''
        for modelname in modelnames:
            for strategy in strategies:
                s = []
                for replace_percentage in replace_percentages:
                    threshold = np.load(os.path.join(PATH_THR, f'{modelname}_thr.npy'))
                    cos = np.load(os.path.join(PATH_COS, strategy, f'{replace_percentage}_percent', f'{modelname}_cos.npy'))
                    gen_imp = np.load(os.path.join(PATH_COS, strategy, f'{replace_percentage}_percent', f'{modelname}_gen_imp.npy'))

                    s.append(rate_function(cos, gen_imp, threshold))

                x_axis = replace_percentages / 100.
                linestyle = '--' if strategy == 'random' else '-'
                if modelname == 'ElasticArcFace':
                    color = 'blue'
                elif modelname == 'CurricularFace':
                    color = 'red'
                else:
                    color = 'purple'
                plt.plot(x_axis, s, label=f"{approach} {modelname} {strategy}", linestyle=linestyle, color=color)

    plt.ylabel(metric)
    plt.xlabel("Replace Percentage")
    plt.title("Pixel Replacement")
    plt.legend()
    plt.savefig(f'plot_{metric}_{APPROACHES}_{MODELNAMES}.jpg')
    plt.close()


def main():
    """Plot Decision-based Patch Replacement (DPR) curves"""
    plot_all(compute_FMR, APPROACHES, MODELNAMES, 'FMR')
    plot_all(compute_FNMR, APPROACHES, MODELNAMES, 'FNMR')


if __name__ == "__main__":
    main()
