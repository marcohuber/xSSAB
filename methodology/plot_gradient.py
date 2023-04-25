import os

import numpy as np
from matplotlib import pyplot as plt

from methodology.gradient_calculator import load_gradient_mean

MODELNAME = 'ElasticArcface'
SRC_PATH_COS = "./cos_sim/" + MODELNAME + "/ds_lfw_mag_mtcnn/"
PATH_THR = "/cos_sim/ElasticArcface_threshold.npy"
DEST_PATH_HIST = "./hist/hist_" + MODELNAME + ".jpg"


def plot_gradient_2(img_path1, img_path2, grad_pos_path, grad_neg_path, plot_path, masked):
    """
    Save positive, negative and combined gradient maps for image_path1
    :param img_path1: path to input image
    :param img_path2: path to reference image
    :param grad_pos_path: path to positive gradient argument file
    :param grad_neg_path: path to negative gradient argument file
    :param plot_path: save directory for gradient images
    :param masked: masked percentage
    :return:
    """
    id1 = os.path.basename(img_path1).strip('.jpg')
    id2 = os.path.basename(img_path2).strip('.jpg')

    # get mean gradient
    gradient_1_2_pos = load_gradient_mean(grad_pos_path)
    gradient_1_2_neg = load_gradient_mean(grad_neg_path)

    max_grad = max(map(max, [gradient_1_2_pos.flatten(), gradient_1_2_neg.flatten()]))
    min_grad = min(map(min, [gradient_1_2_pos.flatten(), gradient_1_2_neg.flatten()]))
    bins = 100

    # visualize positive gradient map
    plt.imshow(gradient_1_2_pos, vmin=min_grad, vmax=max_grad, cmap='Greens')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'pos_grad_input_{}_ref_{}_replace_{}pct'.format(id1, id2, int(masked * 100))),
                bbox_inches='tight')

    # visualize negative gradient maps
    plt.imshow(gradient_1_2_neg, vmin=min_grad, vmax=max_grad, cmap='Reds')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'neg_grad_input_{}_ref_{}_replace_{}pct'.format(id1, id2, int(masked * 100))),
                bbox_inches='tight')

    # visualize combined gradient maps
    combi = gradient_1_2_pos - gradient_1_2_neg
    bound = max(np.abs(min(map(min, combi))), max(map(max, combi)))
    plt.imshow(combi, vmin=bound * -1, vmax=bound, cmap='RdYlGn')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_path, 'combi_grad_input_{}_ref_{}_replace_{}pct'.format(id1, id2, int(masked * 100))),
                bbox_inches='tight')

    plt.close('all')
