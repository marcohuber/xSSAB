import os
import argparse

import numpy as np

import utils_os
from methodology import methodology
from methodology.methodology import DEST_PATH_GRAD, DEST_PATH_GRAD_PLOT, SRC_PATH_GRAD, MODELNAME
from patches import patch, patch_eval
from evaluation import replacement, evaluate


def methodology_main():
    """
    This function computes gradients and thresholds based on our methodology on LFW.
    """
    print('modelname:', methodology.MODELNAME)
    methodology.compute_threshold()
    methodology.compute_gradients()
    methodology.plot_gradient_2("./dataset/ds_lfw_mag_mtcnn/1.jpg", "./dataset/ds_lfw_mag_mtcnn/0.jpg",
                              os.path.join(DEST_PATH_GRAD, '1_0_gradient_pos.npy'),
                              os.path.join(DEST_PATH_GRAD, '1_0_gradient_neg.npy'),
                              DEST_PATH_GRAD_PLOT, 0
                              )


def methodology_patches():
    """
    This function computes gradients and cosine similarity values on PatchedLFW.
    :return:
    """
    methodology.compute_gradients()
    imgs = utils_os.load_all_images(utils_os.image_iter(SRC_PATH_GRAD))
    all_pairs, gen_imp = utils_os.get_pairs_genimp()
    methodology.compute_cos(all_pairs, gen_imp, imgs, f'{replacement.replace_strategy}/0_percent')


def methodology_replacement():
    """This function computes cosine similarity values for Patched LFW with replaced pixels."""
    print('modelname:', methodology.MODELNAME)
    all_pairs, gen_imp = utils_os.get_pairs_genimp()
    for r_p in replacement.replace_percentages:
        replace_str = int(np.around(r_p * 100))
        print('\nReplace', replace_str, 'percent', '-'*20)
        imgs = utils_os.load_all_images(utils_os.image_iter(f'./evaluation/replacement/{replacement.replace_strategy}/{MODELNAME}/{replace_str}_percent'))
        methodology.compute_cos(all_pairs, gen_imp, imgs, f'{replacement.replace_strategy}/{replace_str}_percent')


def create_patches():
    """This function creates and saves images of PatchLFW."""
    num_patches = 27
    pairs, gen_imp = utils_os.get_pairs_genimp()
    imposter_pairs = pairs[np.where(gen_imp == -1)]
    patch.patch_imp_pair(imposter_pairs, num_patches)

    genuine_pairs = pairs[np.where(gen_imp == 1)]
    patch.patch_gen_pair(genuine_pairs, num_patches)


def evaluate_patches():
    """ This function runs the entire evaluation on PatchLFW."""
    patch_eval.main()


def replace_pixel():
    """This function runs the placement of pixels based on the gradient maps of PatchLFW."""
    replacement.main()


def evaluate_model_explainability():
    """This function evaluates the ability of explainability models to replace pixels/regions."""
    evaluate.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify script to run",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--script", help="Possible modes: methodology_main, methodology_patches, "
                                               "methodology_replacement, create_patches, evaluate_patches, "
                                               "replace_pixel, evaluate_model_explainability")
    args = vars(parser.parse_args())
    mode = locals()[args['script']]
    mode()