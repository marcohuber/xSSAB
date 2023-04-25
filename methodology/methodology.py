import os
import numpy as np

from tqdm import tqdm

import utils_os

from methodology.gradient_calculator import get_gradient, get_cosine, save_combined_map
from methodology.face_matching_module import FaceMatching
from methodology.plot_gradient import plot_gradient_2

###############################################################
# Specify model and datapaths
MODELNAME = 'CurricularFace'  # 
MODEL_PATH = "./models/CurricularFace_Backbone.pth"  # 
PATCHED_IMGS = False  # True, if PatchLFW is used
SRC_PATH_GRAD = "./dataset/ds_lfw_mag_mtcnn/"  # "./dataset/ds_lfw_mag_mtcnn/"  './patches/PatchLFW'

dataset = 'patch' if PATCHED_IMGS else 'ds_lfw_mag_mtcnn'
###############################################################

SRC_PATH_THR = f"./dataset/{dataset}/"
DEST_PATH_COS = f'./methology/cos_sim/{dataset}'
DEST_PATH_THR = f'./methology/cos_sim/ds_lfw_mag_mtcnn/{MODELNAME}_thr.npy'
DEST_PATH_GRAD = f"./methology/gradient/{MODELNAME}/{dataset}/"
DEST_PATH_GRAD_PLOT = f"./methology/gradient_img/{MODELNAME}/{dataset}/"

if not os.path.exists(DEST_PATH_GRAD):
    os.makedirs(DEST_PATH_GRAD)
if not os.path.exists(DEST_PATH_GRAD_PLOT):
    os.makedirs(DEST_PATH_GRAD_PLOT)
if not os.path.exists(DEST_PATH_COS):
    os.makedirs(DEST_PATH_COS)

backbone = utils_os.get_model(MODEL_PATH, MODELNAME)
model = FaceMatching(backbone, MODELNAME)
model.eval()


def compute_cos(pairs, gen_imp, imgs, specification):
    """
    Compute cosine similarity values for pairs.
    :param pairs: id list of image pairs
    :param gen_imp: list identifying genuine and imposter pairs
    :param imgs: list of images
    :return: list of cosine similarities for image pairs
    """
    cos_list = []

    for pair, g_i in tqdm(zip(pairs, gen_imp)):
        id1 = pair[0]
        id2 = pair[1]

        # get standard cosine similarity
        cos_compl, _ = get_cosine(imgs[id1], imgs[id2], model, backbone, 0, MODELNAME, None)
        cos_list.append(cos_compl.detach().cpu().numpy()[0])
    res = np.array(cos_list)

    if not os.path.exists(os.path.join(DEST_PATH_COS, specification)):
        os.makedirs(os.path.join(DEST_PATH_COS, specification))

    np.save(os.path.join(DEST_PATH_COS, specification, f"{MODELNAME}_cos.npy"), res)
    np.save(os.path.join(DEST_PATH_COS, specification, f"{MODELNAME}_pairs.npy"), pairs)
    np.save(os.path.join(DEST_PATH_COS, specification, f"{MODELNAME}_gen_imp.npy"), gen_imp)
    return res, pairs, gen_imp


def compute_threshold():
    """EER Threshold computation"""
    print('Compute Threshold', '-'*20)

    print('Load Images')
    imgs = utils_os.load_all_images(utils_os.image_iter(SRC_PATH_THR))
    all_pairs, gen_imp = utils_os.get_pairs_genimp()

    print('Compute Cosine Similarities')
    all_cos, pairs, gen_imp = compute_cos(all_pairs, gen_imp, imgs, '')

    print('Compute EER Threshold')
    thr_path = os.path.join(DEST_PATH_COS, f'{MODELNAME}_thr.npy')
    utils_os.save_threshold(all_cos, gen_imp, thr_path)


def compute_gradients():
    """Gradient computation"""
    print('Compute gradients', '-'*20)

    print('Load Images')
    imgs = utils_os.load_all_images(utils_os.image_iter(SRC_PATH_GRAD))
    all_pairs, gen_imp = utils_os.get_pairs_genimp()
    thr = np.load(DEST_PATH_THR)

    print('Compute Gradients')
    for x in range(2):
        for pair in tqdm(all_pairs):
            id1 = int(pair[0])
            id2 = int(pair[1])

            # compute positive gradient
            gradient = get_gradient(imgs[id1], imgs[id2], model, backbone, 1, MODELNAME, thr)
            np.save(os.path.join(DEST_PATH_GRAD, '{}_{}_gradient_{}.npy'.format(id1, id2, "pos")), gradient)

            # compute negative gradient
            gradient = get_gradient(imgs[id1], imgs[id2], model, backbone, 2, MODELNAME, thr)
            np.save(os.path.join(DEST_PATH_GRAD, '{}_{}_gradient_{}.npy'.format(id1, id2, "neg")), gradient)

        save_combined_map(all_pairs, DEST_PATH_GRAD)
        if PATCHED_IMGS:
            break
        all_pairs = all_pairs[:, [1, 0]]


if __name__ == "__main__":
    compute_threshold()
    compute_gradients()
    plot_gradient_2("../dataset/ds_lfw_mag_mtcnn/1.jpg", "../dataset/ds_lfw_mag_mtcnn/0.jpg",
                    os.path.join(DEST_PATH_GRAD, '1_0_gradient_pos.npy'),
                    os.path.join(DEST_PATH_GRAD, '1_0_gradient_neg.npy'),
                    DEST_PATH_GRAD_PLOT, 0
                    )
