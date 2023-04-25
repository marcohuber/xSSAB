import os

import cv2
import numpy as np
from tqdm import tqdm

from utils_os import get_del_mask, get_ins_mask, \
    load_all_images, image_iter, get_pairs_genimp

#############################################################
# Specify strategy, percentage, datapaths for patched_imgs
replace_percentages = np.arange(0.05, 1., 0.05)
replace_strategy = 'random'  # 'random' 'best_worst'

MODELNAME = 'ElasticArcFace'
PATH_THR = f'./methodology/cos_sim/ds_lfw_mag_mtcnn/{MODELNAME}_thr.npy'
PATH_ORIGINAL_IMGS = './dataset/ds_lfw_mag_mtcnn'
PATH_PATCHED_IMGS = './patches/PatchLFW'
##############################################################

PATH_PATCHED_COS = f'./methodology/cos_sim/patch/{replace_strategy}/0_percent/{MODELNAME}_cos.npy'
PATH_PATCHED_GRAD = "./methodology/gradient/" + MODELNAME + "/patch/"

DEST_REPL_IMGS = './evaluation/replacement/' + replace_strategy + '/' + MODELNAME
DEST_RAND_PXL_LST = './evaluation/replaced_pixel'


if not os.path.exists(DEST_REPL_IMGS):
    os.makedirs(DEST_REPL_IMGS)
if not os.path.exists(DEST_RAND_PXL_LST):
    os.makedirs(DEST_RAND_PXL_LST)


def get_random_masks(grad, n_bounds, id1, id2):
    """Random indices for patch locations."""
    np.random.seed(3)
    ravels = grad.ravel()
    length = int(len(ravels))

    if not os.path.exists(os.path.join(DEST_RAND_PXL_LST, f'{MODELNAME}_pxl_{id1}_{id2}.npy')):
        pixs = np.arange(0, length)
        np.random.shuffle(pixs)
        np.save(os.path.join(DEST_RAND_PXL_LST, f'{MODELNAME}_pxl_{id1}_{id2}.npy'), pixs)
    else:
        pixs = np.load(os.path.join(DEST_RAND_PXL_LST, f'{MODELNAME}_pxl_{id1}_{id2}.npy'))

    steps = int(length / n_bounds)
    masks = []
    for i in range(0, n_bounds):
        selected = pixs[i]
        masks.append(selected)
    return np.array(masks)


def get_bound_for_highest_XX_percent(combi_1_2, XX):
    """Returns threshold for highest XX percent of values in gradient matrix."""
    sorted_combi_1_2 = np.argsort(combi_1_2.reshape(-1))
    idx_XX_percent_highest = int(len(combi_1_2.reshape(-1)) * (1 - XX))
    if idx_XX_percent_highest == len(sorted_combi_1_2):
        bound = np.max(sorted_combi_1_2)
    else:
        bound = combi_1_2.reshape(-1)[sorted_combi_1_2[idx_XX_percent_highest]]
    return bound


def get_bound_for_lowest_XX_percent(combi_1_2, XX):
    """Returns threshold for lowest XX percent of values in gradient matrix."""
    sorted_combi_1_2 = np.argsort(combi_1_2.reshape(-1))
    idx_XX_percent_lowest = int(len(combi_1_2.reshape(-1)) * XX)
    bound = combi_1_2.reshape(-1)[sorted_combi_1_2[idx_XX_percent_lowest]]
    return bound


def insertion(original_img, patched_img, replacement):
    """Replace pixel locations given by replacement in patched_img with pixels in original_img."""
    patched_img[replacement[0], replacement[1]] = original_img[replacement[0], replacement[1]]
    return patched_img


def insert_random_XX_percent(grad, replace_percentage, original_img1, id1, id2, patched_img1):
    """Randomly replace pixels in patched_img1 by pixels in original_img1"""
    length = len(grad.ravel())
    n_bounds = int(replace_percentage * length)
    mask = get_random_masks(grad, n_bounds, id1, id2)

    rows = (mask / len(grad[0])).astype(int)
    columns = mask - rows * len(grad[0])

    patched_img1[rows, columns] = original_img1[rows, columns]
    return patched_img1


def replace_img_pixel(method, pairs, replace_percentage, originals):
    """Replace pixels in PATCH_LFW based on explainability maps of our approach."""
    cos_sim_pairs = np.load(PATH_PATCHED_COS)
    threshold = np.load(PATH_THR)
    replace_str = int(np.around(replace_percentage * 100))

    for idx, pair in tqdm(enumerate(pairs), total=len(pairs)):

        id1 = int(pair[0])
        id2 = int(pair[1])

        # load combined gradient map of image 1
        combi_1_2 = np.load(os.path.join(PATH_PATCHED_GRAD, f'{id1}_{id2}_gradient_combi.npy'))

        # load image 1
        img1 = cv2.imread(os.path.join(PATH_PATCHED_IMGS, f'{id1}.jpg'))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        # uncomment for highlighting
        # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        # img1 = np.zeros_like(img1)
        # img1[:, :, 0] = gray
        # img1[:, :, 1] = gray
        # img1[:, :, 2] = gray

        # load image 2
        img2 = cv2.imread(os.path.join(PATH_PATCHED_IMGS, f'{id2}.jpg'))

        # get cosine similarity of pair
        cos = cos_sim_pairs[idx]

        if method == 'best_worst':
            # replace XX lowest gradient values
            if cos < threshold:
                bound = get_bound_for_lowest_XX_percent(combi_1_2, replace_percentage)
                img1 = insertion(originals[id1], img1, get_ins_mask(combi_1_2, bound))

            # replace XX highest gradient values
            else:
                bound = get_bound_for_highest_XX_percent(combi_1_2, replace_percentage)
                img1 = insertion(originals[id1], img1, get_del_mask(combi_1_2, bound))
        else:  # method == 'random'
            # replace random pixel
            img1 = insert_random_XX_percent(combi_1_2, replace_percentage, originals[id1], id1, id2, img1)

        # save patched images with pixel replacement
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEST_REPL_IMGS, f'{replace_str}_percent', str(id1)+'.jpg'), img1)

        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # uncomment for highlighting
        cv2.imwrite(os.path.join(DEST_REPL_IMGS, f'{replace_str}_percent', str(id2)+'.jpg'), img2)


def main():
    print('\nStart Replacement', '-'*20)
    print('load images')
    originals = load_all_images(image_iter(PATH_ORIGINAL_IMGS))

    all_pairs, gen_imp = get_pairs_genimp()

    for replace in replace_percentages:
        replace_str = int(np.around(replace * 100))
        print('replace', replace_str, 'percent')
 
        # create folder for masked images
        if not os.path.isdir(os.path.join(DEST_REPL_IMGS, f"{replace_str}_percent")):
            os.makedirs(os.path.join(DEST_REPL_IMGS, f"{replace_str}_percent"))

        replace_img_pixel(replace_strategy, all_pairs, replace, originals)


if __name__ == "__main__":
    main()