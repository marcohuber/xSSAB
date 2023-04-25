import os

import cv2
import numpy as np
from tqdm import tqdm

from utils_os import load_all_images, get_pairs_genimp, list_identity, dict_identities_to_img, image_iter

NUM_PATCHES = 27

SRC_PATH = "./dataset/ds_lfw_mag_mtcnn"
DEST_PATH = "./patches/PatchLFW"

if not os.path.exists(DEST_PATH):
    os.makedirs(DEST_PATH)


def insert(row, column, img1, img2):
    """
    Insert 16x16 pxl patches of img2 in img1.
    :param row: Row indices of lower left corner of patches
    :param column: Column indices of lower left corner of patches
    :param img1: image to insert patches in
    :param img2: image to take patches from
    :return:
    """
    img1[row-16:row, column:column+16] = img2[row-16:row, column:column+16]
    return img1


def patch_imp_pair(pairs, number_patches):
    """
    Replace patches in first image of image pairs by patches taken from second image.
    :param pairs: imposter pairs
    :param number_patches: number of patches to replace in first image.
    :return:
    """
    imgs = load_all_images(image_iter(SRC_PATH))
    index_list = []

    np.random.seed(3)
    for pair in pairs:
        id1 = pair[0]
        id2 = pair[1]
        img1 = imgs[id1]
        img2 = imgs[id2]

        indexes_columns = np.random.randint(0, len(img1)-16, number_patches)
        indexes_rows = np.random.randint(16, len(img1), number_patches)

        for i in np.stack((indexes_columns, indexes_rows), axis=1):
            row = i[1]
            column = i[0]
            img1 = insert(row, column, img1, img2)

        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEST_PATH, str(id1)+".jpg"), img1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEST_PATH, str(id2)+".jpg"), img2)

        index_list.append([id1, id2, indexes_rows, indexes_columns])

    # write patch indexes in file
    with open('./patches/patch_idx_imp.txt', 'w') as f:
        f.write(f"id1, id2 \t row indices \t column indices \n")
        for item in index_list:
            f.write(f"{item[0]}, {item[1]} \t {list(item[2])} \t {list(item[3])} \n")


def patch_gen_pair(pairs, number_patches):
    """
    Replace patches in first image of image pairs by patches taken from an image of a random different identity.
    :param pairs: genuine pairs
    :param number_patches: number of patches to replace in first image.
    :return:
    """
    imgs = load_all_images(image_iter(SRC_PATH))
    index_list = []

    list_id = list_identity()
    dict_identites = dict_identities_to_img()

    np.random.seed(3)
    for pair in tqdm(pairs):
        id1 = pair[0]
        id2 = pair[1]
        img1 = imgs[id1]
        img2 = imgs[id2]

        indexes_columns = np.random.randint(0, len(img1) - 16, number_patches)
        indexes_rows = np.random.randint(16, len(img1), number_patches)

        gen_identity = list_id[id1]
        identity_imgs = dict_identites[gen_identity]

        # select image for patching
        imp_ids = np.delete(list(imgs.keys()), identity_imgs)
        id_imp = np.random.choice(imp_ids, 1)[0]
        img_imp = imgs[id_imp]

        for i in np.stack((indexes_columns, indexes_rows), axis=1):
            row = i[1]
            column = i[0]
            img1 = insert(row, column, img1, img_imp)

        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEST_PATH, str(id1) + ".jpg"), img1)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(DEST_PATH, str(id2) + ".jpg"), img2)

        index_list.append([id1, id2, id_imp, indexes_rows, indexes_columns])

        # write patch indexes in file
        with open('./patches/patch_idx_gen.txt', 'w') as f:
            f.write(f"id1, id2 \t patch_id \t row indices \t column indices \n")
            for item in index_list:
                f.write(f"{item[0]}, {item[1]} \t {item[2]} \t {list(item[3])} \t {list(item[4])} \n")


if __name__ == "__main__":
    pairs, gen_imp = get_pairs_genimp()
    imposter_pairs = pairs[np.where(gen_imp == -1)]
    patch_imp_pair(imposter_pairs, NUM_PATCHES)

    genuine_pairs = pairs[np.where(gen_imp == 1)]
    patch_gen_pair(genuine_pairs, NUM_PATCHES)
