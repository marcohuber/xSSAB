import os

import numpy as np
from tqdm import tqdm

from evaluation.evaluate import compute_FNMR, compute_FMR
from methodology.face_matching_module import FaceMatching
from methodology.gradient_calculator import get_cosine
from utils_os import get_pairs_genimp, load_all_images, image_iter, get_model

###############################################################
# Specify model and model path.
MODEL_PATH = "./models/295672backbone.pth"  #
MODELNAME = 'ElasticArcface'  # 'Arcface_xcos'
###############################################################

PATH_THR = f"./methodology/cos_sim/ds_lfw_mag_mtcnn/{MODELNAME}_thr.npy"
PATH_ORIGINAL_COS = f"./methodology/cos_sim/ds_lfw_mag_mtcnn/{MODELNAME}_cos.npy"
SRC_PATH_IMG = "./patches/PatchLFW"
DEST_PATH_COS = "./methodology/cos_sim/patch/0_percent"

if not os.path.exists(DEST_PATH_COS):
    os.makedirs(DEST_PATH_COS)


def main():
    """Compute FMR and FNMR of model."""
    thr = np.load(PATH_THR)

    pairs, gen_imp = get_pairs_genimp()
    imgs = load_all_images(image_iter(SRC_PATH_IMG))

    backbone = get_model(MODEL_PATH, MODELNAME)
    model = FaceMatching(backbone, MODELNAME)
    backbone.eval()
    model.eval()

    cos = []
    for pair in tqdm(pairs):
        id1 = pair[0]
        id2 = pair[1]
        img1 = imgs[id1]
        img2 = imgs[id2]

        c, _ = get_cosine(img1, img2, model, backbone, 0, MODELNAME, thr)
        cos.append(c.detach().cpu().numpy()[0])
    cos = np.array(cos)
    np.save(os.path.join(DEST_PATH_COS, f'{MODELNAME}_cos.npy'), cos)
    np.save(os.path.join(DEST_PATH_COS, f'{MODELNAME}_pairs.npy'), pairs)
    np.save(os.path.join(DEST_PATH_COS, f'{MODELNAME}_gen_imp.npy'), gen_imp)

    # original fmr, fnmr
    cos_original = np.load(PATH_ORIGINAL_COS)
    print('FMR and FNMR of LFW ---------------')
    fnmr = compute_FNMR(cos_original, gen_imp, thr)
    print("FNMR:", fnmr)
    fmr = compute_FMR(cos_original, gen_imp, thr)
    print('FMR:', fmr)

    print('FMR and FNMR of PatchLFW ---------------')
    fnmr = compute_FNMR(cos, gen_imp, thr)
    print("FNMR:", fnmr)
    fmr = compute_FMR(cos, gen_imp, thr)
    print('FMR:', fmr)


if __name__ == "__main__":
    main()
