import ntpath
import os

import cv2
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

from backbones.CurricularFace.curr_resnet import curr_iresnet100
from backbones.ElasticArcface.iresnet import iresnet100
from evaluation.utils import compute_eer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_del_mask(grads, bound):
    """Return indices to be replaced."""
    mask = np.nonzero(grads > bound)
    return mask


def get_ins_mask(grads, bound):
    """Return indices to be replaced."""
    mask = np.nonzero(grads < bound)
    return mask


def load_all_images(all_paths):
    """Load images from paths."""
    imgs = {}
    for p in tqdm(all_paths):
        img = cv2.imread(p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs[int(ntpath.basename(p).split(".")[0].split("_")[0])] = img
    return imgs


def image_iter(path):
    """Return image paths in filepath."""
    image_paths = []

    for path, subdirs, files in os.walk(path):
        for name in files:
            image_paths.append(os.path.join(path, name))
    image_paths.sort()
    return image_paths


def get_image(path):
    """Load image from path."""
    img_bgr = cv2.imread(path)  # (112, 112, 3) aligned face with mtcnn

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_rgb = np.transpose(img_rgb, (2, 0, 1))  # (channel, height, width)
    img_rgb = np.asarray([img_rgb], dtype="float32")
    img_rgb = ((img_rgb / 255) - 0.5) / 0.5

    img_rgb = torch.tensor(img_rgb, requires_grad=True)

    return img_rgb


def transform_img(img):
    """Transform image."""
    img_rgb = np.transpose(img, (2, 0, 1))  # (channel, height, width)
    img_rgb = np.asarray([img_rgb], dtype="float32")

    img_rgb = ((img_rgb / 255) - 0.5) / 0.5

    img_rgb = torch.tensor(img_rgb, requires_grad=True, device=device)
    return img_rgb


def get_model(path_model, modelname):
    """Load model from path_model."""
    if not os.path.exists(path_model):
        raise Exception("Model file does not exist!", path_model)
    if modelname in ['ElasticArcface', 'ElasticCosface']:
        backbone = iresnet100(num_features=512).to(device)
    elif modelname == 'CurricularFace':
        backbone = curr_iresnet100().to(device)
    
    backbone.load_state_dict(torch.load(path_model, map_location=device))
    return backbone


def save_threshold(score, y_comp, thr_path):
    """Compute and save threshold."""
    far, tar, thr = metrics.roc_curve(y_comp, score)
    eer, threshold = compute_eer(far, tar, thr)
    np.save(thr_path, threshold)
    print('Threshold:', threshold)
    print('EER:', eer)
    return threshold


def get_pairs_genimp():
    """Returns image pair and genuine/imposter indication lists."""
    pairs = []
    g_i = []
    with open('./dataset/lfw_pair.txt') as f:
        for line in f:
            elems = line.split(" ")
            pairs.append(np.array([int(elems[0].split(".")[0]) - 1,
                                   int(elems[1].split(".")[0]) - 1]))
            g_i.append(int(elems[2]))
    return np.array(pairs), np.array(g_i)


def save_embedding(emb, img_path):
    """Save embedding."""
    name = img_path.split("/")[-1].split(".")[0]
    np.save("./embeddings/emb_" + name, emb)


def dict_identities_to_img():
    """Return dictionary for LFW: identity -> image"""
    dict_iden = {}
    img_counter = 0
    with open("./patches/lfw_iden_pairs.txt") as f:
        for line in f:
            elems = line.split("\t")
            if len(elems) == 3:  # gen pair
                if elems[0] in dict_iden:
                    dict_iden[elems[0]].append(img_counter)
                    dict_iden[elems[0]].append(img_counter + 1)
                    img_counter += 2
                else:
                    dict_iden[elems[0]] = [img_counter]
                    dict_iden[elems[0]].append(img_counter + 1)
                    img_counter += 2
            else:  # imp pair
                if elems[0] in dict_iden:
                    dict_iden[elems[0]].append(img_counter)
                    img_counter += 1
                else:
                    dict_iden[elems[0]] = [img_counter]
                    img_counter += 1
                if elems[2] in dict_iden:
                    dict_iden[elems[2]].append(img_counter)
                    img_counter += 1
                else:
                    dict_iden[elems[2]] = [img_counter]
                    img_counter += 1
    return dict_iden


def list_identity():
    """Returns a list of identities in LFW."""
    list_id = []
    with open("./patches/lfw_iden_pairs.txt") as f:
        for line in f:
            elems = line.split("\t")
            if len(elems) == 3:  # genuine pair
                list_id.extend((elems[0], elems[0]))
            else:  # imposter pair
                list_id.extend([elems[0], elems[2]])
    return np.array(list_id)
