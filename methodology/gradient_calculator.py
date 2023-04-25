import numpy as np
import torch
import os

from torch import nn
from tqdm import tqdm

import utils_os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_embedding(img, model, model_name):
    """Compute embedding for img with model"""
    img = utils_os.transform_img(img)
    if model_name == 'CurricularFace':
        _embedding = model(img)[0]
    else:   # self.name in ['ElasticArcface', 'ElasticCosface']:
        _embedding = model(img)

    return _embedding, img


def get_cosine(img_1, img_2, model, model_emb, version, model_name, thr):
    """Compute cosine similarity of img_1 and img_2 with possibility to hide positive/negative features."""
    # get current embedding
    emb_1, _ = get_embedding(img_1, model_emb, model_name)
    emb_2, _ = get_embedding(img_2, model_emb, model_name)

    # normalize embedding
    emb_1 = nn.functional.normalize(emb_1, p=2.0, dim=1)
    emb_1 = emb_1.detach().cpu().numpy().squeeze()
    emb_2 = nn.functional.normalize(emb_2, p=2.0, dim=1)

    # remove irrelevant features
    if version == 0:  # standard cosine similarity computation
        emb_2 = emb_2.detach().squeeze()
        model.cosine_layer.weight = nn.Parameter(emb_2)
    elif version == 1:  # only keep positive features
        bound = thr / len(emb_1)
        emb_2 = emb_2.detach().cpu().numpy().squeeze()
        weights = torch.tensor([x[1] if x[0] >= bound else 0.0 for x in np.column_stack((np.multiply(emb_1, emb_2), emb_2))])
        weights = weights.to(device)
        model.cosine_layer.weight = nn.Parameter(weights)
    else:  # only keep negative features
        bound = thr / len(emb_1)
        emb_2 = emb_2.detach().cpu().numpy().squeeze()
        weights = torch.tensor([x[1] if x[0] < bound else 0.0 for x in np.column_stack((np.multiply(emb_1, emb_2), emb_2))])
        weights = weights.to(device)
        model.cosine_layer.weight = nn.Parameter(weights)

    img = utils_os.transform_img(img_1)
    cos = model.forward(img)
    return cos, img


def get_gradient(img1, img2, model, model_emb, version, model_name, thr):
    """Compute gradient of img1 in model."""
    cos, root_1 = get_cosine(img1, img2, model, model_emb, version, model_name, thr)

    feature = cos.squeeze()
    feature.backward(retain_graph=True)
    feature_gradients = root_1.grad
    fg = feature_gradients.detach().cpu().numpy().squeeze()
    fg = np.transpose(fg, (1, 2, 0))  # (height, width, channel)

    return fg


def load_gradient_mean(path_gradient):
    """
    Compute gradient map with RGB-channels combined via mean
    :param path_gradient: system path to gradient image
    :return: 2D matrix with RGB-channels combined with mean
    """
    gradient = np.load(path_gradient)
    return np.mean(np.abs(gradient), axis=2)


def save_combined_map(pairs, dest_path_grad):
    """combined gradient map of positive gradient map - negative gradient map."""
    for pair in pairs:
        id1 = int(pair[0])
        id2 = int(pair[1])

        # load gradient image 1
        grad_1_2_pos_path = os.path.join(dest_path_grad, '{}_{}_gradient_{}.npy'.format(id1, id2, "pos"))
        gradient_1_2_pos = load_gradient_mean(grad_1_2_pos_path)

        grad_1_2_neg_path = os.path.join(dest_path_grad, '{}_{}_gradient_{}.npy'.format(id1, id2, "neg"))
        gradient_1_2_neg = load_gradient_mean(grad_1_2_neg_path)

        combi_1_2 = gradient_1_2_pos - gradient_1_2_neg
        grad_1_2_combi_path = os.path.join(dest_path_grad, '{}_{}_gradient_{}.npy'.format(id1, id2, "combi"))
        np.save(grad_1_2_combi_path, combi_1_2)
