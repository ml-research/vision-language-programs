import argparse
import os
import json
import sys

import torch
from tqdm import tqdm
from rtpt import RTPT
from utils.dataset_utils import load_data, load_dataset

from models.internvl.main import InternVLPrompter

# from vlm.gpt.prompt_llm import GPT4Prompter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
from PIL import Image
import requests
from io import BytesIO


def plot_train_imgs(data_sample, id, dataset, title="", target_path=None):
    pos_imgs_paths, neg_imgs_paths, pos_test_imgs, neg_test_imgs, gt = data_sample

    if len(pos_imgs_paths) >= 6:
        pos_imgs_paths = pos_imgs_paths[:6]
    if len(neg_imgs_paths) >= 6:
        neg_imgs_paths = neg_imgs_paths[:6]
    if len(pos_test_imgs) >= 1:
        pos_test_imgs = pos_test_imgs[:1]
    if len(neg_test_imgs) >= 1:
        neg_test_imgs = neg_test_imgs[:1]

    image_paths = pos_imgs_paths + pos_test_imgs + neg_imgs_paths + neg_test_imgs

    plot_only_train_imgs(data_sample, id, dataset, title=title, target_path=target_path)


def plot_only_train_imgs(data_sample, id, dataset, title="", target_path=None):
    pos_imgs_paths, neg_imgs_paths, pos_test_imgs, neg_test_imgs, gt_rule = data_sample

    print(pos_imgs_paths)
    print(gt_rule)

    if len(pos_imgs_paths) >= 6:
        pos_imgs_paths = pos_imgs_paths[:6]
    if len(neg_imgs_paths) >= 6:
        neg_imgs_paths = neg_imgs_paths[:6]

    image_paths = pos_imgs_paths + neg_imgs_paths

    plot_images(image_paths, id, dataset, title=title, target_path=target_path)


def plot_images(image_paths, id, dataset, title="", target_path=None):

    if len(image_paths) == 12:
        fig, axs = plt.subplots(3, 4, figsize=(30, 20))
    else:  # 14 images
        fig, axs = plt.subplots(4, 4, figsize=(30, 40))
    # axs = axs.flatten()

    for i, path in enumerate(image_paths):
        if i <= 6:
            x = i % 2
            y = i // 2
        else:
            x = i % 2 + 2
            y = (i - 7) // 2

        # Open the image
        img = Image.open(path)
        axs[y, x].imshow(img)
        # axs[y, x].set_title(f"Image {i + 1}")

        # Open the image
        img = Image.open(path)
        axs[y, x].imshow(img)
        # axs[y, x].set_title(f"Image {i + 1}")

    # remove axes
    for ax in axs.flat:
        ax.axis("off")

    # set title of figure
    if title != "":
        plt.suptitle(title, fontsize=16)

    plt.tight_layout()

    if target_path is None:
        target_path = f"results/qualitative/{dataset}/img_{id}.png"

    # create folder if it does not exist
    folder = os.path.dirname(target_path)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # save the figure
    plt.savefig(target_path, dpi=300)
    plt.show()


def plot_single_images(image_paths, id, dataset):

    for i, path in enumerate(image_paths):
        plt.figure(figsize=(10, 10))
        # Open the image
        img = Image.open(path)
        plt.imshow(img)
        plt.title(f"Image {i + 1}")

        # remove axes
        plt.axis("off")

        plt.tight_layout()

        # save img
        plt.savefig(f"results/qualitative/{dataset}/img_{id}_{i}.png")
        plt.show()


def main():

    data = load_data("bongard-op")

    for bp_id in range(4):
        plot_train_imgs(data[bp_id], bp_id, "bongard-op")


if __name__ == "__main__":
    main()
