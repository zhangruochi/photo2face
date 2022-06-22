#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/photo2face/photo2face.py
# Project: /Users/ZRC
# Created Date: Tuesday, June 21st 2022, 4:10:36 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Wed Jun 22 2022
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2022 Ruochi Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###

import pickle as pkl
import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment


def classify_hist_with_split(image1, image2, size=(32, 32)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate_sim(im1, im2)
        sub_data = sub_data/3
    return sub_data


def calculate_sim(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + \
                (1 - abs(hist1[i]-hist2[i])/max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
            degree = degree/len(hist1)
    return degree[0]


def sort_list(result_list):

    return sorted(result_list)


def construct_match_graph(result_dict, block_shape):
    all_images = [_[-1] for _ in result_dict[0].keys()]
    id2img = {id: img for id, img in zip(range(len(all_images)), all_images)}
    cost_matrix = np.empty((block_shape[0] * block_shape[1], len(id2img)))

    all_dict = {}
    for dic_ in result_dict:
        all_dict.update(dic_)

    for i in range(block_shape[1]):
        for j in range(block_shape[0]):
            for id, file in id2img.items():
                r = i*block_shape[1] + j
                cost_matrix[r, id] = all_dict[(i, j, file)]

    return cost_matrix, id2img


def his_match(src, dst):
    res = np.zeros_like(dst)
    cdf_src = np.zeros(256)
    cdf_dst = np.zeros(256)
    cdf_res = np.zeros(256)

    kw = dict(bins=256, range=(0, 256), normed=True)

    his_src, _ = np.histogram(src[:, :], **kw)
    hist_dst, _ = np.histogram(dst[:, :], **kw)
    cdf_src = np.cumsum(his_src)
    cdf_dst = np.cumsum(hist_dst)
    index = np.searchsorted(cdf_src, cdf_dst, side='left')
    np.clip(index, 0, 255, out=index)
    res[:, :] = index[dst[:, :]]
    his_res, _ = np.histogram(res[:, :], **kw)
    cdf_res = np.cumsum(his_res)

    return res, (cdf_src, cdf_dst, cdf_res)


def get_mean_and_std(img):
    x_mean, x_std = cv2.meanStdDev(img)
    x_mean = np.hstack(np.around(x_mean, 2))
    x_std = np.hstack(np.around(x_std, 2))
    return x_mean, x_std


def color_transfer(sc, dc):
    sc = cv2.cvtColor(sc, cv2.COLOR_BGR2LAB)
    s_mean, s_std = get_mean_and_std(sc)
    dc = cv2.cvtColor(dc, cv2.COLOR_BGR2LAB)
    t_mean, t_std = get_mean_and_std(dc)
    img_n = ((sc-s_mean)*(t_std/s_std))+t_mean
    np.putmask(img_n, img_n > 255, 255)
    np.putmask(img_n, img_n < 0, 0)
    dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2BGR)
    return dst


def reconstruct(result_dict, block_shape, target_img_path, output_path="./reconstruct.png"):

    target_img = cv2.imread(target_img_path)
    target_img = cv2.resize(
        target_img, (target_img.shape[1] * 20, target_img.shape[0] * 20))

    h, w, _ = target_img.shape

    num_row, num_column = block_shape
    h_, w_ = h // num_row, w // num_column

    if not isinstance(result_dict, list):
        with open(result_dict, "rb") as f:
            result_dict = pkl.load(f)[0]

    cost_matrix, id2img = construct_match_graph(result_dict, block_shape)

    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
    row_ind, col_ind = list(row_ind), list(col_ind)

    canvas = np.ones((h_ * num_row + (num_row + 1), w_ *
                     num_column + (num_column+1), 3)) * 255

    idx = 0
    start_row = 1
    for j in tqdm(range(num_row), total=num_row):
        end_row = start_row + h_
        start_col = 1
        # put imgs on every row
        for i in range(num_column):
            end_col = start_col + w_
            loc = row_ind.index(idx)

            img = cv2.imread(str(id2img[col_ind[loc]]))
            src = cv2.resize(img, (w_, h_))
            dst = target_img[j*h_: j*h_+h_, i*w_:i*w_ + w_, :]
            src = color_transfer(src, dst)

            # res_src, (cdf_src, cdf_dst, cdf_res) = his_match(dst, src)

            canvas[start_row:end_row, start_col:end_col, :] = src
            start_col = end_col + 1
            idx += 1

        start_row = end_row + 1

    cv2.imwrite(output_path, canvas)

    return canvas


if __name__ == "__main__":

    reconstruct("result_map.pkl", block_shape=(32, 32),
                target_img_path="anchor.jpeg", output_path="./reconstruct.png")
