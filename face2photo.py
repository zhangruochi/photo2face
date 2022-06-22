#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/ZRC/Desktop/photo2face/face2photo.py
# Project: /Users/ZRC
# Created Date: Tuesday, June 21st 2022, 2:59:56 pm
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
# Copyright (c) 2022  Ruochi Zhang
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

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from utils import calculate_sim, reconstruct, classify_hist_with_split
from collections import defaultdict
import os
import multiprocessing
from multiprocessing import Pool
import pickle as pkl
from config import Config

from typing import List, Dict, Tuple

def worker(anchor_tuple: Tuple) -> Dict:
    """worker of multiprocessing. this function is used to compute similary score of a sub_img with all other images. 

    Args:
        anchor_tuple (Tuple): (row_id, col_id, sub_img)

    Returns:
        _type_: _description_
    """
    sim_map = {}
    i, j, anchor_sub = anchor_tuple

    for file in Path(Config.img_lib).glob("*.jpeg"):
        target_img = cv2.imread(str(file))
        sim_score = classify_hist_with_split(anchor_sub, target_img)
        sim_map[(i, j, str(file))] = sim_score
        
    print("finished... {}, {}".format(i, j))
    return sim_map


def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    result_list.append(result)


def generate_works(anchor_img_path: str, num_row: int, num_column: int) -> List:
    """generate works for multiprocessing

    Args:
        anchor_img_path (str): path of anchor image

    Returns:
        work_list (List): list of works, each of work is (row_id, col_id, sub_img)
    """
    anchor_array = cv2.imread(anchor_img_path)
    height, width = anchor_array.shape[0], anchor_array.shape[1]
    
    sub_height, sub_width = int(height / num_row), int(width / num_column)

    work_list = []
    for i_idx, i in enumerate(range(0, width, sub_width)):
        for j_idx, j in enumerate(range(0, height, sub_height)):
            anchor_sub = anchor_array[j:j + sub_height, i:i + sub_width, :]
            work_list.append((i_idx, j_idx, anchor_sub))
    return work_list

if __name__ == "__main__":

    num_row, num_column = Config.num_row, Config.num_column
    anchor_img_path = Config.anchor_img_path
    
    work_list = generate_works(anchor_img_path, num_row, num_column)
    result_list = []
    cpu_count = int(multiprocessing.cpu_count() * 1)

    print("total works: {}".format(len(work_list)))
    print("launch {} processes".format(cpu_count))
    
    pool = Pool(processes=cpu_count)
    pool.map_async(worker, work_list, callback = log_result)

    pool.close()
    pool.join()
    
    # with open("result_map.pkl", "wb") as f:
    #     pkl.dump(result_list, f)

    reconstruct(result_list, block_shape = (num_row, num_column), target_img_path = anchor_img_path, output_path=Config.output_path)
