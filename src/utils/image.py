#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image


def array2image(array):
    """Transform a numpy array to an Image"""

    if array.max() <= 1.0:
        array = array * 255.
    if len(array.shape) == 2 or len(array.shape) == 3 and array.shape[-1] == 1:
        image = Image.fromarray(array.astype("uint8").squeeze()).convert("L")
    else:
        image = Image.fromarray(array.astype("uint8")).convert("RGB")
    return image


def image2array(image_path):
    """Transform an image to a numpy array"""

    with Image.open(image_path) as image:
        array = np.array(image, dtype=np.float32)
    return array
