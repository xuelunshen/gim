# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2


def read_image(h5file, name):
    image = cv2.imdecode(h5file[name][:], cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
