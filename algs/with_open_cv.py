import time

import cv2 as cv


def with_open_cv(image, kernel_size, sigma, only_duration=False):
    start_time = time.time()

    res_image = cv.GaussianBlur(src=image,
                                ksize=(kernel_size, kernel_size),
                                borderType=cv.BORDER_DEFAULT,
                                sigmaX=sigma,
                                sigmaY=sigma)

    duration = time.time() - start_time

    if only_duration:
        return duration

    return res_image, duration
