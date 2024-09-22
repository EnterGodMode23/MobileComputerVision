import numpy as np
import time


def gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size), dtype=float)
    center = size // 2
    sum_val = 0.0

    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
            sum_val += kernel[i, j]

    kernel /= sum_val

    return kernel


def apply_gaussian_blur(image, kernel):
    img_height, img_width = image.shape
    blurred_image = np.zeros_like(image, dtype=float)
    kernel_size = kernel.shape[0]
    pad = kernel_size // 2

    padded_image = np.pad(image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    for i in range(img_height):
        for j in range(img_width):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            blurred_image[i, j] = np.sum(region * kernel)

    return blurred_image[pad:pad + img_height, pad:pad + img_width]


def without_open_cv(image, kernel_size, sigma, only_duration=False):
    start_time = time.time()
    res_image = apply_gaussian_blur(image, gaussian_kernel(kernel_size, sigma))
    duration = time.time() - start_time

    if only_duration:
        return duration

    return res_image, duration
