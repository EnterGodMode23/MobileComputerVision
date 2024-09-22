import cv2
import numpy as np
from matplotlib import pyplot as plt

from utils.is_prime import is_prime
from algs.with_open_cv import with_open_cv
from algs.without_open_cv import without_open_cv


def start_benchmark():
    image_path = 'image.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sigma = 4

    primes = [x for x in range(1, 202) if is_prime(x) and x % 2 != 0]

    times_with_cv = []
    times_without_cv = []

    for kernel_size in primes:
        time_cv = with_open_cv(image, kernel_size, sigma, True)
        time_no_cv = without_open_cv(image, kernel_size, sigma, True)

        times_with_cv.append(time_cv)
        times_without_cv.append(time_no_cv)

    plt.plot(primes, times_with_cv, label='With OpenCV')
    plt.plot(primes, times_without_cv, label='Without OpenCV')
    plt.xlabel('Kernel Size')
    plt.ylabel('Time (seconds)')
    plt.title('OpenCV time vs Custom algorithm time')
    plt.legend()

    plt.xticks(np.arange(min(primes), max(primes) + 1, 2))
    plt.yticks(np.arange(0, max(times_without_cv + times_with_cv) + 0.5, 0.5))

    plt.show()


if __name__ == '__main__':
    start_benchmark()
