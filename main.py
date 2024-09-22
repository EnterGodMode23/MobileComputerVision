import cv2
import matplotlib.pyplot as plt

from utils.show_image import show_image
from algs.with_open_cv import with_open_cv
from algs.without_open_cv import without_open_cv

if __name__ == '__main__':
    image_path = 'image.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    kernel_size = 51
    sigma = 4

    plt.figure(figsize=(10, 10))

    plt.subplot2grid((2, 2), (0, 0), colspan=2)
    show_image(plt, image, 'Original image.')

    plt.subplot2grid((2, 2), (1, 0))
    cv_image, cv_duration = with_open_cv(image, kernel_size, sigma)
    show_image(plt, cv_image, 'With open CV.', cv_duration)

    plt.subplot2grid((2, 2), (1, 1))
    alg_image, alg_duration = without_open_cv(image, kernel_size, sigma)

    show_image(plt, alg_image, 'Custom alg.', alg_duration)

    plt.tight_layout()
    plt.show()
