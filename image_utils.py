import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class ImageUtils:

    @staticmethod
    def convert_img_source_to_pil(img: np.ndarray):
        return Image.fromarray(img)

    @staticmethod
    def display_pil_image(img: Image.Image):
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    @staticmethod
    def get_output_path(img, output_path="output"):
        return f"{output_path}/{img}"

    @staticmethod
    def pil_to_numpy(img: Image.Image):
        return np.array(img)

    @staticmethod
    def display_masks(masks: list[np.ndarray]):
        for mask in masks:
            plt.imshow(mask)
            plt.axis('off')
            plt.show()

