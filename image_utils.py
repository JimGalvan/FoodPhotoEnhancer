import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, patches

from core.models import RegionPrompt


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

    @staticmethod
    def display_regions_with_masks(regions: list[RegionPrompt]):
        for region in regions:
            plt.imshow(region.mask)
            plt.title(region.label)
            plt.axis('off')
            plt.show()

    @staticmethod
    def display_bounding_boxes(image_source: np.ndarray, boxes: list):
        fig, ax = plt.subplots()
        ax.imshow(image_source if image_source.ndim == 3 else image_source, cmap="gray")

        for x1, y1, x2, y2 in boxes:
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2
                )
            )

        ax.axis("off")
        plt.show()

