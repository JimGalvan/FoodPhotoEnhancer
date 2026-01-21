import logging
import shutil
import uuid

import cv2
import matplotlib
import numpy as np
from PIL import Image
from django.core.files.uploadedfile import UploadedFile
from matplotlib import pyplot as plt, patches
import time
import os
from core.models import RegionPrompt, Box, Vector2

logger = logging.getLogger(__name__)


class ImageUtils:

    @staticmethod
    def convert_payload_file_to_bgr(photo: UploadedFile):
        file_bytes = np.frombuffer(photo.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode image: {photo.name}")
        return image


    @staticmethod
    def copy_to_local_tmp(src_path: str, suffix: str = ".jpg") -> str:
        """
        Copy a file to a unique local /tmp path and return the new path.
        Fails fast if the copy cannot complete.
        """
        logger.info(f"Copying file: {src_path}")
        dst_path = os.path.join("/tmp", f"{uuid.uuid4()}{suffix}")
        shutil.copyfile(src_path, dst_path)
        return dst_path

    @staticmethod
    def wait_for_image(image_path, timeout=30):
        logger.info(f"Waiting for image: {image_path}, timeout: {timeout}")
        start = time.time()
        while time.time() - start < timeout:
            if os.path.isfile(image_path):
                logger.info(f"Found image: {image_path}")
                return
            time.sleep(0.5)
        logger.warning(f"Timeout waiting for file: {image_path}")

    @staticmethod
    def get_center(img: np.ndarray) -> Vector2:
        h, w, _ = img.shape
        center_x = w / 2
        center_y = h / 2
        return Vector2(center_x, center_y)

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
    def display_bounding_boxes(image_source: np.ndarray, boxes: list[Box]):
        fig, ax = plt.subplots()
        ax.imshow(image_source if image_source.ndim == 3 else image_source, cmap="gray")

        for box in boxes:
            x1 = box.x_min
            y1 = box.y_min
            width = box.get_width()
            height = box.get_height()

            ax.add_patch(
                patches.Rectangle(
                    (x1, y1),
                    width,
                    height,
                    fill=False,
                    edgecolor="lime",
                    linewidth=2
                )
            )

        ax.axis("off")
        plt.show()

    @staticmethod
    def depth_to_cmap(depth, vmin=None, vmax=None):
        cmap = matplotlib.colormaps.get_cmap("Spectral_r")

        # choose normalization range
        if vmin is None:
            vmin = np.nanmin(depth)
        if vmax is None:
            vmax = np.nanmax(depth)

        # normalize to [0, 1]
        depth_norm = (depth - vmin) / (vmax - vmin)
        depth_norm = np.clip(depth_norm, 0, 1)

        # apply colormap (RGBA in [0,1])
        colored_depth = cmap(depth_norm)[..., :3]

        # convert to uint8 RGB
        colored_depth = (colored_depth * 255).astype(np.uint8)

        return colored_depth

    @staticmethod
    def mask_depth_stat(depth, mask):
        return np.nanmedian(depth[mask])

    @staticmethod
    def show_mask_depth(mask: np.ndarray,
                        image_source_depth: np.ndarray,
                        vmin=None,
                        vmax=None,
                        title="Mask Depth"):

        if vmin is None:
            vmin = np.nanmin(image_source_depth)
        if vmax is None:
            vmax = np.nanmax(image_source_depth)

        # create masked depth with NaNs
        depth_masked = np.full_like(image_source_depth, np.nan, dtype=float)
        depth_masked[mask] = image_source_depth[mask]

        media_depth = np.nanmedian(depth_masked)

        plt.imshow(depth_masked, cmap="inferno", vmin=vmin, vmax=vmax)
        plt.title(f"{title} - Median Depth: {media_depth}")
        plt.colorbar()
        plt.show()

    @staticmethod
    def show_depth(depth, title=None, vmin=None, vmax=None, cmap="inferno"):
        if vmin is None:
            vmin = np.nanmin(depth)
        if vmax is None:
            vmax = np.nanmax(depth)

        plt.imshow(depth, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.colorbar()
        if title:
            plt.title(title)
        plt.show()

    @staticmethod
    def to_float01(image):
        if image.dtype != np.float32:
            return image.astype(np.float32) / 255.0
        return image

    @staticmethod
    def to_uint8(image):
        return (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    @staticmethod
    def gaussian_blur(image, sigma):
        return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)

    @staticmethod
    def compare_images(image_source: np.ndarray, enhanced: np.ndarray):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image_source)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(enhanced)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
