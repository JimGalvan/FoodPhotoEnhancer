import logging

import numpy as np
from matplotlib import pyplot as plt
from core.constants import Constants
from core.models import RegionPrompt, CenterRectangle, Vector2, Box
from image_utils import ImageUtils
from core.math import Math


class Depth:
    logger = logging.getLogger(__name__)

    @staticmethod
    def calculate_distance(point1: Vector2, point2: Vector2):
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    @staticmethod
    def normalize_depth(depth: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
        # choose normalization range
        if vmin is None:
            vmin = np.nanmin(depth)
        if vmax is None:
            vmax = np.nanmax(depth)

        # normalize to [0, 1]
        depth_norm = (depth - vmin) / (vmax - vmin)
        return np.clip(depth_norm, 0, 1)

    @staticmethod
    def compute_mask_nan_median(total_depth, mask):
        depth_masked = np.full_like(total_depth, np.nan, dtype=float)
        depth_masked[mask] = total_depth[mask]
        return np.nanmedian(depth_masked)

    @staticmethod
    def compute_mask_stats(total_depth, mask):
        masked_values = total_depth[mask]

        return {
            "min": masked_values.min(),
            "max": masked_values.max(),
            "median": np.median(masked_values),
        }

    @staticmethod
    def compute_mask_depth_stats(total_depth, mask):
        masked_values = total_depth[mask]
        stats = {
            "farthest_depth": masked_values.min(),
            "closest_depth": masked_values.max(),
            "mean_depth": np.mean(masked_values),
        }
        Depth.logger.debug("closest_depth: %s", stats.get("closest_depth"))
        Depth.logger.debug("farthest_depth: %s", stats.get("farthest_depth"))
        Depth.logger.debug("mean_depth: %s", stats.get("mean_depth"))
        return stats

    @staticmethod
    def compute_depth_in_range(total_depth: np.ndarray, mask: np.ndarray, range_end=1.5) -> float:
        stats = Depth.compute_mask_depth_stats(total_depth, mask)
        closest_depth = stats["closest_depth"]
        farthest_depth = stats["farthest_depth"]
        masked_depth = total_depth[mask]
        custom_depth_end_range = (closest_depth - farthest_depth) / range_end
        range_mask = (masked_depth >= farthest_depth) & (masked_depth <= custom_depth_end_range)
        percentage = (np.sum(range_mask) / masked_depth.size) * 100

        Depth.logger.debug(f"custom_depth_end_range: {custom_depth_end_range}")
        Depth.logger.debug(
            f"Percentage of depth values between {farthest_depth} and {custom_depth_end_range}: {percentage:.2f}%")
        return percentage

    @staticmethod
    def merge_regions_mask(regions: list[RegionPrompt]) -> RegionPrompt:
        curr_subject_mask = None
        for region in regions:
            if curr_subject_mask is None:
                curr_subject_mask = region.mask.copy()
            else:
                curr_subject_mask |= region.mask

        min_xy, max_xy = Depth.get_mask_box_dimension(curr_subject_mask)
        box = Box(min_xy.x, min_xy.y, max_xy.x, max_xy.y)
        return RegionPrompt(box=box, mask=curr_subject_mask, label="curr_subject_mask")

    @staticmethod
    def get_supported_subject_regions(
            depth,
            main_dish,
            regions_with_masks,
            support_threshold=0.6,
            dmin=None,
            dmax=None,
    ):
        masked_depth = depth[main_dish.mask]
        d_far = np.percentile(masked_depth, 5)
        d_near = np.percentile(masked_depth, 95)
        ten_p = np.percentile(masked_depth, 10)

        Depth.logger.debug(f"d_far: {d_far}, d_near: {d_near}")

        subject_regions = []
        for region in regions_with_masks:
            media_depth = Depth.compute_mask_nan_median(depth, region.mask)
            in_range = d_far < media_depth < d_near
            supported = Math.compute_is_overlap(
                region.box, main_dish.box, support_threshold
            )

            if in_range and supported:
                subject_regions.append(region)
                if Constants.DEBUG:
                    ImageUtils.show_mask_depth(
                        mask=region.mask,
                        image_source_depth=depth,
                        vmin=dmin,
                        vmax=dmax,
                        title=region.label,
                    )

        return subject_regions

    @staticmethod
    def compute_distance_to_center():
        pass

    @staticmethod
    def compute_main_dish(depth, regions_with_masks: list[RegionPrompt], image_source: np.ndarray) -> RegionPrompt:
        area_w = 0.5
        median_w = 0.9
        dist_w = 0.9
        scores = []
        candidates = {}
        image_center = ImageUtils.get_center(image_source)
        areas = {}
        distances = {}
        median_depths = {}

        for region in regions_with_masks:
            if region.label in Constants.CONTAINER_PROMPTS:
                box_center = region.box.get_relative_center_location()
                distance = Depth.calculate_distance(image_center, box_center)
                area = np.sum(region.mask)
                median_depth = Depth.compute_mask_nan_median(depth, region.mask)
                obj_hash = hash(region)
                areas[obj_hash] = area
                distances[obj_hash] = distance
                median_depths[obj_hash] = median_depth

        normalized_areas = Math.min_max_normalize_hashmap(areas)
        normalized_distances = Math.min_max_normalize_hashmap(distances)
        normalized_median_depths = Math.min_max_normalize_hashmap(median_depths)

        for region in regions_with_masks:
            if region.label in Constants.CONTAINER_PROMPTS:
                obj_hash = hash(region)
                norm_area = normalized_areas[obj_hash]
                norm_distance = normalized_distances[obj_hash]
                norm_median_depth = normalized_median_depths[obj_hash]
                score = (norm_area * area_w) + (median_w * float(norm_median_depth)) + np.exp(-norm_distance / dist_w)
                scores.append(score)
                candidates[score] = region

        max_score = max(scores)
        return candidates[max_score]

    @staticmethod
    def computer_center_rectangle(image_source: np.ndarray, scale=0.5) -> CenterRectangle:
        height, width = image_source.shape[:2]

        # Box dimensions
        box_w = scale * width
        box_h = scale * height

        # Image center
        cx = width / 2
        cy = height / 2

        # top left coordinates
        x_min = int(cx - box_w / 2)
        y_min = int(cy - box_h / 2)

        return CenterRectangle(width=box_w, height=box_h, x_min=x_min, y_min=y_min)

    @staticmethod
    def draw_center_box(image_source: np.ndarray, scale=0.5):
        height, width = image_source.shape[:2]

        # Box dimensions
        box_w = scale * width
        box_h = scale * height

        # Image center
        cx = width / 2
        cy = height / 2

        # top left coordinates
        x_min = int(cx - box_w / 2)
        y_min = int(cy - box_h / 2)

        plt.figure(figsize=(10, 10))
        plt.imshow(image_source)
        ax = plt.gca()

        rect = plt.Rectangle(
            (x_min, y_min),
            box_w,
            box_h,
            edgecolor="green",
            facecolor="none",
            linewidth=2
        )

        ax.add_patch(rect)
        ax.axis("off")
        plt.show()

    @staticmethod
    def get_mask_box_dimension(mask: np.ndarray) -> tuple[Vector2, Vector2]:
        height, width = mask.shape[:2]
        return Vector2(0, 0), Vector2(width, height)

    @staticmethod
    def find_related_dish_region(
            depth,
            main_dish,
            regions_with_masks,
            dmin,
            dmax,
            init_support_threshold=0.6,
            reduction_rate=0.9,
            max_attempts=3,
            min_increase=0.6,
    ):
        curr_threshold = init_support_threshold
        attempt = 1
        prev_percentage = None
        had_significant_increase = False
        curr_region = None
        thresholds = []
        coverages = []

        while attempt <= max_attempts:
            subject_regions = Depth.get_supported_subject_regions(
                depth=depth,
                main_dish=main_dish,
                regions_with_masks=regions_with_masks,
                support_threshold=curr_threshold,
                dmin=dmin,
                dmax=dmax,
            )

            if not subject_regions:
                break

            curr_region = Depth.merge_regions_mask(subject_regions)

            curr_percentage = Depth.compute_depth_in_range(
                depth,
                curr_region.mask
            )

            if prev_percentage is not None and prev_percentage > 0:
                increase = (curr_percentage - prev_percentage) / prev_percentage
                if increase >= min_increase:
                    had_significant_increase = True
                elif had_significant_increase:
                    break

            criterion = curr_percentage / 100.0
            curr_threshold = Math.exp_decay(
                start_threshold=curr_threshold,
                criterion=criterion,
                reduction_rate=reduction_rate
            )

            thresholds.append(curr_threshold)
            coverages.append(curr_percentage)

            prev_percentage = curr_percentage
            attempt += 1

        return curr_region
