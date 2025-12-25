import numpy as np

from core.models import RegionPrompt
from core.constants import Constants


class Depth:

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
    def compute_main_dish(depth, regions_with_masks: list[RegionPrompt]) -> RegionPrompt:
        area_w = 1.0
        median_w = 1.0
        scores = []
        candidates = {}
        for region in regions_with_masks:
            if region.label in Constants.CONTAINER_PROMPTS:
                area = np.sum(region.mask)
                median_depth = Depth.compute_mask_nan_median(depth, region.mask)
                score = (area * area_w) + (median_w * float(median_depth))
                scores.append(score)
                candidates[score] = region
        max_score = max(scores)
        main_dish = candidates[max_score]
        return main_dish
