import numpy as np
from matplotlib import pyplot as plt

from core.depth import Depth
from core.math import Math
from core.models import RegionPrompt, Box
from image_utils import ImageUtils


class PlotUtils:
    @staticmethod
    def draw_circle(image, x, y, radius=1):
        fig, ax = plt.subplots()

        ax.imshow(image)

        circle = plt.Circle((x, y), radius, fill=False, color="red")
        ax.add_patch(circle)

        ax.set_aspect("equal")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)  # invert y-axis for images

        plt.show()

    # @staticmethod
    # def draw_regions_center(image_source, regions: list[RegionPrompt], radius=1):
    #     fig, ax = plt.subplots()
    #     ax.imshow(image_source)
    #
    #     image_source_center_x, image_source_center_y = ImageUtils.get_center(image_source)
    #
    #     circle = plt.Circle((image_source_center_x, image_source_center_y), radius, fill=False, color="red")
    #     ax.add_patch(circle)
    #     plt.show()
    #
    #     for region in regions:
    #         x, y = region.box.get_relative_center_location()
    #         circle = plt.Circle((x, y), radius, fill=False, color="green")
    #         ax.add_patch(circle)
    #         plt.show()

    @staticmethod
    def draw_regions_center(image_source, regions: list[RegionPrompt], radius=1):
        fig, ax = plt.subplots()
        ax.imshow(image_source)

        img_source_center = ImageUtils.get_center(image_source)
        ax.add_patch(
            plt.Circle(
                (img_source_center.x, img_source_center.y),
                radius,
                fill=False,
                color="red"
            )
        )

        for region in regions:
            center = region.box.get_relative_center_location()
            distance_from_center = Depth.calculate_distance(img_source_center, center)

            # Draw region center
            ax.add_patch(
                plt.Circle((center.x, center.y), radius, fill=False, color="green")
            )

            # Add distance label next to the circle
            ax.text(
                center.x + radius + 2,  # small horizontal offset
                center.y,
                f"{distance_from_center:.2f}",
                color="green",
                fontsize=9,
                verticalalignment="center"
            )

            print(f"Region {region}, distance from center: {distance_from_center:.2f}")

        plt.axis("off")
        plt.show()

    @staticmethod
    def show_mask_depth_with_center_distance(mask: np.ndarray,
                                             image_source_depth: np.ndarray,
                                             region_label: str,
                                             vmin=None,
                                             vmax=None,
                                             region_box:Box=None,
                                             title="Mask Depth + Center Distance"):
        """
        Displays masked depth and 2D distance from the image center,
        and overlays the region label at the mask centroid.
        """

        if vmin is None:
            vmin = np.nanmin(image_source_depth)
        if vmax is None:
            vmax = np.nanmax(image_source_depth)

        h, w = image_source_depth.shape[:2]

        # Image center (pixel coordinates)
        center_x = w / 2.0
        center_y = h / 2.0

        # Masked depth
        depth_masked = np.full_like(image_source_depth, np.nan, dtype=float)
        depth_masked[mask] = image_source_depth[mask]

        # 2D distance-from-center map
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        distance_map = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

        distance_masked = np.full_like(distance_map, np.nan, dtype=float)
        distance_masked[mask] = distance_map[mask]

        # Statistics
        median_depth = np.nanmedian(depth_masked)
        median_distance = np.nanmedian(distance_masked)

        # Mask centroid (for label placement)
        ys, xs = np.where(mask)
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)

        # Plot
        fig, ax = plt.subplots()
        im = ax.imshow(depth_masked, cmap="inferno", vmin=vmin, vmax=vmax)

        # Image center marker
        ax.scatter(center_x, center_y, c="cyan", s=40, marker="+", label="Image Center")

        # Region label + distance
        ax.text(
            centroid_x,
            centroid_y,
            f"{region_label}\nDist: {median_distance:.2f}px",
            color="white",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none")
        )

        box_center = region_box.get_relative_center_location()
        # # draw circle in the center
        ax.add_patch(
            plt.Circle((box_center.x, box_center.y), radius=5, fill=False, color="red")
        )

        ax.set_title(
            f"{title}\n"
            f"Median Depth: {median_depth:.2f} | "
            f"Median 2D Distance: {median_distance:.2f}px"
        )

        plt.colorbar(im, ax=ax, label="Depth")
        ax.legend(loc="upper right")
        ax.axis("off")
        plt.show()



