from core.models import Box


class Math:
    @staticmethod
    def get_center(width, height):
        center_x = width / 2
        center_y = height / 2
        return center_x, center_y

    @staticmethod
    def min_max_normalize_hashmap(dataset: dict):
        normalized = {}
        min_val = min(dataset.values())
        max_val = max(dataset.values())
        for key in dataset.keys():
            val = dataset[key]
            normalized[key] = (val - min_val) / (max_val - min_val)
        return normalized

    @staticmethod
    def calc_rel_overlap_ratio(a: Box, b: Box):
        inter_x_min = max(a.x_min, b.x_min)
        inter_y_min = max(a.y_min, b.y_min)
        inter_x_max = min(a.x_max, b.x_max)
        inter_y_max = min(a.y_max, b.y_max)
        inter_w = inter_x_max - inter_x_min
        inter_h = inter_y_max - inter_y_min
        inter_area = inter_w * inter_h
        a_area = a.get_area()
        return inter_area / a_area

    @staticmethod
    def compute_is_overlap(a: Box, b: Box, threshold: float):
        ratio = Math.calc_rel_overlap_ratio(a, b)
        return ratio >= threshold
