from dataclasses import dataclass

import numpy as np

@dataclass
class Vector2:
    x: float
    y: float

@dataclass
class CenterRectangle:
    width: float
    height: float
    x_min: float
    y_min: float


@dataclass
class Box:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def get_relative_center_location(self):
        w = self.get_width()
        h = self.get_height()
        x = self.x_min + w / 2
        y = self.y_min + h / 2
        return Vector2(x, y)

    def get_coordinates_as_array(self):
        return self.x_min, self.y_min, self.x_max, self.y_max

    def get_height(self):
        return abs(self.y_min - self.y_max)

    def get_width(self):
        return abs(self.x_max - self.x_min)

    def get_area(self):
        return self.get_height() * self.get_width()

@dataclass
class RegionPrompt:
    box: Box
    label: str
    mask: np.ndarray = None

    def __str__(self):
        return self.label

    def __hash__(self):
        location = self.box.get_relative_center_location()
        area = self.box.get_area()
        return hash((location.x, location.y, area))


