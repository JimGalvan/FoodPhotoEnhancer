from dataclasses import dataclass

import numpy as np


@dataclass
class RegionPrompt:
    box: list
    label: str
    mask: np.ndarray = None
