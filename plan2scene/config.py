from dataclasses import dataclass


@dataclass
class Defaults:
    wall_height_meters: float = 3.0
    wall_thickness_meters: float = 0.2
    door_height_meters: float = 2.1
    window_height_meters: float = 1.2
    image_dpi: int = 300
    min_line_length_pixels: int = 30
    max_line_gap_pixels: int = 5
    canny_threshold1: int = 50
    canny_threshold2: int = 150
    hough_threshold: int = 50


DEFAULTS = Defaults()
