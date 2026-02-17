from .geometric_augmentations import (
    rotate_randomly,
)

ALL_AUGMENTATIONS = [
    # Geometric augmentations
    "rotate_randomly",
]


# Define __all__ to control what gets imported with "from augmentations import *"
__all__ = ALL_AUGMENTATIONS
