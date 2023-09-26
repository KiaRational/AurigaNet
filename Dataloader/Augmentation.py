import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
import numpy as np

def add_random_shadow(image, min_alpha=0.25, max_alpha=0.45):
    """
    Adds a random shadow to a random region of the input image.

    Args:
    - image (numpy.ndarray): The input image.
    - min_alpha (float, optional): The minimum opacity of the shadow. Default is 0.5.
    - max_alpha (float, optional): The maximum opacity of the shadow. Default is 0.75.

    Returns:
    - shadowed_image (numpy.ndarray): The input image with a random shadow added.
    """

    shadowed_image = image.copy()

    # Select random region for the shadow
    top_x, bottom_x = np.random.randint(0, image.shape[1], 2)
    rand = np.random.randint(5)
    vertices = np.array([[top_x, 0], [0, 0], [0, image.shape[0]], [bottom_x, image.shape[0]]], dtype=np.int32)
    if rand == 1:
        vertices = np.array([[top_x, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [bottom_x, image.shape[0]]], dtype=np.int32)
    # Create mask for the shadow
    mask = np.zeros_like(image)
    channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (0,) * channel_count
    cv2.fillPoly(mask, [vertices],(255,255,255))
    # Apply shadow to image
    rand_alpha = np.random.uniform(min_alpha, max_alpha)
    cv2.addWeighted(mask, rand_alpha, image, 1 - rand_alpha, 0., shadowed_image)

    return shadowed_image

class RandomShadow(ImageOnlyTransform):
    """
    RamdomA transformation
    """
    def __init__(self, safe_db_lists=[], prob =0.5) -> None:
        super(RandomShadow, self).__init__()
        self.safe_db_lists = safe_db_lists
        self.prob = prob

    def apply(self, img, copy=True, **params):
        if np.random.uniform(0, 1) > self.prob:
            return add_random_shadow(img)
        if copy:
            img = img.copy()
        return img



