"""Helper functions for the document shadow.

Defines helper functions that are used in the document shadow
"""


import matplotlib.pyplot as plt
import numpy as np


def show_img_compare(img_1, img_2, title1, title2):
    """Show the comparison of two images.

    Input
    ----------
    img_1,img_2: numpy array
    title_1,title_2: str

    Output
    ----------
    Two image plots are shown side by side.
    """
    f, ax = plt.subplots(1, 2, figsize=(6, 6))
    ax[0].imshow(img_1)
    ax[0].set_title(title1)
    ax[0].axis('off')
    
    ax[1].imshow(img_2)
    ax[1].set_title(title2)
    ax[1].axis('off')

    f.tight_layout()
    plt.show()

    return f, ax
