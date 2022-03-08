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
    f, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0].imshow(img_1)
    ax[0].set_title(title1)
    ax[0].axis('off')

    ax[1].imshow(img_2)
    ax[1].set_title(title2)
    ax[1].axis('off')

    f.tight_layout()
    plt.show()

    return f, ax


def get_global_colour_1(image,is_0_255):
    # Global average over each channel  (Approach #1)  
    global_col = np.zeros(image.shape)
    global_col[:,:,0], global_col[:,:,1], global_col[:,:,2] = np.average(image, axis=(0,1))
    if is_0_255:
        global_col = global_col.astype(int)
    return global_col

def get_global_colour_2(image,is_0_255):
    # Max pixel value for each channel  (Approach #2)
    global_col = np.zeros(image.shape)
    global_col[:,:,0] = np.ones(image.shape[0:2]) * np.max(image[:,:,0])
    global_col[:,:,1] = np.ones(image.shape[0:2]) * np.max(image[:,:,1])
    global_col[:,:,2] = np.ones(image.shape[0:2]) * np.max(image[:,:,2])
    if is_0_255:
        global_col = global_col.astype(int)
    return global_col
    
def get_global_colour_3(image,is_0_255):
    # Average of top 50 pixels   (Approach #3)
    global_col = np.zeros(image.shape)

    # Extracting dominant pixels
    dom_r = np.partition(image[:,:,0].flatten(), -50)[-50:]
    mean_val_r = np.mean(dom_r)

    dom_g = np.partition(image[:,:,1].flatten(), -50)[-50:]
    mean_val_g = np.mean(dom_g)

    dom_b = np.partition(image[:,:,2].flatten(), -50)[-50:]
    mean_val_b = np.mean(dom_b)

    global_col[:,:,0] = np.ones(image.shape[0:2]) * mean_val_r
    global_col[:,:,1] = np.ones(image.shape[0:2]) * mean_val_g
    global_col[:,:,2] = np.ones(image.shape[0:2]) * mean_val_b
    if is_0_255:
        global_col = global_col.astype(int)
    return global_col 
    
    
    
    
    
    
    
    
