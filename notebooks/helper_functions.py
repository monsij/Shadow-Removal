"""Helper functions for the document shadow.

Defines helper functions that are used in the document shadow
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output


def plot_img(img,title):
    """Show the image with appropriate changes"""
    
    f,ax = plt.subplots(1,1, figsize=(5,5))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    f.tight_layout()
    plt.show()

    return f, ax

    


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

def ratio_local_bg(ip_image, p, block_size, is_0_255):
    d = block_size//2
    m = ip_image.shape[0]
    n = ip_image.shape[1]
    
    I_local = np.zeros((m,n,3))
    
    for channel in range(3): #loop for each color channel
        print("Evaluating for color channel:",channel+1)
        for row in tqdm(range(m)):
            for col in range(n):
                block_intensities = ip_image[max(row-d,0):min(row+d+1,m-1),max(col-d,0):min(col+d+1,n-1),channel].flatten()
                I_local[row][col][channel] = np.percentile(block_intensities,100*p)
        clear_output(wait=True)
    if is_0_255:
        I_local = I_local.astype(int)
    return I_local


def ratio_local_bg_refined(I_local, ip_img, threshold, median_block_size, is_0_255):
    median_d = median_block_size//2
    t = threshold
    I_local_refined = np.zeros(I_local.shape)
    
    m = ip_img.shape[0]
    n = ip_img.shape[1]
    
    for channel in range(3):
        print("Evaluating for color channel:",channel+1)
        for row in tqdm(range(m)):
            for col in range(n):
                if I_local[row][col][channel] <= (1+t)*ip_img[row][col][channel] and (1-t)*ip_img[row][col][channel] <= I_local[row][col][channel]:
                    I_local_refined[row][col][channel] = ip_img[row][col][channel]
                else:
                    I_local_refined[row][col][channel] = np.median(I_local[max(row-median_d,0):min(row+median_d+1,m-1),max(col-median_d,0):min(col+median_d+1,n-1),channel].flatten())
        clear_output(wait=True)
    if is_0_255:
        I_local_refined = I_local_refined.astype(int)
    return I_local_refined
    

    
    
    
