"""Helper functions for the document shadow.

Defines helper functions that are used in the document shadow
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
from PIL import Image
from skimage import color,filters,transform,io



##### General functions #####



def load_img(path):
    img = Image.open(path)
    ip_img = np.array(img)
    return ip_img


def plot_img(img,title,r_lim=None, c_lim=None):
    """Show the image with appropriate changes"""
    
    f,ax = plt.subplots(1,1, figsize=(5,5))
    if r_lim is not None and c_lim is not None:
        ax.imshow(img[:r_lim,:c_lim,:])
    elif r_lim is not None and c_lim is None:
        ax.imshow(img[:r_lim,:,:])
    elif r_lim is None and c_lim is not None:
        ax.imshow(img[:,:c_lim,:])
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    f.tight_layout()
    plt.show()

    return f, ax


def show_img_compare(img_1, img_2, title1, title2, r_lim=None, c_lim=None,img_2_binary=False):
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
    if r_lim==None:
        ax[0].imshow(img_1)
    else:
        ax[0].imshow(img_1[:r_lim,:c_lim])
    ax[0].set_title(title1)
    ax[0].axis('off')
    
    if r_lim==None:
        if img_2_binary:
            ax[1].imshow(img_2, cmap='gray')
        else:
            ax[1].imshow(img_2)
    else:
        if img_2_binary:
            ax[1].imshow(img_2[:r_lim,:c_lim], cmap='gray')
        else:
            ax[1].imshow(img_2[:r_lim,:c_lim])
    ax[1].set_title(title2)
    ax[1].axis('off')

    f.tight_layout()
    plt.show()

    return f, ax



##### Functions for approach-1 #####



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


def get_local_bg(ip_image, p, block_size, is_0_255):
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


def get_local_bg_refined(I_local, ip_img, threshold, median_block_size, is_0_255):
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


def generate_deshadow(ip_img, I_local, I_global, is_0_255):
    shadow_map = I_local / I_global #mostly decimals < 1
    
    # Preventing division by zero (see next step)
    zero_loc = np.where(shadow_map[:,:,:]==0)
    shadow_map[zero_loc] = 1  # change maybe
    I_deshadow = ip_img / shadow_map
    
    if is_0_255:
        I_deshadow = I_deshadow.astype(int).clip(0,255) #change maybe for [0-1]
    return I_deshadow



##### Function for approach-3 #####



def estimate_shading_reflectance(ip_img, binary_img, window_size):
    """Works for image scaled to 0-255"""
    d = window_size//2
    shadow_map = np.zeros(ip_img.shape)
    m = ip_img.shape[0]
    n = ip_img.shape[1]
    
    for channel in range(3):
        print("Evaluating for color channel:",channel+1)
        for row in tqdm(range(m)):
            for col in range(n):
                if binary_img[row, col] == 1:
                    shadow_map[row, col, channel] = ip_img[row, col, channel]
                else:
                    window = ip_img[max(row-d,0):min(row+d+1,m-1), max(col-d,0):min(col+d+1,n-1), channel]
                    shadow_map[row][col][channel] = np.mean(window)
        clear_output(wait=True)
    shadow_map = shadow_map.astype(np.uint8)
    
    op_img = ip_img / shadow_map
    op_img = op_img.clip(0,1)
    op_img = np.round(op_img*255.0).astype(np.uint8)

    
    op_img_gray = color.rgb2gray(op_img)
    threshold_mask = filters.threshold_local(op_img_gray, block_size=501)
    op_binary_image = op_img_gray > threshold_mask
    
    return op_img, op_binary_image
