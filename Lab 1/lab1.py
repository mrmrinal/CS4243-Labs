""" CS4243 Lab 1: Template Matching
"""

import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import math

##### Part 1: Image Preprossessing #####

def rgb2gray(img):
    """
    5 points
    Convert a colour image greyscale
    Use (R,G,B)=(0.299, 0.587, 0.114) as the weights for red, green and blue channels respectively
    :param img: numpy.ndarray (dtype: np.uint8)
    :return img_gray: numpy.ndarray (dtype:np.uint8)
    """
    if len(img.shape) != 3:
        print('RGB Image should have 3 channels')
        return
    
    """ Your code starts here """
    height = img.shape[0]
    width = img.shape[1]

    img_gray = np.zeros((height, width), dtype = np.uint8)
    for i in range(height):
        for j in range(width):
            img_gray[i, j] = int(img[i, j, 0] * 0.299) + int(img[i, j, 1] * 0.587) + int(img[i, j, 2] * 0.114)
    """ Your code ends here """
    return img_gray

def flip_filter(filter):
    height = filter.shape[0]
    width = filter.shape[1]

    filter_flipped = np.zeros((height, width), dtype = np.float)
    for i in range(height):
        for j in range(width):
            filter_flipped[i, j] = filter[height - i - 1, width - j - 1]
    return filter_flipped

def cross_correlation(img, filter):
    height, width = img.shape[:2]
    filter_height, filter_width = filter.shape[:2]
    img_filtered = np.zeros((height, width), dtype = np.float)

    img = pad_zeros(img, filter_height // 2, filter_height // 2, filter_width // 2, filter_width // 2)

    for i in range(0, height):
        for j in range(0, width):
            for k in range(0, filter_height):
                for l in range(0, filter_width):
                    img_filtered[i, j] += img[i + k, j + l] * filter[k, l]
    return img_filtered

def gray2grad(img):
    """
    5 points
    Estimate the gradient map from the grayscale images by convolving with Sobel filters (horizontal and vertical gradients) and Sobel-like filters (gradients oriented at 45 and 135 degrees)
    The coefficients of Sobel filters are provided in the code below.
    :param img: numpy.ndarray
    :return img_grad_h: horizontal gradient map. numpy.ndarray
    :return img_grad_v: vertical gradient map. numpy.ndarray
    :return img_grad_d1: diagonal gradient map 1. numpy.ndarray
    :return img_grad_d2: diagonal gradient map 2. numpy.ndarray
    """
    sobelh = np.array([[-1, 0, 1], 
                       [-2, 0, 2], 
                       [-1, 0, 1]], dtype = float)
    sobelv = np.array([[-1, -2, -1], 
                       [0, 0, 0], 
                       [1, 2, 1]], dtype = float)
    sobeld1 = np.array([[-2, -1, 0],
                        [-1, 0, 1],
                        [0,  1, 2]], dtype = float)
    sobeld2 = np.array([[0, -1, -2],
                        [1, 0, -1],
                        [2, 1, 0]], dtype = float)
    

    """ Your code starts here """
    img_grad_h = cross_correlation(img, flip_filter(sobelh))
    img_grad_v = cross_correlation(img, flip_filter(sobelv))
    img_grad_d1 = cross_correlation(img, flip_filter(sobeld1))
    img_grad_d2 = cross_correlation(img, flip_filter(sobeld2))

    """ Your code ends here """
    return img_grad_h, img_grad_v, img_grad_d1, img_grad_d2

def pad_zeros(img, pad_height_bef, pad_height_aft, pad_width_bef, pad_width_aft):
    """
    5 points
    Add a border of zeros around the input images so that the output size will match the input size after a convolution or cross-correlation operation.
    e.g., given matrix [[1]] with pad_height_bef=1, pad_height_aft=2, pad_width_bef=3 and pad_width_aft=4, obtains:
    [[0 0 0 0 0 0 0 0]
    [0 0 0 1 0 0 0 0]
    [0 0 0 0 0 0 0 0]
    [0 0 0 0 0 0 0 0]]
    :param img: numpy.ndarray
    :param pad_height_bef: int
    :param pad_height_aft: int
    :param pad_width_bef: int
    :param pad_width_aft: int
    :return img_pad: numpy.ndarray. dtype is the same as the input img. 
    """
    height, width = img.shape[:2]
    new_height, new_width = (height + pad_height_bef + pad_height_aft), (width + pad_width_bef + pad_width_aft)
    img_pad = np.zeros((new_height, new_width)) if len(img.shape) == 2 else np.zeros((new_height, new_width, img.shape[2]))

    """ Your code starts here """
    for i in range(0, height):
        for j in range(0, width):
            img_pad[i + pad_height_bef, j + pad_width_bef] = img[i, j]
    """ Your code ends here """
    return img_pad.astype(img.dtype)



##### Part 2: Normalized Cross Correlation #####
def norm_single(grid):
    res = 0
    h, w = grid.shape[:2]
    for i in range(h):
        for j in range(w):
            res += (grid[i, j])**2
    return math.sqrt(res)

def norm_rgb(grid):
    res_r, res_g, res_b = 0, 0, 0
    h, w = grid.shape[:2]
    for i in range(h):
        for j in range(w):
            res_r += (grid[i, j, 0])**2
            res_g += (grid[i, j, 1])**2
            res_b += (grid[i, j, 2])**2
    return math.sqrt(res_r + res_g + res_b)

def normalized_cross_correlation(img, template):
    """
    10 points.
    Implement the cross-correlation operation in a naive 6 nested for-loops. 
    The 6 loops include the height, width, channel of the output and height, width and channel of the template.
    :param img: numpy.ndarray.
    :param template: numpy.ndarray.
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    img = img.astype('float64')
    template = template.astype('float64')

    # Single Channel
    response = np.zeros((Ho, Wo))
    if len(img.shape) != 3:
        filter_norm = norm_single(template)
        for i in range(Ho):
            for j in range(Wo):
                image_norm = norm_single(img[i: i + Hk, j: j + Wk])
                for k in range(Hk):
                    for l in range(Wk):
                        response[i, j] += (img[i + k, j + l] * template[k, l])
                response[i, j] /= (image_norm * filter_norm)
    
    # RGB
    else:
        filter_norm = norm_rgb(template)
        for i in range(Ho):
            for j in range(Wo):
                image_norm = norm_rgb(img[i: i + Hk, j: j + Wk:])
                for m in range(3):
                    for k in range(Hk):
                        for l in range(Wk):
                            response[i, j] += img[i + k, j + l, m] * template[k, l, m]
                response[i, j] /= (image_norm * filter_norm)
    """ Your code ends here """
    
    return response


def normalized_cross_correlation_fast(img, template):
    """
    10 points.
    Implement the cross correlation with 3 nested for-loops. 
    The for-loop over the template is replaced with the element-wise multiplication between the kernel and the image regions.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    img = img.astype('float64')
    template = template.astype('float64')

    # Single Channel
    response = np.zeros((Ho, Wo))

    if len(img.shape) != 3:
        filter_norm = norm_single(template)
        for i in range(Ho):
            for j in range(Wo):
                image_norm = norm_single(img[i: i + Hk, j: j + Wk])
                img_box, template_box = img[i: i + Hk, j: j + Wk], template
                res = np.multiply(img_box, template_box)
                response[i, j] += np.sum(res)
                response[i, j] /= (image_norm * filter_norm)
    
    # RGB
    else:
        filter_norm = norm_rgb(template)
        for i in range(Ho):
            for j in range(Wo):
                image_norm = norm_rgb(img[i: i + Hk, j: j + Wk:])
                for k in range(3):
                    img_box, template_box = img[i: i + Hk, j: j + Wk, k], template[:, :, k]
                    res = np.multiply(img_box, template_box)
                    res = np.sum(res)
                    response[i, j] += res
                response[i, j] /= (image_norm * filter_norm)

    """ Your code ends here """
    return response

def reshape_template(template):
    if len(template.shape) == 2:
        return np.column_stack([template.flatten()])
    else:
        h = template.shape[0]
        w = template.shape[1]
        c = template.shape[2]

        res = []
        for i in range(c):
            for j in range(h):
                for k in range(w):
                    res.append(template[j, k, i])
        return np.column_stack([np.array(res)])

def reshape_image(img, template):
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    res = []

    # greyscale
    if len(img.shape) != 3:
        for i in range(Ho):
            for j in range(Wo):
                res.append(img[i:i+Hk, j:j+Wk].flatten())
    # rgb
    else:
        temp = []
        c = img.shape[2]
        for i in range(Ho):
            for j in range(Wo):
                if temp != []:
                    res.append(temp)
                temp = []
                for k in range(c):
                    temp += img[i:i+Hk, j:j+Wk, k].flatten().tolist()
        res.append(temp)
    
    return np.array(res)


def normalized_cross_correlation_matrix(img, template):
    """
    10 points.
    Converts cross-correlation into a matrix multiplication operation to leverage optimized matrix operations.
    Please check the detailed instructions in the pdf file.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    response = np.zeros((Ho, Wo))

    #convert the matrix to the intended shape
    img = img.astype('float64')
    template = template.astype('float64')

    # Single Channel
    if len(img.shape) != 3:
        reshaped_template = reshape_template(template)
        reshaped_img = reshape_image(img, template)
        filter_norm = norm_single(template)
        matrix = np.matmul(reshaped_img, reshaped_template)

        normalization_term = np.sqrt(np.matmul(reshaped_img ** 2, np.ones((Hk * Wk, 1)))) * filter_norm
        response = matrix / normalization_term
        response = response.reshape(Ho, Wo)

        

    # RGB
    else:
        reshaped_template = reshape_template(template)
        reshaped_img = reshape_image(img, template)
        filter_norm = norm_rgb(template)
        matrix = np.matmul(reshaped_img, reshaped_template)
    
        normalization_term = np.sqrt(np.matmul(reshaped_img ** 2, np.ones((Hk * Wk * 3, 1)))) * filter_norm
        response = matrix / normalization_term
        response = response.reshape(Ho, Wo)

    """ Your code ends here """
    return response


##### Part 3: Non-maximum Suppression #####
def find_global_max(response):
    current_max = 0
    x_coord = -1
    y_coord = -1
    for i in range(len(response)):
        for j in range(len(response[0])):
            if response[i][j] > current_max:
                x_coord = i
                y_coord = j
                current_max = response[i][j]
    return current_max, x_coord, y_coord


def suppress_threshold(response, threshold):
    for i in range(len(response)):
        for j in range(len(response[0])):
            if response[i][j] < threshold:
                response[i][j] = 0
    return response


def window_suppression(response, suppress_range, x_coord, y_coord):
    H_range, W_range = suppress_range
    for i in range(x_coord - H_range, x_coord + H_range):
        for j in range(y_coord - W_range, y_coord + W_range):
            try:
                response[i][j] = 0
            except:
                continue
    return response


def non_max_suppression(response, suppress_range, threshold=None):
    """
    10 points
    Implement the non-maximum suppression for translation symmetry detection
    The general approach for non-maximum suppression is as follows:
	1. Set a threshold τ; values in X<τ will not be considered.  Set X<τ to 0.  
    2. While there are non-zero values in X
        a. Find the global maximum in X and record the coordinates as a local maximum.
        b. Set a small window of size w×w points centered on the found maximum to 0.
	3. Return all recorded coordinates as the local maximum.
    :param response: numpy.ndarray, output from the normalized cross correlation
    :param suppress_range: a tuple of two ints (H_range, W_range). 
                           the points around the local maximum point within this range are set as 0. In this case, there are 2*H_range*2*W_range points including the local maxima are set to 0
    :param threshold: int, points with value less than the threshold are set to 0
    :return res: a sparse response map which has the same shape as response
    """
    
    """ Your code starts here """
    # suppress
    response = suppress_threshold(response, threshold)
    local_minimums = []
    # find minimum 
    current_max, x_coord, y_coord = find_global_max(response)

    while current_max != 0:
        # make the area around the current_maximum set to 0
        response = window_suppression(response, suppress_range,x_coord, y_coord)
        local_minimums.append((x_coord,y_coord))
        current_max, x_coord, y_coord = find_global_max(response)

    """ Your code ends here """

    for i,j in local_minimums:
        response[i][j] = 1
    return response

##### Part 4: Question And Answer #####
    
def normalized_cross_correlation_ms(img, template):
    """
    10 points
    Please implement mean-subtracted cross correlation which corresponds to OpenCV TM_CCOEFF_NORMED.
    For simplicty, use the "fast" version.
    :param img: numpy.ndarray
    :param template: numpy.ndarray
    :return response: numpy.ndarray. dtype: float
    """
    Hi, Wi = img.shape[:2]
    Hk, Wk = template.shape[:2]
    Ho = Hi - Hk + 1
    Wo = Wi - Wk + 1

    """ Your code starts here """
    img = img.astype('float64')
    template = template.astype('float64')

    # Single Channel
    response = np.zeros((Ho, Wo))

    template = template - np.mean(template)
    
    if len(img.shape) != 3:
        filter_norm = norm_single(template)
        for i in range(Ho):
            for j in range(Wo):
                image_norm = norm_single(img[i: i + Hk, j: j + Wk] - np.mean(img[i: i + Hk, j: j + Wk]))
                for k in range(Hk):
                    for l in range(Wk):
                        response[i, j] += ((img[i + k, j + l] - np.mean(img[i: i + Hk, j: j + Wk])) * template[k, l])
                response[i, j] /= (image_norm * filter_norm)
    
    # RGB
    else:
        template = template - [np.mean(template[:,:,0]), np.mean(template[:,:,1]), np.mean(template[:,:,2])]
        filter_norm = norm_rgb(template)
        for i in range(Ho):
            for j in range(Wo):
                img_box = img[i: i + Hk, j: j + Wk:]
                image_norm = norm_rgb(img_box - [np.mean(img_box[:,:,0]), np.mean(img_box[:,:,1]), np.mean(img_box[:,:,2])])
                for m in range(3):
                    for k in range(Hk):
                        for l in range(Wk):
                            response[i, j] += (img[i + k, j + l, m] - np.mean(img_box[:,:,m])) * template[k, l, m]
                response[i, j] /= (image_norm * filter_norm)
    """ Your code ends here """
    return response




"""Helper functions: You should not have to touch the following functions.
"""
def read_img(filename):
    '''
    Read HxWxC image from the given filename
    :return img: numpy.ndarray, size (H, W, C) for RGB. The value is between [0, 255].
    '''
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def show_imgs(imgs, titles=None):
    '''
    Display a list of images in the notebook cell.
    :param imgs: a list of images or a single image
    '''
    if isinstance(imgs, list) and len(imgs) != 1:
        n = len(imgs)
        fig, axs = plt.subplots(1, n, figsize=(15,15))
        for i in range(n):
            axs[i].imshow(imgs[i], cmap='gray' if len(imgs[i].shape) == 2 else None)
            if titles is not None:
                axs[i].set_title(titles[i])
    else:
        img = imgs[0] if (isinstance(imgs, list) and len(imgs) == 1) else imgs
        plt.figure()
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)

def show_img_with_points(response, img_ori=None):
    '''
    Draw small red rectangles of size defined by rec_shape around the non-zero points in the image.
    Display the rectangles and the image with rectangles in the notebook cell.
    :param response: numpy.ndarray. The input response should be a very sparse image with most of points as 0.
                     The response map is from the non-maximum suppression.
    :param img_ori: numpy.ndarray. The original image where response is computed from
    :param rec_shape: a tuple of 2 ints. The size of the red rectangles.
    '''
    response = response.copy()
    if img_ori is not None:
        img_ori = img_ori.copy()

    xs, ys = response.nonzero()
    for x, y in zip(xs, ys):
        response = cv2.circle(response, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        if img_ori is not None:
            img_ori = cv2.circle(img_ori, (y, x), radius=0, color=(255, 0, 0), thickness=5)
        
    if img_ori is not None:
        show_imgs([response, img_ori])
    else:
        show_imgs(response)


