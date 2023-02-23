import math
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from matplotlib import cm


##################### TASK 1 ###################

# 1.1 IMPLEMENT
def make_gaussian_kernel(ksize, sigma):
    '''
    Implement the simplified Gaussian kernel below:
    k(x,y)=exp(((x-x_mean)^2+(y-y_mean)^2)/(-2sigma^2))
    Make Gaussian kernel be central symmentry by moving the 
    origin point of the coordinate system from the top-left
    to the center. Please round down the mean value. In this assignment,
    we define the center point (cp) of even-size kernel to be the same as that of the nearest
    (larger) odd size kernel, e.g., cp(4) to be same with cp(5).
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''
    
    # YOUR CODE HERE
    kernel = np.zeros((ksize, ksize))

    for i in range(ksize):
        for j in range(ksize):
            kernel[i][j] = math.exp(((i-ksize//2)**2+(j-ksize//2)**2)/(-2*sigma**2))
    kernel = kernel/np.sum(kernel)

    return kernel



# GIVEN
def cs4243_filter(image, kernel):
    """
    Fast version of filtering algorithm.
    Pre-extract all the regions of kernel size,
    and obtain a matrix of shape (Hi*Wi, Hk*Wk), also reshape the flipped
    kernel to be of shape (Hk*Wk, 1), then do matrix multiplication, and reshape back
    to get the final output image. 
    :param image: numpy.ndarray
    :param kernel: numpy.ndarray
    :return filtered_image: numpy.ndarray
    """
    def cs4243_rotate180(kernel):
        kernel = np.flip(np.flip(kernel, 0),1)
        return kernel
    
    def img2col(input, h_out, w_out, h_k, w_k, stride):
        h, w = input.shape
        out = np.zeros((h_out*w_out, h_k*w_k))
        
        convwIdx = 0
        convhIdx = 0
        for k in range(h_out*w_out):
            if convwIdx + w_k > w:
                convwIdx = 0
                convhIdx += stride
            out[k] = input[convhIdx:convhIdx+h_k, convwIdx:convwIdx+w_k].flatten()
            convwIdx += stride
        return out
    
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    if Hk % 2 == 0 or Wk % 2 == 0:
        raise ValueError
        
    hkmid = Hk//2
    wkmid = Wk//2

    image = cv2.copyMakeBorder(image, hkmid, hkmid, wkmid, wkmid, cv2.BORDER_REFLECT)
    filtered_image = np.zeros((Hi, Wi))
    kernel = cs4243_rotate180(kernel)
    col = img2col(image, Hi, Wi, Hk, Wk, 1)
    kernel_flatten = kernel.reshape(Hk*Wk, 1)
    output = col @ kernel_flatten 
    filtered_image = output.reshape(Hi, Wi)
    
    return filtered_image

# GIVEN
def cs4243_blur(img, gaussian_kernel, display=True):
    '''
    Performing Gaussian blurring on an image using a Gaussian kernel.
    :param img: input image
    :param gaussian_kernel: gaussian kernel
    :return blurred_img: blurred image
    '''

    blurred_img = cs4243_filter(img, gaussian_kernel)

    if display:

        fig1, axes_array = plt.subplots(1, 2)
        fig1.set_size_inches(8,4)
        image_plot = axes_array[0].imshow(img,cmap=plt.cm.gray) 
        axes_array[0].axis('off')
        axes_array[0].set(title='Original Image')
        image_plot = axes_array[1].imshow(blurred_img,cmap=plt.cm.gray)
        axes_array[1].axis('off')
        axes_array[1].set(title='Filtered Image')
        plt.show()  
    return blurred_img

# 2 IMPLEMENT
def estimate_gradients(original_img, display=True):
    '''
    Compute gradient orientation and magnitude for the input image.
    Perform the following steps:
    
    1. Compute dx and dy, responses to the horizontal and vertical Sobel kernel. Make use of the cs4243_filter function.
    
    2. Compute the gradient magnitude which is equal to sqrt(dx^2 + dy^2) 
    
    3. Compute the gradient orientation using the following formula:
        gradient = atan2(dy/dx)
        
    You may want to divide the original image pixel value by 255 to prevent overflow.
    
    Note that our axis choice is as follows:
            --> y
            |   
            ↓ x    
    Where img[x,y] denotes point on the image at coordinate (x,y)
    
    :param original_img: original grayscale image
    :return d_mag: gradient magnitudes matrix
    :return d_angle: gradient orientation matrix (in radian)
    '''
    
    dx = None
    dy = None
    d_mag = None
    d_angle = None
    
    # YOUR CODE HERE
    Kx = np.array([[ 1,  2,  1],
          [ 0,  0,  0],
          [-1, -2, -1]])
    
    Ky = np.array([[ 1,  0, -1],
          [ 2,  0, -2],
          [ 1,  0, -1]])
    
    dx = cs4243_filter(original_img/255, Kx)
    dy = cs4243_filter(original_img/255, Ky)

    d_mag = np.sqrt(dx**2 + dy**2)
    d_angle = np.arctan2(dy, dx)
    
    if display:
    
        fig2, axes_array = plt.subplots(1, 4)
        fig2.set_size_inches(16,4)
        image_plot = axes_array[0].imshow(d_mag, cmap='gray')  
        axes_array[0].axis('off')
        axes_array[0].set(title='Gradient Magnitude')

        image_plot = axes_array[1].imshow(dx, cmap='gray')  
        axes_array[1].axis('off')
        axes_array[1].set(title='dX')

        image_plot = axes_array[2].imshow(dy, cmap='gray')  
        axes_array[2].axis('off')
        axes_array[2].set(title='dY')

        image_plot = axes_array[3].imshow(d_angle, cmap='gray')  
        axes_array[3].axis('off')
        axes_array[3].set(title='Gradient Direction')
        plt.show()
    
    return d_mag, d_angle

# 3a IMPLEMENT
def non_maximum_suppression_interpol(d_mag, d_angle, display=True):
    '''
    Perform non-maximum suppression on the gradient magnitude matrix with interpolation.
    :param d_mag: gradient magnitudes matrix
    :param d_angle: gradient orientation matrix (in radian)
    :return out: non-maximum suppressed image
    '''

    out = np.zeros(d_mag.shape, d_mag.dtype)
    # Change angles to degrees to improve quality of life
    d_angle_180 = d_angle * 180/np.pi
    
    # YOUR CODE HERE
    H, W = d_mag.shape

    for i in range(1, H-1):
        for j in range(1, W-1):
            before = 0
            after = 0
            angle = d_angle_180[i, j]

            if angle >=0 and angle < 45 or angle >= -180 and angle < -135:
                tan = np.tan(angle*np.pi/180)
                before = tan*(d_mag[i-1, j-1] - d_mag[i-1, j]) + d_mag[i-1, j-1]
                after = tan*(d_mag[i+1, j+1] - d_mag[i+1, j]) + d_mag[i+1, j+1]
            elif angle >= 45 and angle < 90 or angle >= -135 and angle < -90:
                tan = np.tan((90-angle)*np.pi/180) if angle >= 45 else np.tan((90+angle)*np.pi/180)
                before = tan*(d_mag[i-1, j-1] - d_mag[i, j-1]) + d_mag[i-1, j-1]
                after = tan*(d_mag[i+1, j+1] - d_mag[i, j+1]) + d_mag[i+1, j+1]
            elif angle >= 90 and angle < 135 or angle >= -90 and angle < -45:
                tan = np.tan((angle-90)*np.pi/180) if angle >= 90 else np.tan((angle+90)*np.pi/180)
                before = tan*(d_mag[i-1, j+1] - d_mag[i, j+1]) + d_mag[i, j+1]
                after = tan*(d_mag[i+1, j-1] - d_mag[i, j-1]) + d_mag[i, j-1]
            elif angle >= 135 and angle < 180 or angle >= -45 and angle < 0:
                tan = np.tan((180-angle)*np.pi/180) if angle >= 135 else np.tan(angle*np.pi/180)
                before = tan*(d_mag[i-1, j+1] - d_mag[i-1, j]) + d_mag[i-1, j]
                after = tan*(d_mag[i+1, j-1] - d_mag[i+1, j]) + d_mag[i+1, j]
            
            if d_mag[i, j] > before and d_mag[i, j] > after:
                out[i, j] = d_mag[i, j]
    

    # END
    if display:
        _ = plt.figure(figsize=(10,10))
        plt.imshow(out, cmap='gray')
        plt.title("Suppressed image (with interpolation)")
    
    return out

# 3b IMPLEMENT
def non_maximum_suppression(d_mag, d_angle, display=True):
    '''
    Perform non-maximum suppression on the gradient magnitude matrix without interpolation.
    Split the range -180° ~ 180° into 8 even ranges. For each pixel, determine which range the gradient
    orientation belongs to and pick the corresponding two pixels from the adjacent eight pixels surrounding 
    that pixel. Keep the pixel if its value is larger than the other two.
    Do note that the coordinate system is as below and angular measurements are counter-clockwise.

    ----------→ y  
    |
    |
    |
    |        x X x
    ↓ x       \|/   
             x-o-x  
              /|\    
             x X x 
         -22.5 0 22.5
         
    For instance, 
        in the example above if the orientation at the coordinate of interest (x,y) is 20°, 
            it belongs to the -22.5°~22.5° range, 
            and the two pixels to be compared with are at (x+1,y) and (x-1,y) (aka the two big X's).
        If the angle was instead 40°,
            it belongs to the 22.5°-67.5° 
            and the two pixels we need to consider will be (x+1, y+1) and (x-1,y-1)

    There are only 4 sets of offsets: (0,1), (1,0), (1,1), and (1,-1), since to find the second pixel offset you just need 
    to multiply the first tuple by -1.
    
    :param d_mag: gradient magnitudes matrix
    :param d_angle: gradient orientation matrix (in radian)
    :return out: non-maximum suppressed image
    '''

    out = np.zeros(d_mag.shape, d_mag.dtype)
    # Change angles to degrees to improve quality of life
    d_angle_180 = d_angle * 180/np.pi
 
    # YOUR CODE HERE
    H, W = d_mag.shape

    for i in range(1, H-1):
        for j in range(1, W-1):
            angle = d_angle_180[i, j]
            before = 0
            after = 0
            
            if angle >= -22.5 and angle < 22.5 or angle >= 157.5 and angle < 180 or angle >= -180 and angle < -157.5:
                before = d_mag[i+1, j]
                after = d_mag[i-1, j]
            elif angle >= 22.5 and angle < 67.5 or angle >= -157.5 and angle < -112.5:
                before = d_mag[i+1, j-1]
                after = d_mag[i-1, j-1]
            elif angle >= 67.5 and angle < 112.5 or angle >= -112.5 and angle < -67.5:
                before = d_mag[i, j-1]
                after = d_mag[i, j+1]
            else:
                before = d_mag[i+1, j-1]
                after = d_mag[i-1, j+1]
            if d_mag[i, j] > before and d_mag[i, j] > after:
                out[i, j] = d_mag[i, j]
            

    # END
    if display:
        _ = plt.figure(figsize=(10,10))
        plt.imshow(out)
        plt.title("Suppressed image (without interpolation)")
    
    return out



# 4 IMPLEMENT
def double_thresholding(inp, perc_weak=0.1, perc_strong=0.3, display=True):
    '''
    Perform double thresholding. Use on the output of NMS. The high and low thresholds are computed as follow:
    
    range = max_val - min_val
    high_threshold = min_val + perc_strong * range 
    low_threshold = min_val + perc_weak * range
    
    perc_weak being 0 is possible
    Do note that the return edge images should be binary (0-1 or True-False)
    :param inp: numpy.ndarray
    :param perc_weak: value to determine low threshold
    :param perc_strong: value to determine high threshold
    :return weak_edges, strong_edges: binary edge images
    '''
    weak_edges = strong_edges = None
    
    # YOUR CODE HERE
    min_val = np.min(inp)
    max_val = np.max(inp)

    r = max_val - min_val

    high_threshold = min_val + perc_strong * r
    low_threshold = min_val + perc_weak * r

    strong_edges = inp > high_threshold
    weak_edges = np.zeros(inp.shape, dtype=bool)

    for i in range(inp.shape[0]):
        for j in range(inp.shape[1]):
            if inp[i, j] > low_threshold and inp[i, j] < high_threshold:
                weak_edges[i, j] = inp[i, j]

    # END
    
    if display:

        fig2, axes_array = plt.subplots(1, 2)
        fig2.set_size_inches(10,5)
        image_plot = axes_array[0].imshow(strong_edges, cmap='gray')  
        axes_array[0].axis('off')
        axes_array[0].set(title='Strong ')

        image_plot = axes_array[1].imshow(weak_edges, cmap='gray')  
        axes_array[1].axis('off')
        axes_array[1].set(title='Weak')
        
    return weak_edges, strong_edges

# 5 IMPLEMENT
def edge_linking(weak, strong, n=200, display=True):
    '''
    Perform edge-linking on two binary weak and strong edge images. 
    A weak edge pixel is linked if any of its eight surrounding pixels is a strong edge pixel.
    You may want to avoid using loops directly due to the high computational cost. One possible trick is to generate
    8 2D arrays from the strong edge image by offseting and sum them together; entries larger than 0 mean that at least one surrounding
    pixel is a strong edge pixel (otherwise the sum would be 0).
    
    You may also want to limit the number of iterations (test with 10-20 iterations first to check your implementation speed), and use a stopping condition (stop if no more pixel is added to the strong edge image).
    Also, when a weak edge pixel is added to the strong set, remember to remove it.


    :param weak: weak edge image (binary)
    :param strong: strong edge image (binary)
    :param n: maximum number of iterations
    :return out: final edge image
    '''
    assert weak.shape == strong.shape, \
        "Weak and strong edge image have to have the same dimension"
    out = None
    
    # YOUR CODE HERE
    out = strong.copy()
    H = out.shape[0]
    W = out.shape[1]

    x_weak, y_weak = np.where(weak == 1)


    for i in range(n):
        for j in range(len(x_weak)):
            x = x_weak[j]
            y = y_weak[j]
            if x > 0 and x < H-1 and y > 0 and y < W-1:
                if strong[x-1, y-1] == 1 or strong[x-1, y] == 1 or strong[x-1, y+1] == 1 or strong[x, y-1] == 1 or strong[x, y+1] == 1 or strong[x+1, y-1] == 1 or strong[x+1, y] == 1 or strong[x+1, y+1] == 1:
                    out[x, y] = 1
                    weak[x, y] = 1
        strong = out.copy()

    
    # END
    if display:
        _ = plt.figure(figsize=(10,10))
        plt.imshow(out)
        plt.title("Edge image")
    return out

##################### TASK 2 ######################

# 1/2/3 IMPLEMENT
def hough_vote_lines(img):
    '''
    Use the edge image to vote for 2 parameters: distance and theta
    Beware of our coordinate convention.
    :param img: edge image
    :return A: accumulator array
    :return distances: distance values array
    :return thetas: theta values array
    '''
    # YOUR CODE HERE
    H = img.shape[0]
    W = img.shape[1]

    d_max = np.sqrt(H**2 + W**2)
    d_min = -d_max
    d_step = 1

    t_min = 0
    t_max = np.pi
    t_step = np.pi/180

    distances = np.arange(d_min, d_max, d_step)
    thetas = np.arange(t_min, t_max, t_step)

    dist_lines = math.ceil((d_max - d_min)/d_step)
    theta_lines = math.ceil((t_max - t_min)/t_step)

    A = np.zeros((dist_lines, theta_lines), dtype=np.int)

    for i in range(H):
        for j in range(W):
            if img[i, j] > 0:
                for k in range(len(thetas)):
                    theta = thetas[k]
                    d = i * np.cos(theta) + j * np.sin(theta)
                    d_idx = np.argmin(np.abs(distances - d))
                    A[d_idx, k] += 1
    # END
            
    return A, distances, thetas

# 4 GIVEN
from skimage.feature import peak_local_max
def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters).

    Also include the array of values corresponding to the bins, in descending order.
    
    e.g.
    Suppose for a line detection case, you get the following output:
    [
    [122, 101, 93],
    [3,   40,  21],
    [0,   1.603, 1.605]
    ]
    This means that the local maxima with the highest vote gets a vote score of 122, and the corresponding parameter value is distance=3, 
    theta = 0.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]} {i}"
    peaks_indices = peak_local_max(hspace.copy(), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])] for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    print(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
        
    return res


##################### TASK 3 ######################

# 1/2/3 IMPLEMENT
from skimage.draw import circle_perimeter
def hough_vote_circles(img, radius = None):
    '''
    Use the edge image to vote for 3 parameters: circle radius and circle center coordinates.
    We also accept a range of radii to save computation costs. If the radius range is not given, it is default to
    [3, diagonal of the circle]. This parameter is very useful.
    
    Hint: You can use the function circle_perimeter to make a circular mask. Center the mask over the accumulator array and increment the array. In this case, you will have to pad the accumulator array first, and clip it afterwards. Remember that the return accumulator array should have matching dimension with the lengths of the parameter ranges. 
    
    :param img: edge image
    :param radius: min radius, max radius
    :return A: accumulator array
    :return R: radius values array
    :return X: x-coordinate values array
    :return Y: y-coordinate values array
    '''
    
    
    # Check the radius range
    h, w = img.shape[:2]    
    if radius == None:
        R_max = np.hypot(h,w)
        R_min = 3
    else:
        [R_min,R_max] = radius

    R = np.arange(R_min, R_max + 1/2)
    X = np.arange(0, h  + 1/2, 1)
    Y = np.arange(0, w  + 1/2, 1)

    A = np.zeros((len(R), len(X), len(Y)))

   
    #2. Extracting all edge coordinates
    edges = []

    for r in range(0, h):
        for c in range(0, w):
            if img[r][c] > 0:
                edges.append((r, c))

    #3. For each radius:
    for rad_idx in range(0, len(R)):
        rad = R[rad_idx]
        
        #3.1 Creating a circular mask
        rr, cc = circle_perimeter(0, 0, int(rad))
        
        #3.2 Compute the number of non_zero values on the mask
        #3.3 For each edge point:
        #    Center the mask over that point and update the accumulator array

        for (r, c) in edges:
            current_rr = rr + r
            current_cc = cc + c
            for p in range(0, len(rr)):
                try:
                    # code change for coin problem
                    # if rad < 20:
                    #     A[rad_idx][current_rr[p]][current_cc[p]] += 0.5

                    A[rad_idx][current_rr[p]][current_cc[p]] += 1
                except:
                    pass

    # END
   
    return A, R, X, Y


##################### TASK 4 ######################

# IMPLEMENT
def hough_vote_circles_grad(img, d_angle, radius = None):
    '''
    Use the edge image to vote for 3 parameters: circle radius and circle center coordinates.
    We also accept a range of radii to save computation costs. If the radius range is not given, it is default to
    [3, diagonal of the circle].
    This time, gradient information is used to avoid casting too many unnecessary votes.
    
    Remember that for a given pixel, you need to cast two votes along the orientation line. One in the positive direction, the other in
    negative direction.
    
    :param img: edge image
    :param d_angle: corresponding gradient orientation matrix
    :param radius: min radius, max radius
    :return A: accumulator array
    :return R: radius values array
    :return X: x-coordinate values array
    :return Y: y-coordinate values array
    '''
    # Check the radius range
    h, w = img.shape[:2]    
    if radius == None:
        R_max = np.hypot(h,w)
        R_min = 3
    else:
        [R_min,R_max] = radius
    
    R = np.arange(0, R_max + 1/2)
    X = np.arange(0, h + 1/2)
    Y = np.arange(0, w + 1/2)

    num_r = R.shape[0]
    num_x = X.shape[0]
    num_y = Y.shape[0]

    A = np.zeros((num_r, num_x, num_y))

    for r in range(0, h):
        for c in range(0, w):
            if img[r][c] > 0:
                theta = d_angle[r][c]
                for rad in range(R_min, R_max):
                    x_offset = np.cos(theta) * rad
                    y_offset = np.sin(theta) * rad
                    (r1, c1) = (int(r - x_offset), int(c - y_offset))
                    (r2, c2) = (int(r + x_offset), int(c + y_offset))
                    try:
                        A[rad][r1][c1] += 1
                    except:
                        pass
                    try:
                        A[rad][r2][c2] += 1
                    except:
                        pass
    return A, R, X, Y



###############################################
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

def draw_lines(hspace, dists, thetas, hs_maxima, file_path):
    im_c = read_img(file_path)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(im_c, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(thetas).mean()
    d_step = 0.5 * np.diff(dists).mean()
    bounds = [np.rad2deg(thetas[0] - angle_step),
              np.rad2deg(thetas[-1] + angle_step),
              dists[-1] + d_step, dists[0] - d_step]

    ax[1].imshow(np.log(1 + hspace), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(im_c, cmap=cm.gray)
    ax[2].set_ylim((im_c.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    # You may want to change the codes below if you use a different axis choice.
    for _, dist, angle in zip(*hs_maxima):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((y0, x0), slope=np.tan(np.pi-angle))

    plt.tight_layout()
    plt.show()

def draw_circles(local_maxima, file_path, title):
    # If this function does not work, use the other version (v2) below.
    img = cv2.imread(file_path)
    fig = plt.figure(figsize=(7,7))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    circle = []
    for _,r,x,y in zip(*local_maxima):
        circle.append(plt.Circle((y,x),r,color=(1,0,0),fill=False))
        fig.add_subplot(111).add_artist(circle[-1])
    plt.title(title)    
    plt.show()

from matplotlib.patches import Circle
def draw_circles_v2(local_maxima, file_path, title):
    img = cv2.imread(file_path)
    plt.rcParams["figure.figsize"] = [7.0, 7.0]
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    circle = []
    for _,r,x,y in zip(*local_maxima):
         ax.add_patch(Circle((y, x), r, color='red', fill=False))
    plt.title(title)    
    plt.show()