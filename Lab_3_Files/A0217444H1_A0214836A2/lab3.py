""" CS4243 Lab 3: Feature Matching and Applications
"""
import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_filter
from utils import pad, unpad
import math
import cv2
_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)

_COLOR_RED = (255, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_BLUE = (0, 0, 255)



##### Part 1: Keypoint Detection, Description, and Matching #####

def harris_corners(img, window_size=3, k=0.04):
    '''
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the functions filters.sobel_v filters.sobel_h & scipy.ndimage.filters.convolve, 
        which are already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    '''

    H, W= img.shape
    window = np.ones((window_size, window_size))
    response = np.zeros((H, W))

    """ Your code starts here """
    sobel_x = filters.sobel_v(img)
    sobel_y = filters.sobel_h(img)

    # calculate x e y derivatives
    Ix2 = sobel_x ** 2
    Iy2 = sobel_y ** 2

    Ixy = sobel_x * sobel_y

    # calculate product and second derivatives
    Sxx = convolve(Ix2, window)
    Syy = convolve(Iy2, window)
    Sxy = convolve(Ixy, window)

    det = (Sxx * Syy) - (Sxy ** 2)
    trace = Sxx + Syy

    # calculate response
    response = det - k * (trace ** 2)

    """ Your code ends here """ 
    
    return response

def naive_descriptor(patch):
    '''
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.

    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    '''
    feature = []
    
    """ Your code starts here """
    patch = (patch - np.mean(patch)) / (np.std(patch) + 0.0001)
    feature = patch.flatten()
    
    """ Your code ends here """

    return feature

# GIVEN
def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    '''
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (x, y) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    '''

    image.astype(np.float32)
    desc = []
    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[np.max([0,y-(patch_size//2)]):y+((patch_size+1)//2),
                      np.max([0,x-(patch_size//2)]):x+((patch_size+1)//2)]
      
        desc.append(desc_func(patch))
   
    return np.array(desc)

# GIVEN
def make_gaussian_kernel(ksize, sigma):
    '''
    Good old Gaussian kernel.
    :param ksize: int
    :param sigma: float
    :return kernel: numpy.ndarray of shape (ksize, ksize)
    '''

    ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
    yy, xx = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(yy) + np.square(xx)) / np.square(sigma))

    return kernel / kernel.sum()


def simple_sift(patch):
    '''
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    Use the gradient orientation to determine the bin, and the gradient magnitude * weight from
    the Gaussian kernel as vote weight.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    '''
    
    # You can change the parameter sigma, which has been default to 3
    weights = np.flipud(np.fliplr(make_gaussian_kernel(patch.shape[0],3)))
    
    histogram = np.zeros((4,4,8))
    
    """ Your code starts here """
    sobel_x = filters.sobel_v(patch)
    sobel_y = filters.sobel_h(patch)

    # calculate the magnitude and orientation
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    magnitude *= weights
    orientation = np.arctan2(sobel_y, sobel_x) * 180 / np.pi
    orientation[orientation < 0] += 360

    feature = np.zeros(128)

    for i in range(4):
        for j in range(4):
            orientation_window = orientation[i*4:(i+1)*4, j*4:(j+1)*4]
            magnitude_window = magnitude[i*4:(i+1)*4, j*4:(j+1)*4]
            hist, _ = np.histogram(orientation_window, bins=8, range=(0, 360), weights=magnitude_window)
            feature[i*32 + j*8:i*32 + j*8+8] = hist
    
    feature /= np.linalg.norm(feature)
    """ Your code ends here """

    return feature

def top_k_matches(desc1, desc2, k=2):
    '''
    Compute the Euclidean distance between each descriptor in desc1 versus all descriptors in desc2 (Hint: use cdist).
    For each descriptor Di in desc1, pick out k nearest descriptors from desc2, as well as the distances themselves.
    Example of an output of this function:
    
        [(0, [(18, 0.11414082134194799), (28, 0.139670625444803)]),
         (1, [(2, 0.14780585099287238), (9, 0.15420019834435536)]),
         (2, [(64, 0.12429203239414029), (267, 0.1395765079352806)]),
         ...<truncated>
    '''
    match_pairs = []
    
    """ Your code starts here """
    # calculate the Euclidean distance
    distance = cdist(desc1, desc2, 'euclidean')

    # find the k nearest descriptors
    for i in range(len(desc1)):
        # find the index of k nearest descriptors
        index = np.argsort(distance[i])[:k]
        # find the distance of k nearest descriptors
        dist = distance[i][index]
        match_pairs.append((i, list(zip(index, dist))))
    
    """ Your code ends here """

    return match_pairs

def ratio_test_match(desc1, desc2, match_threshold):
    '''
    Match two set of descriptors using the ratio test.
    Output should be a numpy array of shape (k,2), where k is the number of matches found. 
    In the following sample output:
        array([[  3,   0],
               [  5,  30],
               [ 11,   9],
               [ 18,   7],
               [ 24,   5],
               [ 30,  17],
               [ 32,  24],
               [ 46,  23], ... <truncated>
              )
              
        desc1[3] is matched with desc2[0], desc1[5] is matched with desc2[30], and so on.
    
    All other match functions will return in the same format as does this one.
    
    '''
    match_pairs = []
    top_2_matches = top_k_matches(desc1, desc2)
    
    """ Your code starts here """
    # find the match pairs
    for i in range(len(top_2_matches)):
        ratio = top_2_matches[i][1][0][1] / top_2_matches[i][1][1][1]
        if ratio < match_threshold:
            match_pairs.append((top_2_matches[i][0], top_2_matches[i][1][0][0]))

    """ Your code ends here """

    # Modify this line as you wish
    match_pairs = np.array(match_pairs)
    return match_pairs

# GIVEN
def compute_cv2_descriptor(im, method=cv2.SIFT_create()):
    '''
    Detects and computes keypoints using one of the implementations in OpenCV
    You can use:
        cv2.SIFT_create()

    Do note that the keypoints coordinate is (col, row)-(x,y) in OpenCV. We have changed it to (row,col)-(y,x) for you. (Consistent with out coordinate choice)
    '''
    kpts, descs = method.detectAndCompute(im, None)
    
    keypoints = np.array([(kp.pt[1],kp.pt[0]) for kp in kpts])
    angles = np.array([kp.angle for kp in kpts])
    sizes = np.array([kp.size for kp in kpts])
    
    return keypoints, descs, angles, sizes

##### Part 2: Image Stitching #####

# GIVEN
def transform_homography(src, h_matrix, getNormalized = True):
    '''
    Performs the perspective transformation of coordinates

    Args:
        src (np.ndarray): Coordinates of points to transform (N,2)
        h_matrix (np.ndarray): Homography matrix (3,3)

    Returns:
        transformed (np.ndarray): Transformed coordinates (N,2)

    '''
    transformed = None

    input_pts = np.insert(src, 2, values=1, axis=1)
    transformed = np.zeros_like(input_pts)
    transformed = h_matrix.dot(input_pts.transpose())
    if getNormalized:
        transformed = transformed[:-1]/transformed[-1]
    transformed = transformed.transpose().astype(np.float32)
    
    return transformed


def compute_homography(src, dst):
    '''
    Calculates the perspective transform from at least 4 points of
    corresponding points using the **Normalized** Direct Linear Transformation
    method.

    Args:
        src (np.ndarray): Coordinates of points in the first image (N,2)
        dst (np.ndarray): Corresponding coordinates of points in the second
                          image (N,2)

    Returns:
        h_matrix (np.ndarray): The required 3x3 transformation matrix H.

    Prohibited functions:
        cv2.findHomography(), cv2.getPerspectiveTransform(),
        np.linalg.solve(), np.linalg.lstsq()
    '''
    h_matrix = np.eye(3, dtype=np.float64)
  
    """ Your code starts here """

    # start with normaliszation

    average_src_x = np.mean([i[0] for i in src])
    average_src_y = np.mean([i[1] for i in src])
    average_dst_x = np.mean([i[0] for i in dst])
    average_dst_y = np.mean([i[1] for i in dst])


    stddev_src_x = np.std([i[0] for i in src])
    stddev_src_y = np.std([i[1] for i in src])
    stddev_dst_x = np.std([i[0] for i in dst])
    stddev_dst_y = np.std([i[1] for i in dst])

    # dlt

    T_src = [[(1/stddev_src_x), 0, (-average_src_x/stddev_src_x)],
             [0, (1/stddev_src_y), (-average_src_y/stddev_src_y)],
             [0, 0, 1]]
    
    T_dst = [[1/stddev_dst_x, 0, -average_dst_x/stddev_dst_x],
             [0, 1/stddev_dst_y, -average_dst_y/stddev_dst_y],
             [0, 0, 1]]
    

    src = np.insert(src, 2, 1, axis=1)
    dst = np.insert(dst, 2, 1, axis=1)

    normalised_src = np.matmul(T_src, np.matrix.transpose(src))
    normalised_dst = np.matmul(T_dst, np.matrix.transpose(dst))

    normalised_src = np.matrix.transpose(normalised_src)
    normalised_dst = np.matrix.transpose(normalised_dst)



    # for each transformation, calculate the Ai matrix

    a = np.zeros(shape=(2 * len(src), 9))
    counter = 0

    for i in range(len(src)):
        x_src, y_src, throwaway = normalised_src[i]
        x_dst, y_dst, throwaway = normalised_dst[i]


        a[counter] = [-x_src, -y_src, -1, 0, 0, 0, x_src * x_dst, y_src * x_dst, x_dst]
        a[counter + 1] = [0, 0, 0, -x_src, -y_src, -1, x_src *y_dst, y_src * y_dst, y_dst]
        counter = counter + 2
    
    u, s, vh = np.linalg.svd(a)

    homography = vh[-1].reshape((3,3))

    homography =  homography/homography[2,2]

    # denormalisation
    return np.matmul(np.matmul(np.linalg.inv(T_dst),homography), T_src)


def ransac_homography(keypoints1, keypoints2, matches, sampling_ratio=0.5, n_iters=500, delta=20):
    """
    Use RANSAC to find a robust affine transformation
        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers
    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        sampling_ratio: percentage of points selected at each iteration
        n_iters: the number of iterations RANSAC will run
        threshold: the threshold to find inliers
    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * sampling_ratio)

    matched1_unpad = keypoints1[matches[:,0]]
    matched2_unpad = keypoints2[matches[:,1]]

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    for i in range(n_iters):
        sample = np.random.choice(N, n_samples, replace=False)
        matched1 = matched1_unpad[sample]
        matched2 = matched2_unpad[sample]

        H = compute_homography(matched1, matched2)
        
        # project points from image 2 to image 1
        projected_points = transform_homography(matched1_unpad, H)

        # calculate distance between projected points and matched points
        distance = np.linalg.norm(projected_points - matched2_unpad, axis=1)

        inliers = np.where(distance < delta)[0]

        if len(inliers) > n_inliers:
            max_inliers = inliers
            n_inliers = len(inliers)
        
    H = compute_homography(matched1_unpad[max_inliers], matched2_unpad[max_inliers])

    ### END YOUR CODE
    return H, matches[max_inliers]

##### Part 3: Mirror Symmetry Detection #####

# GIVEN 
from skimage.feature import peak_local_max
def find_peak_params(hspace, params_list,  window_size=1, threshold=0.5):
    '''
    Given a Hough space and a list of parameters range, compute the local peaks
    aka bins whose count is larger max_bin * threshold. The local peaks are computed
    over a space of size (2*window_size+1)^(number of parameters)

    Also include the array of values corresponding to the bins, in descending order.
    '''
    assert len(hspace.shape) == len(params_list), \
        "The Hough space dimension does not match the number of parameters"
    for i in range(len(params_list)):
        assert hspace.shape[i] == len(params_list[i]), \
            f"Parameter length does not match size of the corresponding dimension:{len(params_list[i])} vs {hspace.shape[i]}"
    peaks_indices = peak_local_max(hspace.copy(), exclude_border=False, threshold_rel=threshold, min_distance=window_size)
    peak_values = np.array([hspace[tuple(peaks_indices[j])] for j in range(len(peaks_indices))])
    res = []
    res.append(peak_values)
    for i in range(len(params_list)):
        res.append(params_list[i][peaks_indices.T[i]])
    return res

# GIVEN
def angle_with_x_axis(pi, pj):  
    '''
    Compute the angle that the line connecting two points I and J make with the x-axis (mind our coordinate convention)
    Do note that the line direction is from point I to point J.
    '''
    # get the difference between point p1 and p2
    y, x = pi[0]-pj[0], pi[1]-pj[1] 
    
    if x == 0:
        return np.pi/2  
    
    angle = np.arctan(y/x)
    if angle < 0:
        angle += np.pi
    return angle

# GIVEN
def midpoint(pi, pj):
    '''
    Get y and x coordinates of the midpoint of I and J
    '''
    return (pi[0]+pj[0])/2, (pi[1]+pj[1])/2

# GIVEN
def distance(pi, pj):
    '''
    Compute the Euclidean distance between two points I and J.
    '''
    y,x = pi[0]-pj[0], pi[1]-pj[1] 
    return np.sqrt(x**2+y**2)

def shift_sift_descriptor(desc):
    '''
       Generate a virtual mirror descriptor for a given descriptor.
       Note that you have to shift the bins within a mini histogram, and the mini histograms themselves.
       e.g:
       Descriptor for a keypoint
       (the dimension is (128,), but here we reshape it to (16,8). Each length-8 array is a mini histogram.)
      [[  0.,   0.,   0.,   5.,  41.,   0.,   0.,   0.],
       [ 22.,   2.,   1.,  24., 167.,   0.,   0.,   1.],
       [167.,   3.,   1.,   4.,  29.,   0.,   0.,  12.],
       [ 50.,   0.,   0.,   0.,   0.,   0.,   0.,   4.],
       
       [  0.,   0.,   0.,   4.,  67.,   0.,   0.,   0.],
       [ 35.,   2.,   0.,  25., 167.,   1.,   0.,   1.],
       [167.,   4.,   0.,   4.,  32.,   0.,   0.,   5.],
       [ 65.,   0.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  74.,   1.,   0.,   0.],
       [ 36.,   2.,   0.,   5., 167.,   7.,   0.,   4.],
       [167.,  10.,   0.,   1.,  30.,   1.,   0.,  13.],
       [ 60.,   2.,   0.,   0.,   0.,   0.,   0.,   1.],
       
       [  0.,   0.,   0.,   0.,  54.,   3.,   0.,   0.],
       [ 23.,   6.,   0.,   4., 167.,   9.,   0.,   0.],
       [167.,  40.,   0.,   2.,  30.,   1.,   0.,   0.],
       [ 51.,   8.,   0.,   0.,   0.,   0.,   0.,   0.]]
     ======================================================
       Descriptor for the same keypoint, flipped over the vertical axis
      [[  0.,   0.,   0.,   3.,  54.,   0.,   0.,   0.],
       [ 23.,   0.,   0.,   9., 167.,   4.,   0.,   6.],
       [167.,   0.,   0.,   1.,  30.,   2.,   0.,  40.],
       [ 51.,   0.,   0.,   0.,   0.,   0.,   0.,   8.],
       
       [  0.,   0.,   0.,   1.,  74.,   0.,   0.,   0.],
       [ 36.,   4.,   0.,   7., 167.,   5.,   0.,   2.],
       [167.,  13.,   0.,   1.,  30.,   1.,   0.,  10.],
       [ 60.,   1.,   0.,   0.,   0.,   0.,   0.,   2.],
       
       [  0.,   0.,   0.,   0.,  67.,   4.,   0.,   0.],
       [ 35.,   1.,   0.,   1., 167.,  25.,   0.,   2.],
       [167.,   5.,   0.,   0.,  32.,   4.,   0.,   4.],
       [ 65.,   1.,   0.,   0.,   0.,   0.,   0.,   0.],
       
       [  0.,   0.,   0.,   0.,  41.,   5.,   0.,   0.],
       [ 22.,   1.,   0.,   0., 167.,  24.,   1.,   2.],
       [167.,  12.,   0.,   0.,  29.,   4.,   1.,   3.],
       [ 50.,   4.,   0.,   0.,   0.,   0.,   0.,   0.]]
    '''
    
    """ Your code starts here """
    res = np.reshape(desc, (16,8))

    # keep first element of each row the same
    # swap second and last element of each row
    # swap third and second last element of each row
    for i in range(16):
        res[i,1], res[i,7] = res[i,7], res[i,1]
        res[i,2], res[i,6] = res[i,6], res[i,2]
        res[i,3], res[i,5] = res[i,5], res[i,3]
    
    # swap first 4 rows with last 4 rows
    res[0:4], res[12:16] = res[12:16], res[0:4].copy()
    # swap second 4 rows with second last 4 rows
    res[4:8], res[8:12] = res[8:12], res[4:8].copy()

    res = np.reshape(res, (128,))
    """ Your code ends here """
    return res

def create_mirror_descriptors(img):
    '''
    Return the output for compute_cv2_descriptor (which you can find in utils.py)
    Also return the set of virtual mirror descriptors.
    Make sure the virtual descriptors correspond to the original set of descriptors.
    '''
    
    """ Your code starts here """
    kps, descs, sizes, angles = compute_cv2_descriptor(img)

    mir_descs = []
    for desc in descs:
        mir_descs.append(shift_sift_descriptor(desc))
    mir_descs = np.array(mir_descs)
    
    """ Your code ends here """
    
    return kps, descs, sizes, angles, mir_descs

def match_mirror_descriptors(descs, mirror_descs, threshold = 0.7):
    '''
    First use `top_k_matches` to find the nearest 3 matches for each keypoint. Then eliminate the mirror descriptor that comes 
    from the same keypoint. Perform ratio test on the two matches left. If no descriptor is eliminated, perform the ratio test 
    on the best 2. 
    '''
    three_matches = top_k_matches(descs, mirror_descs, k=3)

    match_result = []

    
    """ Your code starts here """
    # eliminate the mirror descriptor that comes from the same keypoint
    for keypoint, match in three_matches:
        if match[0][0] == keypoint:
            match.remove((keypoint, 0.0))
    
    three_matches = np.array(three_matches)

    # perform ratio test on the two matches left
    for kp, match in three_matches:
        if match[0][1] / match[1][1] < threshold:
            match_result.append((kp, match[0][0]))
        
    
    match_result = np.array(match_result)
    """ Your code ends here """
    
    return match_result


def find_symmetry_lines(matches, kps):
    '''
    For each pair of matched keypoints, use the keypoint coordinates to compute a candidate symmetry line.
    Assume the points associated with the original descriptor set to be I's, and the points associated with the mirror descriptor set to be
    J's.
    '''
    rhos = []
    thetas = []
    """ Your code starts here """
    for match in matches:
        kp1 = kps[match[0]]
        kp2 = kps[match[1]]
        theta = angle_with_x_axis(kp1,kp2)

        mx, my = midpoint(kp1, kp2)

        rho = my * np.cos(theta) + mx * np.sin(theta)
        rhos.append(rho)
        thetas.append(theta)
    
    """ Your code ends here """

    return rhos, thetas

def hough_vote_mirror(matches, kps, im_shape, window=5, threshold=0.5, num_lines=1):
    '''
    Hough Voting:
                 0<=thetas<= 2pi      , interval size = 1 degree
        -diagonal <= rhos <= diagonal , interval size = 1 pixel
    Feel free to vary the interval size.
    '''
    rhos, thetas = find_symmetry_lines(matches, kps)

    (r,c) = im_shape
    diagonal = math.ceil(np.sqrt(r*r + c*c))
    r_step = window - 3

    t_max = 2 * np.pi
    t_min = 0
    t_step = float(np.pi/180)

    """ Your code starts here """

    r2 = np.arange(-1 * diagonal, diagonal + r_step, r_step)
    t2 = np.arange(t_min, t_max + t_step, t_step)
    A = np.zeros((r2.shape[0], t2.shape[0]))
    for i in range(len(rhos)):
        # for each rho and theta, vote for the nearest one
        try:
            r_idx = int((rhos[i] + diagonal) / r_step)
            t_idx = int(thetas[i] / t_step)
            A[r_idx][t_idx] += 1
        except:
            continue
        
    a = find_peak_params(A, [r2,t2], window, threshold)

    # return rhos,thetas
    return a[1][:num_lines],  a[2][:num_lines]




"""Helper functions: You should not have to touch the following functions.
"""
def trim(frame):
    if not np.sum(frame[0]):
        return trim(frame[1:])
    if not np.sum(frame[-1]):
        return trim(frame[:-2])
    if not np.sum(frame[:,0]):
        return trim(frame[:,1:])
    if not np.sum(frame[:,-1]):
        return trim(frame[:,:-2])
    return frame