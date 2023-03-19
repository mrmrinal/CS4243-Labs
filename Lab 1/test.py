import math
import numpy as np

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
    return math.sqrt(res_r), math.sqrt(res_g), math.sqrt(res_b)

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



a = [[1, 2, 3], [4, 5, 6]]
b = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
nd_a = np.array(a)
nd_b = np.array(b)


b = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]], [[4, 4, 4], [5, 5, 5], [6, 6, 6]]]
nd_b = np.array(b)

c = np.array([[1, 2, 3], [4, 5, 6]])


# img dimensions = (4, 4, 3)
img = np.array([[[0, 16, 32], [1, 17, 33], [2, 18, 34], [3, 19, 35]], [[4, 20, 36], [5, 21, 37], [6, 22, 38], [7, 23, 39]], [[8, 24, 40], [9, 25, 41], [10, 26, 42], [11, 27, 43]], [[12, 28, 44], [13, 29, 45], [14, 30, 46], [15, 31, 47]]])

print(img.shape)
template = np.array([[1, 2], [3, 4]])

print(reshape_image(img, template).shape)

