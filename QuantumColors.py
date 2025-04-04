from PIL import Image
import numpy as np
import math
from os import path
from random import randint, random

#========================================================
#                       Image IO
#========================================================

def open_image(name):
    return np.array(Image.open(name))

def save_image(imageArray, name):
    count = 1
    while path.exists(name[:name.index('.')] + str(count) + name[name.index('.'):]):
        count += 1
    name = name[:name.index('.')] + str(count) + name[name.index('.'):]
    Image.fromarray(imageArray).save(name)
    return name

#========================================================
#                       K-means
#========================================================

def init_means(image, k, size=20):
    size = int(size)
    result = []
    for row in range(0, len(image), size):
        for col in range(0, len(image[row]),size):
            sum = [np.average(image[row:min(row+size, len(image)), col:min(col+size, len(image[row])), i]) for i in range(image.shape[2])]
            sum = np.clip(np.trunc(sum), 0, 255)
            result.append(sum)

    result = np.array(result).astype('uint8')
    indices = np.random.choice(result.shape[0], k, replace=False)
    return result[indices]

def k_means(colors, initial_means, max_iterations=100, tolerance=1e-4):
    """
    Perform K-means clustering on a list of colors in RGB format.

    Args:
        colors (np.ndarray): A 2D numpy array of shape (N, 3), where N is the number of colors
                              and each row represents a color in RGB format.
        initial_means (np.ndarray): A 2D numpy array of shape (K, 3), where K is the number of initial
                                    centroids and each row represents an RGB color.
        max_iterations (int): Maximum number of iterations to run the K-means algorithm.
        tolerance (float): Convergence tolerance, the algorithm stops if the change in centroids is less than this.

    Returns:
        np.ndarray: Final cluster centers (means) in RGB format.
        np.ndarray: Labels indicating which cluster each color belongs to.
    """
    K = initial_means.shape[0]
    N = colors.shape[0]

    # Initialize centroids
    centroids = initial_means.copy()

    for iteration in range(max_iterations):
        # Step 1: Assign each color to the nearest centroid
        distances = np.linalg.norm(colors[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 2: Compute new centroids as the mean of the assigned colors
        new_centroids = np.array([colors[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k] for k in range(K)])

        # Step 3: Check for convergence (if the centroids don't change significantly)
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        if centroid_shift < tolerance:
            print(f'Convergence reached after {iteration+1} iterations.')
            break

        # Update centroids
        centroids = new_centroids

    return centroids, labels

def assign_new_pallete(old_colors, new_colors):
    norms = np.linalg.norm(old_colors[:, np.newaxis] - new_colors, axis=2)
    N = len(old_colors)
    def opt_ordering(order=[]):
        unpicked = list(range(N))
        for o in order:
            unpicked.remove(o)

        if len(unpicked) == 1:
            return order+[unpicked[0]], norms[unpicked[0], N-1]
        min_norm = -1
        opt_order = []
        for i, index in enumerate(unpicked):
            order_copy = order.copy()
            order_copy.append(index)

            order_new, norm = opt_ordering(order_copy)
            norm = norm + norms[index, len(order)]

            if i == 0 or norm < min_norm:
                min_norm = norm
                opt_order = order_new
        return opt_order, min_norm

    opt_order, _ = opt_ordering()
    return opt_order

def locally_scramble_labels(labels, scale=0.5):
    new_labels = labels.copy()
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            label = labels[x,y]

            xn = int(np.round(x + np.random.normal(scale=scale)))
            yn = int(np.round(y + np.random.normal(scale=scale)))
            if 0 <= xn and xn < labels.shape[0] and 0 <= yn and yn < labels.shape[1]:
                 new_labels[xn,yn] = label
    return new_labels

def get_labels(image, centroids):

    colors = image.reshape([image.shape[0]*image.shape[1], 3])
    distances = np.linalg.norm(colors[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    labels = labels.reshape([image.shape[0], image.shape[1]])
    return labels


def map_palette(image, labels, centroids, new_palette, mapping):
    neighbors = np.array([[1,1]]) # specify which dx and dy of the neighbors in one of the quadrants

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            label = labels[x,y]

            """
            label_count = np.zeros(len(mapping))
            label_count[label] += 4
            for _ in range(4):
                for i, neighbor in enumerate(neighbors):
                    xn = x + neighbor[0]
                    yn = y + neighbor[1]
                    if 0 <= xn and xn < image.shape[0] and 0 <= yn and yn < image.shape[1]:
                         label_count[mapping.index(labels[xn,yn])] += 1

                    neighbors[i] = [neighbor[1], -neighbor[0]] # rotate
            label_count = label_count/np.sum(label_count)
            color = np.sum(new_palette*label_count[:, np.newaxis], axis=0)
            if x == 100 and y == 100:
                print(color)
            """
            color = new_palette[mapping.index(label)]

            c = 0
            image[x,y] = image[x,y]*c + (1-c)*color
    return np.clip(image, 0, 255).astype('uint8')

def swap(arr, i, j):
    temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp

def main():
    image = open_image("base_images/brussels.jpg")
    print("Image loaded.")
    NUM_COLORS = 10


    colors = image.reshape([image.shape[0]*image.shape[1], 3])
    centroids = init_means(image, NUM_COLORS, size=2)
    centroids, labels = k_means(colors, centroids)
    centroids = centroids.astype('uint8')
    labels = labels.reshape([image.shape[0], image.shape[1]])
    print(centroids)
    """

    centroids = np.array([[161, 183, 219],
                         [ 36,  46,  50],
                         [119, 129, 142],
                         [137, 149, 165],
                         [180, 179, 179],
                         [ 94, 111, 133],
                         [ 67,  85, 102],
                         [224, 232, 241],
                         [142, 169, 213],
                         [196, 206, 220]])
    labels = get_labels(image, centroids)
    #labels = locally_scramble_labels(labels)


    """
    """
    aro = np.array([[61, 165, 66],
                    [167, 211, 121],
                    [255,255,255],
                    [169, 169, 169],
                    [0,0,0]])
    """

    """
    aro = np.array([[226.25237701, 228.60571271, 229.18341421], # 0, gray 230
                    [174.32817404, 211.0946505,  140.04194009], # 1, light green
                    [125.75347863, 128.12687902, 130.67236658], # 2, gray 130
                    [  5.68240261,   6.18956694,   6.88546664], # 3, black
                    [113.01251307, 159.43222386,  92.19040063], # 4, medium green
                    [ 59.49601538, 131.74868742,  57.21334146], # 5, dark green
                    [190.17555625, 193.94198378, 196.60859846], # 6, gray 193
                    [157.34862714, 159.56541982, 162.19371269], # 7, gray 160
                    [144.0325094,  182.08527673, 110.73680817], # 8, light-medium green
                    [ 63.16066066,  64.18468468,  65.7972973 ]]) # 9, gray 65
    """

    aro_colors = open_image("palette_images/ace2.jpg")
    aro = init_means(aro_colors, NUM_COLORS, size=2)
    aro_colors = aro_colors.reshape([aro_colors.shape[0]*aro_colors.shape[1], 3])
    aro, _ = k_means(aro_colors, aro)

    print(aro)


    mapping = assign_new_pallete(centroids, aro)
    # index: which color on the new palette, value: which color in the old palette
    #mapping = [7, 4, 2, 1, 3, 5, 9, 8, 0, 6]
    #          0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    #mapping = [7, 4, 2, 1, 3, 5, 9, 8, 0, 6]
    #swap(mapping, 8, 7)

    print(mapping)
    #mapping = np.arange(NUM_COLORS)
    #np.random.shuffle(mapping)
    #mapping = [x for x in mapping]
    image = map_palette(image, labels, centroids, aro, mapping)




    print("Saved image as", save_image(image, "output_images/out.png"))

if __name__ == '__main__':
    main()

#=============================================
