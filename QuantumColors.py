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
    Image.fromarray(imageArray).save(name[:name.index('.')] + str(count) + name[name.index('.'):])

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

def map_palette(image, labels, centroids, new_palette, mapping):
    neighbors = np.array([[1,1]])

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

                    neighbors[i] = [neighbor[1], -neighbor[0]]
            label_count = label_count/np.sum(label_count)
            color = np.sum(new_palette*label_count[:, np.newaxis], axis=0)
            if x == 100 and y == 100:
                print(color)
            """
            color = new_palette[mapping.index(label)]

            c = 0
            image[x,y] = image[x,y]*c + (1-c)*color
    return np.clip(image, 0, 255).astype('uint8')

def main():
    image = open_image("matterhorn.jpg")
    print("Image loaded.")
    NUM_COLORS = 5

    #
    colors = image.reshape([image.shape[0]*image.shape[1], 3])
    centroids = init_means(image, NUM_COLORS, size=2)
    centroids, labels = k_means(colors, centroids)
    centroids = centroids.astype('uint8')
    labels = labels.reshape([image.shape[0], image.shape[1]])


    aro = np.array([[61, 165, 66],
                    [167, 211, 121],
                    [255,255,255],
                    [169, 169, 169],
                    [0,0,0]])
    """
    aro_colors = open_image("aro2.jpg")
    aro = init_means(aro_colors, NUM_COLORS, size=2)
    aro_colors = aro_colors.reshape([aro_colors.shape[0]*aro_colors.shape[1], 3])
    aro, _ = k_means(aro_colors, aro)
    """

    #mapping = assign_new_pallete(centroids, aro)
    mapping = []

    #labels = locally_scramble_labels(labels)
    image = map_palette(image, labels, centroids, aro, mapping)



    print("Quantized colors.")

    save_image(image, "out.png")
    print("Done.")

if __name__ == '__main__':
    main()

#=============================================
