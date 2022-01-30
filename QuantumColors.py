from PIL import Image
import numpy as np
import math
from os import path

#Image IO
def openImage(name):
    return np.array(Image.open(name))

def saveImage(imageArray, name):
    count = 1
    while path.exists(name[:name.index('.')] + str(count) + name[name.index('.'):]):
        count += 1
    Image.fromarray(imageArray).save(name[:name.index('.')] + str(count) + name[name.index('.'):])

#=========================================

def getColors(image, size=1):
    size = int(size)
    result = []
    for row in range(0, len(image), size):
        for col in range(0, len(image[row]),size):
            sum = [np.average(image[row:min(row+size, len(image)), col:min(col+size, len(image[row])), i]) for i in range(image.shape[2])]
            sum = np.clip(np.trunc(sum), 0, 255)
            result.append(sum)
    
    return np.array(result).astype('uint8')
                    
def pixelate(image, size):
    if size == 1:
        return image
    
    size = int(size)
    result = []
    for row in range(0, len(image), size):
        newRow = []
        for col in range(0, len(image[row]),size):
            sum = [np.average(image[row:min(row+size, len(image)), col:min(col+size, len(image[row])), i]) for i in range(3)]
            sum = np.clip(np.trunc(sum), 0, 255)
            newRow.append(sum)
        result.append(newRow)
    return np.array(result).astype('uint8')

def getClosestColor(pixel, colors):
    closestColor = colors[0]
    minDist = distance(pixel, closestColor)
    for color in colors:
        dist = distance(pixel, color)
        if minDist > dist:
            closestColor = color
            minDist = dist
    return closestColor

def distance(a, b):
    return math.sqrt(np.sum([(int(a[i])-int(b[i]))**2 for i in range(len(a))]))

#=============================================

def main():
    image = openImage("2021.jpg")
    print("Image loaded.")
    NUM_COLORS = 12

    resolution = (image.shape[0]+image.shape[1])/2
    colors = getColors(image, resolution)
    while len(colors) < NUM_COLORS:
        resolution *= 0.9
        colors = getColors(image, resolution)
    print("Colors found.", len(colors))

    image = pixelate(image, 1)
    print("Pixelated image.")

    for row in range(len(image)):
        for col in range(len(image[row])):
            image[row][col] = getClosestColor(image[row][col], colors)
    print("Quantized colors.")

    saveImage(image, "out.png")
    print("Done.")

if __name__ == '__main__':
    main()

#=============================================
