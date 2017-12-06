import numpy as np
import os
from PIL import Image
import glob

resourcePath = os.getcwd() + "/resources/objects/"
reshapedPath = os.getcwd() + "/resources/objects_reshaped/"
centeredPath = os.getcwd() + "/resources/objects_centered/"

'''
Takes a image and newDim array defining the new dimensions
'''
def distortImageToResolution(image, newDim=[100, 100]):
    newDim.extend(np.zeros(2))
    newDim = newDim[:2]

    img = np.array(image.convert("L"), dtype=np.uint8)

    x_scale = newDim[0] / img.shape[0]
    y_scale = newDim[1] / img.shape[1]
    x_dist = np.arange(newDim[0]) / x_scale
    y_dist = np.arange(newDim[1]) / y_scale

    x_scaled = img[x_dist.astype(int)]
    newImage = np.transpose(x_scaled)[y_dist.astype(int)]

    return Image.fromarray(newImage)

'''
Takes a image and newDim array defining the new dimensions
'''
def convertImageToResolution(image, newDim=[100, 100]):
    newDim.extend(np.zeros(2))
    newDim = newDim[:2]

    img = np.array(image.convert("L"), dtype=np.uint8)
    startIndices = ((newDim - np.array(img.shape)) / 2).astype(int)

    newImage = np.ones(newDim, dtype=np.uint8) * 255
    newImage[startIndices[0]:startIndices[0] + img.shape[0], startIndices[1]:startIndices[1] + img.shape[1]] = img

    return Image.fromarray(newImage)

def processImages():
    for filename in glob.glob(resourcePath + "*"):
        im = Image.open(filename)

        reshaped = distortImageToResolution(im, [100, 100])
        centered = convertImageToResolution(im, [100, 100])

        reshaped.save(reshapedPath + os.path.basename(filename), "PNG")
        centered.save(centeredPath + os.path.basename(filename), "PNG")

processImages()