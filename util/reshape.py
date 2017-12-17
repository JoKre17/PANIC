import numpy as np
import os
from PIL import Image
import glob

resourcePath = os.getcwd() + "/resources/objects/"
reshapedPath = os.getcwd() + "/resources/objects_reshaped/"
centeredPath = os.getcwd() + "/resources/objects_centered/"
reshapedBWPath = os.getcwd() + "/resources/objects_bw_reshaped/"
centeredBWPath = os.getcwd() + "/resources/objects_bw_centered/"

# make dirs
if not os.path.exists(resourcePath):
    os.makedirs(resourcePath)

if not os.path.exists(reshapedPath):
    os.makedirs(reshapedPath)

if not os.path.exists(centeredPath):
    os.makedirs(centeredPath)

if not os.path.exists(reshapedBWPath):
    os.makedirs(reshapedBWPath)

if not os.path.exists(centeredBWPath):
    os.makedirs(centeredBWPath)

'''
Takes a image and newDim array defining the new dimensions
works actually only for RGBA image formats from pillow package
'''
def distortImageToResolution(image, newDim=[100, 100]):
    newDim.extend(np.zeros(2))
    newDim = newDim[:2]

    img = np.array(image, dtype=np.uint8)

    x_scale = newDim[0] / img.shape[0]
    y_scale = newDim[1] / img.shape[1]
    x_dist = (np.array(np.arange(newDim[0])) / x_scale).astype(int)
    y_dist = (np.array(np.arange(newDim[1])) / y_scale).astype(int)

    img = img[x_dist]
    newImage = np.transpose(img, (1,0,2))[y_dist]

    return Image.fromarray(newImage)

'''
Takes a image and newDim array defining the new dimensions
works actually only for RGBA image formats from pillow package
'''
def centerImageToResolution(image, newDim=[100, 100]):
    newDim.extend(np.zeros(2))
    newDim = newDim[:2]

    img = np.array(image, dtype=np.uint8)
    startIndices = ((newDim - np.array(img.shape[:2])) / 2).astype(int)

    newImage = np.transpose(np.array([np.zeros(newDim, dtype=np.uint8)]*4), (1,2,0))
    newImage[startIndices[0]:startIndices[0] + img.shape[0], startIndices[1]:startIndices[1] + img.shape[1]] = img

    return Image.fromarray(newImage)

def processPNGImages():
    for filename in glob.glob(resourcePath + "*"):
        im = Image.open(filename)

        reshaped = distortImageToResolution(im, [100, 100])
        centered = centerImageToResolution(im, [100, 100])

        # convert transparent pixel to white
        reshaped_arr = np.array(reshaped)
        reshaped_arr = reshaped_arr + np.array(reshaped_arr[:, :] == 0) * 255
        reshaped_arr = np.array(reshaped_arr, 'uint8')
        reshaped_bw = Image.fromarray(reshaped_arr, 'RGBA').convert('L')

        centered_arr = np.array(centered)
        centered_arr = centered_arr + np.array(centered_arr[:, :] == 0) * 255
        centered_arr = np.array(centered_arr, 'uint8')
        centered_bw = Image.fromarray(centered_arr, 'RGBA').convert('L')

        reshaped.save(reshapedPath + os.path.basename(filename).split(".")[0] + "." + im.format, im.format)
        centered.save(centeredPath + os.path.basename(filename).split(".")[0] + "." + im.format, im.format)
        reshaped_bw.save(reshapedBWPath + os.path.basename(filename).split(".")[0] + "." + 'JPEG', 'JPEG')
        centered_bw.save(centeredBWPath + os.path.basename(filename).split(".")[0] + "." + 'JPEG', 'JPEG')

def loadImagesFromDirAsArray(dirPath):
    processPNGImages()
    images = []
    for filename in glob.glob(dirPath + "*"):
        im = Image.open(filename)
        images.append(np.array(im.convert("L"), dtype=np.uint8))

    return images

def getFilesInDir(dir):
    filenames = []
    for filename in glob.glob(dir + "*"):
        filenames.append(filename)

    return filenames


def getDistortedImagePaths():
    return getFilesInDir(reshapedPath)

def loadDistortedImagesAsArray():
    return loadImagesFromDirAsArray(reshapedPath)

def getCenterdImagePaths():
    return getFilesInDir(centeredPath)

def loadCenteredImagesAsArray():
    return loadImagesFromDirAsArray(centeredPath)

if __name__ == '__main__':
    print("Running reshape")
    processPNGImages()