import os
import reader
import util
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import multiprocessing as mp
import numpy as np

'''
This function expects a 2 dimensional numpy array with grayscale values
'''
def displayImage(image):
	# Mind that matplotlib adjust pixel scale intensity if vmin and vmax is not used
	plt.imshow(image, cmap='Greys_r', vmin=0, vmax=255, interpolation='nearest')
	plt.show()

resourcePath = os.getcwd() + "/resources/3d_segmentation/"

def loadSegmentationResources():
	tl = reader.TifLoader()
	tifFiles = tl.findTIFFiles(resourcePath)

	input = {}
	for file in tifFiles:
		key = os.path.splitext(os.path.basename(file))[0]
		input[key] = tl.loadTIFFile(file)

	for file in input:
		imageData = input[file][0]
	# Displays the first image
	# displayImage(imageData)

	# Define training and test data for NN action
	x_train = input['training']
	x_test = input['testing']

	return x_train, x_test

def gen_image(arr):
	print(np.asarray(arr) * 255)
	two_d = tf.to_int64(np.asarray(arr) * 255)
	plt.imshow(two_d, interpolation='nearest')
	return plt

def main():
	#x_train, x_test = loadSegmentationResources()
	images = util.loadCenteredImagesAsArray()

	filenames = util.getCenterdImagePaths()

	# step 2
	filename_queue = tf.train.string_input_producer(filenames)

	# step 3: read, decode and resize images
	reader = tf.WholeFileReader()
	filename, content = reader.read(filename_queue)
	image = tf.image.decode_png(content, channels=1)
	image = tf.reshape(image, [100, 100])
	# Generate batch
	image = tf.cast(image, tf.float32)
	num_preprocess_threads = mp.cpu_count()
	batch_size = 10
	image_batch = tf.train.shuffle_batch(
		[image],
		batch_size=batch_size,
		num_threads=num_preprocess_threads,
		capacity=42,
		min_after_dequeue=10)

	gen_image(image_batch[0]).show()

	# Now comes the fun part with neuronal nets !!!

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Catched. :)")
    finally:
        #Like for example save the tensorflow model of the actual iteration
        print("Finished PANIC.")