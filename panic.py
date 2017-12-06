import os
import reader
import matplotlib.pyplot as plt

'''
This function expects a 2 dimensional numpy array with grayscale values
'''
def displayImage(image):
	# Mind that matplotlib adjust pixel scale intensity if vmin and vmax is not used
	plt.imshow(image, cmap='Greys_r', vmin=0, vmax=255, interpolation='nearest')
	plt.show()

resourcePath = os.getcwd() + "/resources/3d_segmentation/"

def main():

	tl = reader.TifLoader()
	tifFiles = tl.findTIFFiles(resourcePath)

	input = {}
	for file in tifFiles:
		key = os.path.splitext(os.path.basename(file))[0]
		input[key] = tl.loadTIFFile(file)

	for file in input:
		imageData = input[file][0]
		# Displays the first image
		displayImage(imageData)

	# Define training and test data for NN action
	x_train = input['training']
	x_test = input['testing']

	# Now comes the fun part with neuronal nets !!!

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Catched. :)")
    finally:
        #Like for example save the tensorflow model of the actual iteration
        print("Finally stopping.")