import os.path
from os import walk
from PIL import Image
import numpy as np

'''
This class is used to load (stacked) TIFF files from specific paths
'''
class TifLoader:
	
	def __init__(self):
		return

	'''
	loads a TIFF file also if it is stacked with multiple images
	'''
	def loadTIFFile(self, file):
		images = []
		print("Loading " + file)
		image = Image.open(file)

		if(image.format == "TIFF"):
			for i in range(image.n_frames):
				image.seek(i)
				images.append(np.asarray(image, dtype=np.uint8))
		else:
			images.append(np.asarray(image, dtype=np.uint8))

		return images

	'''
	finds all TIFF files in path recursive or the file if path is an absolute TIFF file path
	'''
	def findTIFFiles(self, path):

		files = []
		if os.path.isdir(path):
			for(dirpath, dirnames, filenames) in walk(path):
				for f in filenames:
					files.append(os.path.abspath(os.path.join(dirpath, f)))
		else:
			if os.path.isfile(path):
				files.append(os.path.abspath(path))

		# Filter files for the specific TIFF File endings
		files = [f for f in files if os.path.splitext(os.path.basename(f))[1].lower() in [".tif", ".tiff"]]
		return files

	def __str__(self):
		return "I am a TIFF File Loader!"