import os
from PIL import Image
import PIL
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

sess = tf.Session()

# Define directory for tensorboard logs
LOGDIR = "tensorboard/gan_med/bigImages_v2"

# Define input dataset
#data_selection = "standard"
data_selection = "centered_bw"

HEIGHT, WIDTH, CHANNEL = 28, 28, 1

BATCH_SIZE = 50

EPOCH = 5000

# Load dataset
# MNIST contains 28x28 pictures of handwritten numbers
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/")

# Load our dataset
def process_data(): 
	current_dir = os.getcwd()
	# parent = os.path.dirname(current_dir)
	image_dir = os.path.join(current_dir, 'data')
	image_dir = os.path.join(image_dir, data_selection)
	images = []
	for each in os.listdir(image_dir):
		images.append(os.path.join(image_dir,each))
	# print images    
	all_images = tf.convert_to_tensor(images, dtype = tf.string)

	images_queue = tf.train.slice_input_producer([all_images])

	content = tf.read_file(images_queue[0])
	image = tf.image.decode_jpeg(content, channels = CHANNEL)
	#sess1 = tf.Session()
	#sess1.run(image)
	image = tf.image.random_flip_left_right(image)
	#image = tf.image.random_brightness(image, max_delta = 0.1)
	#image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
	# noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
	# print image.get_shape()
	size = [HEIGHT, WIDTH]
	image = tf.image.resize_images(image, size)
	image.set_shape([HEIGHT,WIDTH,CHANNEL])
	# image = image + noise
	# image = tf.transpose(image, perm=[2, 0, 1])
	# print image.get_shape()

	image = tf.cast(image, tf.float32)
	image = image / 255.0
	images_batch = tf.train.shuffle_batch([image], batch_size = BATCH_SIZE, num_threads = 4, capacity = 200 + 3* BATCH_SIZE, min_after_dequeue = 200)
	#num_images = len(images)

	return images_batch

'''
Discriminator that gets images as input and classifies them as real/fake.
This implementation consits of 2 convolutional layers and two fully connected layers.
The 28x28 input images get folded 2  times in the convolotional layers to 14x14 (first) 
and 7x7 (second). Both convolutional layer extract 5x5 features but the first creates 32 
and the second one creates 64. In the fully connected layer the image is reshaped to a one
dimensional array of 64 * 7 * 7 = 3136.  
'''
def discriminator(x_image, reuse=False):
	with tf.name_scope("discriminator"):
		if (reuse):
			tf.get_variable_scope().reuse_variables()

		# x_image is a 28x28 picture

		# First convoluational layer
		# Finds 32 5x5 features
		# After the first convolution the input images will be 14x14 pixel
		d_w1 = tf.get_variable("d_w1", [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b1 = tf.get_variable("d_b1", [32], initializer=tf.constant_initializer(0))

		d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding="SAME")
		d1 = d1 + d_b1
		d1 = tf.nn.relu(d1)
		d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

		# Second convolutional layer
		# This finds 64 5x5 features
		# Second convolution shrinks the images to 7x7 pixels
		d_w2 = tf.get_variable("d_w2", [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b2 = tf.get_variable("d_b2", [64], initializer=tf.constant_initializer(0))

		d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding="SAME")
		d2 = d2 + d_b2
		d2 = tf.nn.relu(d2)
		d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

		# First fully connected layer
		d_w3 = tf.get_variable("d_w3", [7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b3 = tf.get_variable("d_b3", [1024], initializer=tf.constant_initializer(0))

		d3 = tf.reshape(d2, [-1 , 7 * 7 * 64])
		d3 = tf.matmul(d3, d_w3) + d_b3
		d3 = tf.nn.relu(d3)

		# Second fully connected layer
		d_w4 = tf.get_variable("d_w4", [1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable("d_b4", [1], initializer=tf.constant_initializer(0))

		d4 = tf.matmul(d3, d_w4) + d_b4

		return d4

'''
Generator that creates images from it's weights and a noise vector
'''
def generator(batch_size, z_dim):
	with tf.name_scope("gernerator"):
		# Create a noise vector
		z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name="z")

		# Deconvolutional layer
		g_w1 = tf.get_variable("g_w1", [z_dim, 7 * 7 * 64], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b1 = tf.get_variable("g_b1", [7 * 7 * 64], initializer=tf.truncated_normal_initializer(stddev=0.02))

		g1 = tf.matmul(z, g_w1) + g_b1
		g1 = tf.reshape(g1, [-1, 56, 56, 1])
		g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope="bn1")
		g1 = tf.nn.relu(g1)

		# Generate 50 features
		g_w2 = tf.get_variable("g_w2", [3, 3, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b2 = g_b2 = tf.get_variable("g_b2", [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding="SAME")
		g2 = g2 + g_b2
		g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope="bn2")
		g2 = tf.nn.relu(g2)
		g2 = tf.image.resize_images(g2, [56, 56])

		# Generate 25 features
		g_w3 = tf.get_variable("g_w3", [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b3 = tf.get_variable("g_b3", [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding="SAME")
		g3 = g3 + g_b3
		g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope="bn3")
		g3 = tf.nn.relu(g3)
		g3 = tf.image.resize_images(g3, [56, 56])

		# Final convolution with one output channel
		g_w4 = tf.get_variable("g_w4", [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b4 = tf.get_variable("g_b4", [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding="SAME")
		g4 = g4 + g_b4
		g4 = tf.sigmoid(g4)

		# sigmoind to sharpen images
		# dim of g4: batch_size x 28 x 28 x 1

		return g4

'''
Training function that handles the training and placeholder definition
'''
def train():
	
	image_batch = process_data()
	print(image_batch)
	#Image.fromarray(np.asarray(image_batch[0])).show()

	#plt.imshow(image_batch[0])
	#plt.show()

	

	# Variable declaration and initialization
	batch_size = 50
	z_dimensions = 100

	# real images to discriminator
	x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x_placeholder")

	# Gz will contain generated images
	Gz = generator(batch_size, z_dimensions)

	# Dx will contain discrininator's probability values for real images
	Dx = discriminator(x_placeholder)

	# Dg will contain discrininator's probability values for generated (fake) images
	Dg = discriminator(Gz, reuse=True)

	# Losses for generator and discriminator
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([batch_size, 1], 0.9)))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
	d_loss = d_loss_real + d_loss_fake

	tvars = tf.trainable_variables()

	d_vars = [var for var in tvars if 'd_' in var.name]
	g_vars = [var for var in tvars if 'g_' in var.name]

	# Training of generator and discriminator
	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

		d_trainer_fake = tf.train.AdamOptimizer(0.001).minimize(d_loss_fake, var_list=d_vars)
		d_trainer_real = tf.train.AdamOptimizer(0.001).minimize(d_loss_real, var_list=d_vars)

		g_trainer = tf.train.AdamOptimizer(0.001).minimize(g_loss, var_list=g_vars)

	# Outputs scalar values to tensorboard
	tf.summary.scalar('Generator_loss', g_loss)
	tf.summary.scalar('Discriminator_loss_real', d_loss_real)
	tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

	d_real_count_ph = tf.placeholder(tf.float32)
	d_fake_count_ph = tf.placeholder(tf.float32)
	g_count_ph = tf.placeholder(tf.float32)

	tf.summary.scalar('d_real_count', d_real_count_ph)
	tf.summary.scalar('d_fake_count', d_fake_count_ph)
	tf.summary.scalar('g_count', g_count_ph)

	# Sanity check to see how the discriminator evaluates
	# generated and real MNIST images
	d_on_generated = tf.reduce_mean(discriminator(generator(batch_size, z_dimensions)))
	d_on_real = tf.reduce_mean(discriminator(x_placeholder))

	tf.summary.scalar('d_on_generated_eval', d_on_generated)
	tf.summary.scalar('d_on_real_eval', d_on_real)

	sess = tf.Session()
	saver = tf.train.Saver()

	tf.summary.image('Input_images', image_batch, 1)

	images_for_tensorboard = generator(batch_size, z_dimensions)
	tf.summary.image('Generated_images', images_for_tensorboard, 10)
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter(LOGDIR)
	writer.add_graph(sess.graph)
	print("Run tensorboard --logdir=", LOGDIR)

	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())

	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	# Training
	gLoss = 0
	dLossFake, dLossReal = 1, 1
	d_real_count, d_fake_count, g_count = 0, 0, 0

	print("Start training")
	for i in range(2000):
		#real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
		
		print(i)

		with sess.as_default():
			batch = image_batch.eval()

		if dLossFake > 0.6:
			# Train discriminator on generated images
			#print("dlossfake")
			_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: batch})
			#print("after dlossfake")
			d_fake_count += 1

		if gLoss > 0.5:
			# Train the generator
			_, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: batch})
			g_count += 1

		if dLossReal > 0.45:
			# If the discriminator classifies real images as fake,
			# train discriminator on real values
			_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: batch})
			d_real_count += 1

		if i % 10 == 0:
			#real_image_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
			with sess.as_default():
				batch = image_batch.eval()
			summary = sess.run(merged, {x_placeholder: batch, d_real_count_ph: d_real_count,
											d_fake_count_ph: d_fake_count, g_count_ph: g_count})
			writer.add_summary(summary, i)
			d_real_count, d_fake_count, g_count = 0, 0, 0
			print("Iteration ", i)
	
		if i % 5000 == 0:
			save_path = saver.save(sess, "models/pretrained_gan.ckpt", global_step=i)
			print("saved to %s" % save_path)


if __name__ == "__main__":
	train()
	# test()