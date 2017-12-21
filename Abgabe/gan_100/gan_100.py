import os
import tensorflow as tf
import numpy as np
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sess = tf.Session()

# Define directory for tensorboard logs
LOGDIR = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# CONSTANTS

# format of the input reader
HEIGHT, WIDTH, CHANNEL = 100, 100, 1
BATCH_SIZE = 50
EPOCH = 15000

saver = None

# Load dataset
# MNIST contains 28x28 pictures of handwritten numbers
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/")

# Load our dataset
def process_data(): 
	current_dir = os.getcwd()
	# parent = os.path.dirname(current_dir)
	image_dir = os.path.join(current_dir, 'objects_bw_centered')
	images = []
	for each in os.listdir(image_dir):
		images.append(os.path.join(image_dir,each))
	# print images    
	all_images = tf.convert_to_tensor(images, dtype = tf.string)

	images_queue = tf.train.slice_input_producer([all_images])

	content = tf.read_file(images_queue[0])
	image = tf.image.decode_jpeg(content, channels = CHANNEL)
	# sess1 = tf.Session()
	# sess1.run(image)
	image = tf.image.random_flip_left_right(image)
	# image = tf.image.random_brightness(image, max_delta = 0.1)
	# image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)
	# noise = tf.Variable(tf.truncated_normal(shape = [HEIGHT,WIDTH,CHANNEL], dtype = tf.float32, stddev = 1e-3, name = 'noise')) 
	# print image.get_shape()
	size = [HEIGHT, WIDTH]
	# image = tf.image.resize_images(image, size)
	image.set_shape([HEIGHT,WIDTH,CHANNEL])
	# image = image + noise
	# image = tf.transpose(image, perm=[2, 0, 1])
	# print image.get_shape()

	image = tf.cast(image, tf.float32)
	image = image / 255.0
	images_batch = tf.train.shuffle_batch([image], batch_size = BATCH_SIZE, num_threads = 4, capacity = 200 + 3* BATCH_SIZE, min_after_dequeue = 200)
	# num_images = len(images)

	return images_batch

'''
first_image = image_batch[1]
print(first_image)
first_imag = np.array(first_image, dtype='object')
pixels = first_imag.reshape((100, 100))
plt.imshow(first_imag, cmap='gray')
plt.show()
'''



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
		if reuse:
			tf.get_variable_scope().reuse_variables()

		epsilon = 1e-5
		# x_image is a 28x28 picture

		# First convoluational layer
		# Finds 32 5x5 features
		# After the first convolution the input images will be 14x14 pixel
		d_w1 = tf.get_variable("d_w1", [2, 2, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b1 = tf.get_variable("d_b1", [32], initializer=tf.constant_initializer(0))

		d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding="SAME")
		d1 = d1 + d_b1
		#d1_mean, d1_var = tf.nn.moments(d1, [0])
		#d1_beta = tf.Variable(tf.zeros([32]))
		#d1_scale = tf.Variable(tf.ones([32]))
		#d1 = tf.nn.batch_normalization(d1, d1_mean, d1_var, d1_beta, d1_scale, epsilon)
		d1 = tf.contrib.layers.batch_norm(d1, epsilon=epsilon, scope="d_bn1")
		d1 = tf.nn.leaky_relu(d1, 0.2)
		d1 = tf.nn.avg_pool(d1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

		# Second convolutional layer
		# This finds 64 5x5 features
		# Second convolution shrinks the images to 7x7 pixels
		d_w2 = tf.get_variable("d_w2", [2, 2, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b2 = tf.get_variable("d_b2", [64], initializer=tf.constant_initializer(0))

		d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding="SAME")
		d2 = d2 + d_b2
		#d2_mean, d2_var = tf.nn.moments(d2, [0])
		#d2_beta = tf.Variable(tf.zeros([64]))
		#d2_scale = tf.Variable(tf.ones([64]))
		#d2 = tf.nn.batch_normalization(d2, d2_mean, d2_var, d2_beta, d2_scale, epsilon)
		d2 = tf.nn.leaky_relu(d2, 0.2)
		d2 = tf.nn.avg_pool(d2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

		# First fully connected layer
		d_w3 = tf.get_variable("d_w3", [4 * 25 * 25 * 16, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b3 = tf.get_variable("d_b3", [512], initializer=tf.constant_initializer(0))

		d3 = tf.reshape(d2, [-1 ,4 * 25 * 25 * 16])
		d3 = tf.matmul(d3, d_w3) + d_b3
		#d3_mean, d3_var = tf.nn.moments(d3, [0])
		#d3_beta = tf.Variable(tf.zeros([512]))
		#d3_scale = tf.Variable(tf.ones([512]))
		#d3 = tf.nn.batch_normalization(d3, d3_mean, d3_var, d3_beta, d3_scale, epsilon)
		d3 = tf.nn.leaky_relu(d3, 0.2)

		# Second fully connected layer
		d_w4 = tf.get_variable("d_w4", [512, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		d_b4 = tf.get_variable("d_b4", [1], initializer=tf.constant_initializer(0))

		d4 = tf.matmul(d3, d_w4) + d_b4
		print("Discriminator Shapes")
		print(d1.shape)
		print(d2.shape)
		print(d3.shape)
		print(d4.shape)

		return d4

'''
Generator that creates images from it's weights and a noise vector
'''
def generator(batch_size, z_dim):
	with tf.name_scope("gernerator"):
		# Create a noise vector
		z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name="z")

		# Deconvolutional layer
		g_w1 = tf.get_variable("g_w1", [z_dim, 2 * 20 * 20 * 50], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b1 = tf.get_variable("g_b1", [2 * 20 * 20 * 50], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g1 = tf.matmul(z, g_w1) + g_b1
		g1 = tf.reshape(g1, [-1, 200, 200, 1])
		g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope="g_bn1")
		g1 = tf.nn.leaky_relu(g1, 0.2)

		# Generate 75 features
		g_w2 = tf.get_variable("g_w2", [4, 4, 1, 3*z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b2 = tf.get_variable("g_b2", [3*z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding="SAME")
		g2 = g2 + g_b2
		g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope="g_bn2")
		g2 = tf.nn.leaky_relu(g2, 0.3)
		g2 = tf.image.resize_images(g2, [200, 200])

		# Generate 50 features
		g_w3 = tf.get_variable("g_w3", [2, 2, 3*z_dim/4, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		#g_w3 = tf.get_variable("g_w3", [2, 2, 1, z_dim/2], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b3 = tf.get_variable("g_b3", [z_dim/2], initializer=tf.truncated_normal_initializer(stddev=0.02))
		#g3 = tf.nn.conv2d(g1, g_w3, strides=[1, 2, 2, 1], padding="SAME")
		g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding="SAME")
		g3 = g3 + g_b3
		g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope="g_bn3")
		g3 = tf.nn.leaky_relu(g3, 0.2)
		g3 = tf.image.resize_images(g3, [200, 200])

		# Generate 25 features
		g_w4 = tf.get_variable("g_w4", [3, 3, z_dim/2, z_dim/4], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b4 = tf.get_variable("g_b4", [z_dim/4], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding="SAME")
		g4 = g4 + g_b4
		g4 = tf.contrib.layers.batch_norm(g4, epsilon=1e-5, scope="g_bn4")
		g4 = tf.nn.leaky_relu(g4, 0.2)
		g4 = tf.image.resize_images(g4, [200, 200])

		# Final convolution with one output channel
		g_w5 = tf.get_variable("g_w5", [1, 1, z_dim/4, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
		g_b5 = tf.get_variable("g_b5", [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		g5 = tf.nn.conv2d(g4, g_w5, strides=[1, 2, 2, 1], padding="SAME")
		g5 = g5 + g_b5
		g5 = tf.sigmoid(g5)

		print("Generator Shapes")
		print(g1.shape)
		#print(g2.shape)
		print(g3.shape)
		print(g4.shape)
		print(g5.shape)


		# sigmoind to sharpen images
		# dim of g5: batch_size x 28 x 28 x 1

		return g5

def generate():
	import matplotlib.pyplot as plt

	global saver

	image_batch = process_data()

	# Variable declaration and initialization
	# batch_size = 10
	z_dimensions = 100

	# real images to discriminator
	x_placeholder = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="x_placeholder")

	# Gz will contain generated images
	Gz = generator(BATCH_SIZE, z_dimensions)

	# Dx will contain discrininator's probability values for real images
	Dx = discriminator(x_placeholder)

	# Dg will contain discrininator's probability values for generated (fake) images
	Dg = discriminator(Gz, reuse=True)

	# Losses for generator and discriminator
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

	d_loss_real = tf.reduce_mean(
		tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([BATCH_SIZE, 1], 0.9)))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
	d_loss = d_loss_real + d_loss_fake

	tvars = tf.trainable_variables()

	d_vars = [var for var in tvars if 'd_' in var.name]
	g_vars = [var for var in tvars if 'g_' in var.name]

	# Training of generator and discriminator
	#with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
		#d_trainer_fake = tf.train.AdamOptimizer(0.001).minimize(d_loss_fake, var_list=d_vars)
		#d_trainer_real = tf.train.AdamOptimizer(0.001).minimize(d_loss_real, var_list=d_vars)

		#g_trainer = tf.train.AdamOptimizer(0.001).minimize(g_loss, var_list=g_vars)

	sess = tf.Session()
	saver = tf.train.Saver()

	sess.run(tf.local_variables_initializer())
	sess.run(tf.global_variables_initializer())

	amountImages = 10
	with tf.Session() as sess:
		saver.restore(sess, 'models/pretrained_gan.ckpt')

		z_batch = np.random.normal(0, 1, size=[amountImages, z_dimensions])
		z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
		generated_images = generator(BATCH_SIZE, z_dimensions)

		images = sess.run(generated_images, {z_placeholder: z_batch})
		for i in range(amountImages):
			plt.imshow(images[i].reshape([HEIGHT, WIDTH]), cmap='Greys')
			# plt.show()
			print("Saving gen_result_images/result_" + str(i) + ".png")
			plt.savefig("gen_result_images/result_" + str(i) + ".png")


def train():

	global saver
	
	image_batch = process_data()

	# Variable declaration and initialization
	# batch_size = 10
	z_dimensions = 100

	# real images to discriminator
	x_placeholder = tf.placeholder(tf.float32, shape=[None, 100, 100, 1], name="x_placeholder")

	# Gz will contain generated images
	Gz = generator(BATCH_SIZE, z_dimensions)

	# Dx will contain discrininator's probability values for real images
	Dx = discriminator(x_placeholder)

	# Dg will contain discrininator's probability values for generated (fake) images
	Dg = discriminator(Gz, reuse=True)

	# Losses for generator and discriminator
	g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

	d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.fill([BATCH_SIZE, 1], 0.9)))
	d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))

	#g_loss = tf.reduce_mean(tf.log(Dg))
	#d_loss_real = tf.reduce_mean(tf.log(Dx))
	#d_loss_fake = tf.reduce_mean(tf.log(Dg))

	d_loss = d_loss_real + d_loss_fake

	tvars = tf.trainable_variables()

	d_vars = [var for var in tvars if 'd_' in var.name]
	g_vars = [var for var in tvars if 'g_' in var.name]

	# Training of generator and discriminator
	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:

		d_trainer_fake = tf.train.RMSPropOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
		d_trainer_real = tf.train.RMSPropOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

		g_trainer = tf.train.RMSPropOptimizer(0.0003).minimize(g_loss, var_list=g_vars)

	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
	#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	sess = tf.Session()
	saver = tf.train.Saver()

	# Outputs scalar values to tensorboard
	tf.summary.scalar('Generator_loss', g_loss)
	tf.summary.scalar('Discriminator_loss_real', d_loss_real)
	tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

	tf.summary.image('Input_images', image_batch, 2)

	images_for_tensorboard = generator(BATCH_SIZE, z_dimensions)
	tf.summary.image('Generated_images', images_for_tensorboard, 5)
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

	PRE_TRAIN_DIS = 0

	# Pre-train discriminator
	print("Pretrain Discriminator")
	print("Iteration: ", 0)
	for i in np.arange(PRE_TRAIN_DIS)+1:
		batch = image_batch.eval()

		_, __, ___, ____ = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
									{x_placeholder: batch})

		if i % 10 == 0:
			summary = sess.run(merged, {x_placeholder: batch})
			writer.add_summary(summary, i)
			print("Iteration: ", i)

	print("Start normal Training")
	for i in np.arange(EPOCH-PRE_TRAIN_DIS)+1 + PRE_TRAIN_DIS:
		batch = image_batch.eval(session=sess)

		# loss more than 0.5 means discriminator gets fake images at less than 50% of the time

		if dLossFake > 0.65:
			_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_fake, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: batch})

		if gLoss > 0.35:
			# Train the generator
			_, dLossReal, dLossFake, gLoss = sess.run([g_trainer, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: batch})

		# loss more than 0.4 means discriminator is actually not that good at getting real images
		if dLossReal > 0.35:
			_, dLossReal, dLossFake, gLoss = sess.run([d_trainer_real, d_loss_real, d_loss_fake, g_loss],
													{x_placeholder: batch})

		if i % 10 == 0:
			# real_image_batch = mnist.validation.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
			with sess.as_default():
				batch = image_batch.eval()

			summary = sess.run(merged, {x_placeholder: batch})
			writer.add_summary(summary, i)
			print("Iteration ", i)

		if i % 2000 == 0:
			save_path = saver.save(sess, "models/pretrained_gan.ckpt", global_step=i)
			print("saved to %s" % save_path)

	save_path = saver.save(sess, "models/pretrained_gan.ckpt")
	print("saved to %s" % save_path)

if __name__ == "__main__":
	#try:
		#train()
		generate()
	#except KeyboardInterrupt:
	#	print("Catched. :)")
	#finally:
		# Like for example save the tensorflow model of the actual iteration
	#	save_path = saver.save(sess, "models/pretrained_gan.ckpt")
	#	print("saved to %s" % save_path)