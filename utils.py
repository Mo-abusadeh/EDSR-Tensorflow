import tensorflow as tf
import tensorflow.contrib.slim as slim

"""
Creates a convolutional residual block
as defined in the paper. More on
this inside model.py

x: input to pass through the residual block
channels: number of channels to compute
stride: convolution stride
"""

# This is the newly proposed resBlock architecture
def resBlock(x,channels=64,kernel_size=[3,3],scale=1):
	tmp = slim.conv2d(x,channels,kernel_size,activation_fn=None)
	tmp = tf.nn.relu(tmp)
	tmp = slim.conv2d(tmp,channels,kernel_size,activation_fn=None)
	tmp *= scale
	return x + tmp

"""
Method to upscale an image using
conv2d transpose. Based on upscaling
method defined in the paper

x: input to be upscaled
scale: scale increase of upsample
features: number of features to compute
activation: activation function
"""

"""
###########################################
COMMENTED OUT CODE: GAS-CNN
NO UPSAMPLING MODULE
###########################################
"""

"""
# Implementing a pre-trained x2 network for x4 model (EDSR)
# This pre-training strategy accelerates the training and improves the final performance (Graph in paper)
def upsample(x,scale=2,features=64,activation=tf.nn.relu):
	assert scale in [2,3,4]
	x = slim.conv2d(x,features,[3,3],activation_fn=activation)
	if scale == 2:
		ps_features = 3*(scale**2)
		x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
		#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
		x = PS(x,2,color=True)
	elif scale == 3:
		ps_features =3*(scale**2)
		x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
		#x = slim.conv2d_transpose(x,ps_features,9,stride=1,activation_fn=activation)
		x = PS(x,3,color=True)
	elif scale == 4:
		ps_features = 3*(2**2)
		for i in range(2):
			x = slim.conv2d(x,ps_features,[3,3],activation_fn=activation)
			#x = slim.conv2d_transpose(x,ps_features,6,stride=1,activation_fn=activation)
			x = PS(x,2,color=True)
	return x
"""
"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""
def _phase_shift(I, r):
	bsize, a, b, c = I.get_shape().as_list()
	bsize = tf.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
	X = tf.reshape(I, (bsize, a, b, r, r))
	X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
	X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, b, a*r, r
	X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
	X = tf.concat([tf.squeeze(x, axis=1) for x in X],2)  # bsize, a*r, b*r
	return tf.reshape(X, (bsize, a*r, b*r, 1))

"""
Borrowed from https://github.com/tetrachrome/subpixel
Used for subpixel phase shifting after deconv operations
"""
def PS(X, r, color=False):
	if color:
		Xc = tf.split(X, 3, 3)
		X = tf.concat([_phase_shift(x, r) for x in Xc],3)
	else:
		X = _phase_shift(X, r)
	return X

"""
Tensorflow log base 10.
Found here: https://github.com/tensorflow/tensorflow/issues/1666
"""
def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator



"""
###########################################
PRE-PROCESSING FUNCTION: GAS-CNN
FUNCTION FOR GENERATING IMAGES WITH GIBBS
NEEDS EDITING TO FIT IN THIS CODE
###########################################
"""

"""
# Load the image
image = cv2.imread('MRI_image.jpg', cv2.IMREAD_GRAYSCALE)

# Perform Fourier Transform
fft_image = fft2(image)
fft_image_shifted = fftshift(fft_image)

# Create a low-pass filter to attenuate 20% of highest frequencies
rows, cols = image.shape
center_row, center_col = rows // 2, cols // 2
radius = min(center_row, center_col) * 0.2
mask = np.zeros_like(fft_image)
y, x = np.ogrid[-center_row:rows - center_row, -center_col:cols - center_col]
mask_area = x**2 + y**2 <= radius**2
mask[mask_area] = 1

# Apply the filter
filtered_fft = fft_image_shifted * mask

# Inverse Fourier Transform
filtered_image = ifftshift(filtered_fft)
restored_image = ifft2(filtered_image).real

# Display the original and the corrupted image
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Corrupted Image')
plt.imshow(restored_image, cmap='gray')
plt.axis('off')

plt.show()

"""