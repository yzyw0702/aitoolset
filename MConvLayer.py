import tensorflow as tf

class MConvUnit:
	def __init__(self, wK, hK,prevNK, nK, layer_name='L1'):
		self.wK = wK  # kernel width
		self.hK = hK  # kernel height
		self.prevNK = prevNK
		self.nK = nK  # kernel number
		# create var Weights
		self.W = tf.get_variable(
			name=layer_name + '_weights',
			initializer=tf.truncated_normal_initializer(stddev=0.01),
			shape=[self.hK, self.wK, prevNK, self.nK])
		# create var bias
		self.b = tf.get_variable(
			name=layer_name + '_bias',
			initializer=tf.zeros_initializer,
			shape=[self.nK])
	
	def flowFrom(self, input, isTrain=False, isPool=True, probDropout=None):
		act_conv = tf.nn.conv2d(input, self.W, [1, 1, 1, 1], 'SAME') + self.b
		act_relu = tf.nn.relu(act_conv)
		act_dropout = None
		if isTrain:
			act_dropout = tf.nn.dropout(act_relu, probDropout)
		else:
			act_dropout = act_relu
		if isPool:
			return tf.nn.max_pool(act_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
		else:
			return act_dropout


class MFcUnit:
	def __init__(self, nX, nY, layer_name='fc1'):
		self.nX = nX  # input unit number
		self.nY = nY  # label number
		# create var Weights
		self.W = tf.get_variable(
			name=layer_name + '_weights',
			initializer=tf.truncated_normal_initializer(stddev=0.01),
			shape=[nX, nY])
		# create var bias
		self.b = tf.get_variable(
			name=layer_name + '__bias',
			initializer=tf.zeros_initializer,
			shape=[nY])
		
	def flowFrom(self, input):
		return tf.matmul(input, self.W) + self.b
