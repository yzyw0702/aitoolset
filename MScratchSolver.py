from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
#import os.path
import sys
import MNet
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import gfile

def load_variables_from_checkpoint(sess, start_checkpoint):
	saver = tf.train.Saver(tf.global_variables())
	saver.restore(sess, start_checkpoint)

# base class of all network solvers
class MSolver:
	def __init__(self, root_path='/tmp'):
	# prepare parameters
		self.root_path = root_path
		self.fCfg = os.path.join(self.root_path, 'configTrain.txt')
		self.dInput = os.path.join(self.root_path, 'data')
		self.dTrain = os.path.join(self.root_path, 'train')
		self.dLog = os.path.join(self.root_path, 'trainlogs')
		self.net_type = None # init in _createSolverNet()
	# init logging method
		tf.logging.set_verbosity(tf.logging.INFO)
		# tf.logging.set_verbosity(tf.logging.DEBUG)
	# init session
		self.sess = tf.InteractiveSession()
		self.saver = None # init in _createSolverNet()
		self.stepGlobal = None
	# init graph summary writer
		self.hLogTrain = None # init in _createSolverNet()
		self.hLogEval = None # init in _createSolverNet()

	# create solver network, including tf-based compute graphs and feed dictionaries for both train and validation steps.
	## <cfgNet> configuration construct for neural network
	## <net_type> type of the network
	## <nFeature> feature number of input dataset input_feature
	## <nLabel> label number of predicted classes
	## <return: train_fetches> list of fetches in train_step
	## <return: eval_fetches> list of fetches in validation_step
	def _createSolverNet(self, cfgNet, net_type, nFeature, nLabel):
		tf.logging.debug('>>>> start _createSolverNet')
		# init graph summary writer
		self.hLogTrain = tf.summary.FileWriter(os.path.join(self.dLog, 'train'), self.sess.graph)
		self.hLogEval = tf.summary.FileWriter(os.path.join(self.dLog, 'validation')) 
		# record parameters
		self.net_type = net_type
		# input layer
		input_feature = tf.placeholder(tf.float32, [None, nFeature], name='input_feature')
		input_label = tf.placeholder(tf.int64, [None], name='input_label')
		# hidden layers
		logits, dropout_prob = self._create_model(input_feature, cfgNet, net_type, is_training=True)
		# loss
		with tf.name_scope('loss'):
			loss = tf.losses.sparse_softmax_cross_entropy(
			labels=input_label, logits=logits)
		# learning rate and train steps
		with tf.name_scope('train'):
		# mutable learning-rate
			learning_rate = tf.placeholder( tf.float32, [], name='learning_rate')
		# optimizer
			train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		# evaluation step
		Yc = tf.argmax(logits, 1)
		isTrueYc = tf.equal(Yc, input_label)
		mConfuse = tf.confusion_matrix(input_label, Yc, num_classes=nLabel)
		evaluation_step = tf.reduce_mean(tf.cast(isTrueYc, tf.float32))
		# summary curves
		with tf.get_default_graph().name_scope('eval'):
			tf.summary.scalar('loss', loss)
			tf.summary.scalar('accuracy', evaluation_step)
		hSummary = tf.summary.merge_all(scope='eval')
		# loop variable
		stepGlobal = tf.train.get_or_create_global_step()
		increaseStepGlobal = tf.assign(stepGlobal, stepGlobal + 1)
		self.stepGlobal = stepGlobal
		# saver of all network parameters
		self.saver = tf.train.Saver(tf.global_variables())
		# train_feed.keys: input_feature, input_label, learning_rate, dropout_prob
		# <Return> collect train_fetches
		train_fetches = [hSummary, evaluation_step, loss, train_step, increaseStepGlobal]
		# eval_feed.keys: input_feature, input_label, dropout_prob
		# <Return> collect eval_fetches
		eval_fetches = [hSummary, evaluation_step, mConfuse]
		# <Return> collect test_fetches
		test_fetches = [evaluation_step, mConfuse]
		# <Return> collect feed keys
		feed_keys_dict = {
			'input_feature' : input_feature,
			'input_label' : input_label,
			'learning_rate' : learning_rate,
			'dropout_prob' : dropout_prob
		}
		tf.logging.debug('<<<< end _createSolverNet')
		return train_fetches, eval_fetches, test_fetches, feed_keys_dict

	def _getTrainBatch(self, data_src, params):
		raise NotImplementedError

	def _getValidateBatch(self, data_src, params):
		raise NotImplementedError

	def _getTestBatch(self, data_src, params):
		raise NotImplementedError

	# core training func, it will create project directories, running training loops, validation steps, testing steps;
	# and it will periodically evaluate and save network parameters into checkpoints.
	## <data_src> instance generating input_feature and input_label, including trainset, evalset, testset
	##            it will be used by _getTrainBatch, _getValidateBatch, _getTestBatch,
	##            which should be implemented in child solver class
	## <train_fetches> list of tf nodes computed during training_step: [hSummary, evaluation_step, loss, train_step, increaseStepGlobal]
	## <eval_fetches> list of tf nodes computed during validation_step: [hSummary, evaluation_step, mConfuse]
	## <test_fetches> list of tf nodes computed during test_step: [evaluation_step, mConfuse]
	## <feed_keys_dict> dictionary of tf placeholders: input_feature, input_label, learning_rate, dropout_prob
	## <training_steps_list> list of step_number during training_step
	## <learning_rates_list> list of learning_rate during training_step, with same order of that in <training_steps_list>
	## <start_step> int number, the start step index that training starts, helpful when loading a pre-trained model and continue training
	## <period_validate> step interval for validation_step
	## <period_save> step interval to save the network parameters into checkpoint files
	## <params_train_feed> dictionary of parameters configure the feed_list in training_step, including 
	## <params_eval_feed> dictionary of parameters configure the feed_list in validation_step, including 
	## <bsize> batch size during training_step
	## <evalset_size> validation dataset size
	## <testset_size> test dataset size
	def _solve(self, data_src, train_fetches, eval_fetches, test_fetches, feed_keys_dict, training_steps_list, learning_rates_list, start_step, period_validate, period_save, params_train_feed, params_eval_feed, bsize, evalset_size, testset_size):
		tf.logging.debug('>>>> start _solve')
		# check if directories exist
		for sub in [self.dTrain, self.dLog]:
			if not os.path.exists(sub):
				os.makedirs(sub)
		if not os.path.exists(self.dInput):
			tf.logging.info('Error: Failed to find input dataset.')
			raise FileNotFoundError
		# check if size match: training_steps_list v.s. learning_rates_list
		n_train_steps = len(training_steps_list)
		n_learning_rate = len(learning_rates_list)
		if n_train_steps != n_learning_rate:
			raise Exception('training_step_list and learning_rates_list should be with same size, but their current lengths = %d, %d' % (n_train_steps, n_learning_rate))
		# initial status
		tf.global_variables_initializer().run()
		tf.logging.info('Solve from step-%d ', start_step)
		# Save graph.pbtxt.
		tf.train.write_graph(self.sess.graph_def, self.dTrain, self.net_type + '.pbtxt')
		# Training loop.
		tf.logging.debug('\t>> entering training loop in _solve')
		mConfusTotal = None
		maxTrainStep = np.sum(training_steps_list)
		for iStep in range(start_step, maxTrainStep + 1):
			# switch learning rate
			nStepTmp = 0
			currLearningRate = -1
			for i in range(len(training_steps_list)):
				nStepTmp += training_steps_list[i]
				if iStep <= nStepTmp:
					currLearningRate = learning_rates_list[i]
					break
			# get data batch for training
			feed_X, feed_Y = self._getTrainBatch(data_src, params_train_feed)
			print('feed_X')
			print(feed_X.shape)
			# The core of training step
			## sess.run will return a value with same type of fetches
			## in this case, <return> is a list, in the same order of fetch list:
			## feed_dict is a dictionary of input parameter
			## data for corresponding placeholders should be included
			sumTrain, accuracyTrain, lossTrain, _, _ = self.sess.run(train_fetches,
				feed_dict={
					feed_keys_dict['input_feature'] : feed_X, # dataset
					feed_keys_dict['input_label'] : feed_Y, # ground truth
					feed_keys_dict['learning_rate'] : currLearningRate, # lr
					feed_keys_dict['dropout_prob']: 0.5
				}
			)
			# write into summary writer
			self.hLogTrain.add_summary(sumTrain, iStep)
			# display summary data of current step
			tf.logging.info('Step-%d: lr = %f, accuracy = %.1f%%, loss = %f' % (iStep, currLearningRate, accuracyTrain * 100, lossTrain))
			# evaluate model at each checkpoint
			isLastStep = (iStep == maxTrainStep)
			if (iStep % period_validate) == 0 or isLastStep:
				# get size of validation dataset
				accuracyTotal = 0
				total_conf_matrix = None
				for i in range(0, evalset_size, bsize):
					params_eval_feed['idx'] = i
					feed_Xval, feed_Yval = self._getValidateBatch(data_src, params_eval_feed)
					# Run a validation step and capture training summaries for TensorBoard
					# with the `merged` op.
					sumEval, AccuracyEval, mConfus = self.sess.run(
						eval_fetches,
						feed_dict={
							feed_keys_dict['input_feature'] : feed_Xval,
							feed_keys_dict['input_label'] : feed_Yval,
							feed_keys_dict['dropout_prob']: 1.0
						}
					)
					# write into summary
					self.hLogEval.add_summary(sumEval, iStep)
					# compute batch size
					batch_size = min(bsize, evalset_size - i)
					# compute total accuracy
					accuracyTotal += (AccuracyEval * batch_size) / evalset_size
					# compute confusion matrix
					if mConfusTotal is None:
						mConfusTotal = mConfus
					else:
						mConfusTotal += mConfus
				# print validation confusion matrix
				tf.logging.info('Confusion Matrix:\n %s' % (mConfusTotal))
				# print validation accuracy
				tf.logging.info('Step-%d: Validation accuracy = %.1f%% (dataset_size=%d)' % (iStep, accuracyTotal * 100, evalset_size))
			# Save the net parameters periodically.
			if (iStep % period_save == 0 or iStep == maxTrainStep):
				dCheckpoint = os.path.join(self.dTrain, self.net_type + '.ckpt')
				tf.logging.info('Routine auto-save to "%s-%d"', dCheckpoint, iStep)
				# write variables to file named 'dCheckpoint/conv.ckpt-train_step.xx'
				self.saver.save(self.sess, dCheckpoint, global_step=iStep)

		# testing stage
		tf.logging.info('testset_size = %d', testset_size)
		accuracyTotal = 0
		mConfusTotal = None
		# compute accuracy for every batch
		for i in range(0, testset_size, bsize):
			# get test dataset
			params_train_feed['idx'] = i
			feed_Xtest, feed_Ytest = self._getTestBatch(data_src, params_train_feed)
			# compute
			accuracyTest, mConfus = self.sess.run(
				test_fetches,
				feed_dict={
					feed_keys_dict['input_feature'] : feed_Xtest,
					feed_keys_dict['input_label'] : feed_Ytest,
					feed_keys_dict['dropout_prob']: 1.0
				})
			batch_size = min(bsize, testset_size - i)
			accuracyTotal += (accuracyTest * batch_size) / testset_size
			if mConfusTotal is None:
				mConfusTotal = mConfus
			else:
				mConfusTotal += mConfus
		tf.logging.info('Confusion Matrix:\n %s' % (mConfusTotal))
		tf.logging.info('Test accuracy = %.1f%% (N=%d)' % (accuracyTotal * 100, testset_size))
		
		tf.logging.debug('<<<< end _solve')

	def _create_model(self, data_input, model_settings, net_type, is_training):
		"""Builds a model of the requested architecture compatible with the settings.
		Args:
			fingerprint_input: TensorFlow node that will output tracing coordinates vectors.
			model_settings: structure of information about the model.
			net_type: String specifying which kind of model to create.
			is_training: Whether the model is going to be used for training.
		Returns:
			TensorFlow node outputting logits results, and optionally a dropout
			placeholder.
		Raises:
			Exception: If the model type isn't recognized.
		"""
		if net_type == 'conv':
			return self._create_conv_model(data_input, model_settings, is_training)
		elif net_type == 'rcnn':
			return self._create_rcnn_model(data_input, model_settings, is_training)
		else:
			raise Exception('net_type "' + net_type + '" not supported; currently supported types: ["conv"]')

	def _create_conv_model(self, data_input, model_settings, is_training):
		"""
		Here's the layout of the graph:

			(data_input)
				v
			[Conv2D]<-(weights)
				v
			[BiasAdd]<-(bias)
				v
				[Relu]
				v
			[MaxPool]
				v
			[Conv2D]<-(weights)
				v
			[BiasAdd]<-(bias)
				v
				[Relu]
				v
			[MaxPool]
				v
			[MatMul]<-(weights)
				v
			[BiasAdd]<-(bias)
				v
		"""
		tfStackInput = data_input
		shapeInput = [model_settings.win_size, int(model_settings.feature_num / model_settings.win_size)]
		print('shapeInput')
		print(shapeInput)
		lLayerParams = [
			{'name':'first','kernel_width':4,'kernel_height':3,'kernel_number':64},
			{'name':'second','kernel_width':4,'kernel_height':3,'kernel_number':64},
			{'name':'final_fc', 'nLabel':model_settings.label_count}
		]
		isTrain = is_training
		return MNet.createConvNet(tfStackInput, shapeInput, lLayerParams, isTrain)

	def _create_rcnn_model(self, data_input, model_settings, is_training):
		"""
		Here's the layout of the graph:

			(data_input)
				v
			[Conv2D]<-(weights)
				v
			[BiasAdd]<-(bias)
				v
				[Relu]
				v
			[MaxPool]
				v
			[Conv2D]<-(weights)
				v
			[BiasAdd]<-(bias)
				v
				[Relu]
				v
			[MaxPool]
				v
			[MatMul]<-(weights)
				v
			[BiasAdd]<-(bias)
				v
		"""
		tfStackInput = data_input
		shapeInput = [model_settings.win_size, int(model_settings.feature_num / model_settings.win_size)]
		print('shapeInput')
		print(shapeInput)
		lLayerParams = [
			{'name':'first','kernel_width':4,'kernel_height':3,'kernel_number':64},
			{'name':'second','kernel_width':4,'kernel_height':3,'kernel_number':64},
			{'name':'final_fc', 'nLabel':model_settings.label_count}
		]
		isTrain = is_training
		return MNet.createConvNet(tfStackInput, shapeInput, lLayerParams, isTrain)

import input_tracing_data as input_data

class ScratchSolverParams:
	def __init__(self):
		self.batch_size = 100
		self.eval_interv = 100
		self.save_interv = 250
		self.start_checkpoint = False
		self.dataset_name = 'mouse_scratch'
		self.net_type = 'conv'
		self.divide_strategy = 'simple'
		self.training_steps = '5000, 3000'
		self.learning_rate = '0.001, 0.0001'
		self.feature_dict = {'nose':0, 'face1':1, 'face2':2, 'hand1':3, 'hand2':4, 'foot1':5, 'foot2':6, 'tailroot':7}
		self.feature_select = ['nose', 'hand1', 'hand2', 'foot1', 'foot2', 'tailroot']
		self.feature_sel_idx_list = [self.feature_dict[name] for name in self.feature_select]
		self.win_size = 5
		self.feature_num = len(self.feature_select) * 2 * self.win_size
		self.label_count = 2

class ScratchSolver(MSolver):
	def __init__(self, root_path):
		# base class init
		MSolver.__init__(self, root_path)
		
		tf.logging.debug('>>>> start SoundSolver.init')
		# init solver parameters with default
		tf.logging.debug('\t>>prepare network config in SoundSolver.init')
		self.params = ScratchSolverParams()
		params = self.params
		tf.logging.info('\tselected feature name list:')
		tf.logging.info(params.feature_select)
		# prepare data source
		root_path = self.root_path
		data_path = self.dInput
		dataset_name = params.dataset_name
		feature_sel_idx_list = params.feature_sel_idx_list
		win_size = params.win_size
		strategy = params.divide_strategy
		tf.logging.debug('\t>>prepare data source ScratchSolver')
		self.data_src = input_data.createDataset(data_path, dataset_name, feature_sel_idx_list,win_size=win_size, divide_strategy = strategy)
		# set input parameters
		## feature size and label size
		feature_num = params.feature_num
		label_count = params.label_count
		## set learning rate and training steps
		training_steps_list = list(map(int, params.training_steps.split(',')))
		learning_rates_list = list(map(float, params.learning_rate.split(',')))
		# create solver net
		tf.logging.debug('\t>> create solver net in ScratchSolver')
		train_fetches, eval_fetches, test_fetches, feed_keys_dict = self._createSolverNet(params, params.net_type, feature_num, label_count)
		# solve network
		tf.logging.debug('\t>>set start_step in ScratchSolver.init')
		start_step = 1 # start from beginning
		if params.start_checkpoint: # from checkpoint
			load_variables_from_checkpoint(self.sess, params.start_checkpoint)
			start_step = self.stepGlobal.eval(session=self.sess)
		## set periods to validate/save
		period_validate = params.eval_interv
		period_save = params.save_interv
		## set batch_sizes
		bsize = params.batch_size
		evalset_size = self.data_src.evalset.shape[0]
		testset_size = self.data_src.testset.shape[0]
		## set batch parameters
		tf.logging.debug('\t>> set batch parameters in ScratchSolver')
		params_train_feed = {
			'bsize' : bsize,
			'cfgNet': params
		}
		params_eval_feed = {
			'bsize' : bsize,
			'cfgNet': params
		}
		## start training process
		self._solve(self.data_src, train_fetches, eval_fetches, test_fetches, feed_keys_dict, training_steps_list, learning_rates_list, start_step, period_validate, period_save, params_train_feed, params_eval_feed, bsize, evalset_size, testset_size)
	
	def _getTrainBatch(self, data_src, params):
		return data_src.getMiniBatch('trainset', params['bsize'])
	
	def _getValidateBatch(self, data_src, params):
		return data_src.getMiniBatch('evalset', params['bsize'])
	
	def _getTestBatch(self, data_src, params):
		return data_src.getMiniBatch('testset', params['bsize'])

def main():
	# version: 20190108 1855
	root_path = '/home/youngway/data/mouse_scratch'
	solver = ScratchSolver(root_path)

if __name__ == '__main__':
	main()