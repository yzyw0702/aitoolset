import tensorflow as tf
from MConvLayer import MConvUnit, MFcUnit


# factory function: create a tf-style multiple convolution2d network with one last fully-connected layer
# [Args] tfStackInput: a placeholder for input data stack
# [Args] shapeInput: dimentions of input stack
# [Args] nLabel: dimentions of label data
# [Args] lLayerParams: dictionary list of layer parameters,
#      all for conv2d layers and the last one for fully-connected layer:
#      [{'name'='conv1','kernel_width'=8,'kernel_height'=24,'kernel_number'=5},
#      {'name'='conv2',......}
#      {'name'='fc', 'nLabel'=10}
def createConvNet(tfStackInput, shapeInput, lLayerParams, isTrain):
	probDropout = None
	if isTrain:
		probDropout = tf.placeholder(tf.float32, name='dropout_prob')
	hInput = shapeInput[0]
	wInput = shapeInput[1]
	
	# build convolution network
	layerData = tf.reshape(tfStackInput, [-1, hInput, wInput, 1])
	inputL = layerData
	outputL = None
	paramsLastConv = None
	prevNK=1
	for iL, paramsL in enumerate(lLayerParams):
		# collect layer-unit parameters
		# create current layer-unit
		# prepare input interface for next layer
		if iL < len(lLayerParams) - 2:
			nameL = paramsL['name']
			wK = paramsL['kernel_width']
			hK = paramsL['kernel_height']
			nK = paramsL['kernel_number']
			print('L-%d\t%dx%dx%dx%d' % (iL, wK, hK, prevNK, nK))
			layerunit = MConvUnit(wK, hK, prevNK, nK, layer_name=nameL)
			outputL = layerunit.flowFrom(inputL, isTrain, isPool=True, probDropout=probDropout)
			inputL = outputL
			prevNK = nK
		elif iL == len(lLayerParams) - 2:
			nameL = paramsL['name']
			wK = paramsL['kernel_width']
			hK = paramsL['kernel_height']
			nK = paramsL['kernel_number']
			layerunit = MConvUnit(wK, hK, prevNK, nK, layer_name=nameL)
			outputL = layerunit.flowFrom(inputL, isTrain, isPool=False, probDropout=probDropout)
			shapeOutput = outputL.get_shape()
			nLastConv = shapeOutput[1] * shapeOutput[2] * layerunit.nK
			inputL = tf.reshape(outputL,shape=[-1, nLastConv])
			prevNK = nK
		else:
			break
	
	# attach to fully-connected network
	paramsFcl = lLayerParams[-1]
	nameFcl = paramsFcl['name']
	nX = nLastConv
	nY = paramsFcl['nLabel']
	layerunit = MFcUnit(nX, nY, nameFcl)
	outputL = layerunit.flowFrom(inputL)
	
	# return the network
	if isTrain:
		return outputL, probDropout
	else:
		return outputL
