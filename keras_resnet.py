import numpy as np
import cv2
import keras
import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import add, Input
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

def create_shape(y, map_w=48, map_h=48, shape_w=8, shape_h=8):
	img = np.zeros((map_h, map_w, 3))
	ptx = np.random.randint(map_w - shape_w)
	pty = np.random.randint(map_h - shape_h)
	d_w = np.random.randint((shape_w - 1) * 2) - shape_w
	d_h = np.random.randint((shape_h - 1) * 2) - shape_h
	clr = (1,1,1)
	stype = np.argmax(y)
	if stype == 0:
		cv2.rectangle(img, (ptx, pty, shape_w + d_w, shape_h + d_h), clr)
	elif stype == 1:
		cv2.circle(img, (ptx, pty), int((shape_w + d_w) / 2), clr)
	elif stype == 2:
		pt0 = (ptx + int((shape_w + d_w) / 2), pty)
		pt1 = (ptx + shape_w + d_w, int((pty + shape_h + d_h)/ 2))
		pt2 = (ptx, pty + shape_h + d_h)
		cv2.line(img, pt0, pt1, clr)
		cv2.line(img, pt0, pt2, clr)
		cv2.line(img, pt1, pt2, clr)
	return img.astype(float)

def create_dataset(N, shape_img):
	"""
	aim: [rectangle, circle, triangle]
	"""
	# generate raw dataset
	y_all = []
	for i in range(N):
		tmp = np.zeros(3)
		idx = np.random.randint(3)
		tmp[idx] = 1
		y_all.append(tmp)
	y_all = np.array(y_all)
	x_all = np.array([create_shape(y, shape_img[0], shape_img[1]) for y in y_all])
	# divide dataset
	ratio = 0.8
	cutoff = int(ratio * N)
	x_train, y_train = x_all[:cutoff], y_all[:cutoff]
	x_test, y_test = x_all[cutoff:], y_all[cutoff:]
	return x_train, y_train, x_test, y_test

def identity_block(x, filter_num, stride, chn_order, is_reduce=False):
	epsilon = 2e-5
	momentum = 0.9 
	reg_factor = 1e-4
	data_format = 'channels_last'
	
	residue = BatchNormalization(axis=chn_order, epsilon=epsilon, momentum=momentum)(x)
	residue = Conv2D(filter_num, (3,3), activation='relu', strides=stride, padding='same', data_format=data_format, use_bias=False, kernel_regularizer=l2(reg_factor))(residue)
	
	residue = BatchNormalization(axis=chn_order, epsilon=epsilon, momentum=momentum)(residue)
	residue = Conv2D(filter_num, (3,3), activation='relu', strides=stride, padding='same', data_format=data_format, use_bias=False, kernel_regularizer=l2(reg_factor))(residue)
	
	# compute identity
	identity = x # need dimension expansion
	if is_reduce:
		identity = Conv2D(filter_num, (1,1), activation='relu', strides=stride, padding='valid', data_format=data_format, use_bias=False, kernel_regularizer=l2(reg_factor))(x)
	return add([identity, residue])

def bottleneck_block(x, filter_num, stride, chn_order, is_reduce=False):
	epsilon = 2e-5
	momentum = 0.9 
	reg_factor = 1e-4
	data_format = 'channels_last'
	
	residue = BatchNormalization(axis=chn_order, epsilon=epsilon, momentum=momentum)(x)
	residue = Conv2D(filter_num, (1,1), activation='relu', strides=stride, padding='valid', data_format=data_format, use_bias=False, kernel_regularizer=l2(reg_factor))(residue)
	
	residue = BatchNormalization(axis=chn_order, epsilon=epsilon, momentum=momentum)(x)
	residue = Conv2D(filter_num, (3,3), activation='relu', strides=stride, padding='same', data_format=data_format, use_bias=False, kernel_regularizer=l2(reg_factor))(residue)
	
	residue = BatchNormalization(axis=chn_order, epsilon=epsilon, momentum=momentum)(residue)
	residue = Conv2D(filter_num * 4, (1,1), activation='relu', strides=stride, padding='valid', data_format=data_format, use_bias=False, kernel_regularizer=l2(reg_factor))(residue)
	
	# compute identity
	identity = x # need dimension expansion
	if is_reduce:
		identity = Conv2D(filter_num * 4, (1,1), activation='relu', strides=stride, padding='valid', data_format=data_format, use_bias=False, kernel_regularizer=l2(reg_factor))(x)
	return add([identity, residue])

def resnet(name, input_dim, output_dim, block_func, stage_list, filter_num_list):
	# internal parameters
	epsilon = 2e-5
	momentum = 0.9
	reg_factor = 1e-4
	chn_order = -1 # channel-last mode in tensorflow, or use 1 in theano
	stage_num = len(stage_list)
	
	# input stage
	input = Input(shape=input_dim)
	x = Conv2D(64, 7, strides=2, padding='same', activation='relu', data_format='channels_last', use_bias=False, kernel_regularizer=l2(reg_factor))(input)
	x = BatchNormalization(axis=chn_order, epsilon=epsilon, momentum=momentum)(x)
	x = MaxPooling2D((3,3), strides=2, padding='same')(x)
	
	# shortcut stages
	for i in range(stage_num):
		stride = 1
		x = block_func(x, filter_num_list[i], stride=stride, chn_order=chn_order, is_reduce=True)
		for j in range(stage_list[i]-1):
			x = block_func(x, filter_num_list[i], stride=stride, chn_order=chn_order)
	
	# finish stage
	x = BatchNormalization(axis=chn_order, epsilon=epsilon, momentum=momentum)(x)
	x = Activation('relu')(x)
	x = AveragePooling2D((8,8))(x)
	
	# fcl stage
	x = Flatten()(x)
	x = Dense(output_dim, kernel_regularizer=l2(reg_factor))(x)
	x = Activation('softmax')(x)
	
	# network compilation
	net = Model(inputs=input, outputs=x, name=name)
	#sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
	net.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
	return net

def create_net(net_type, input_dim, output_dim, learning_rate=0.01):
	net = None
	if net_type == 'fcl':
		net = Sequential()
		net.add(Dense(64, input_dim=input_dim, activation='relu'))
		net.add(Dropout(0.5))
		net.add(Dense(32, activation='relu'))
		net.add(Dropout(0.5))
		net.add(Dense(output_dim, activation='softmax'))
		sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
		net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	elif net_type == 'vgg':
		net = Sequential()
		net.add(Conv2D(32,(3,3), activation='relu', input_shape=input_dim))
		net.add(Conv2D(32,(3,3), activation='relu'))
		net.add(MaxPooling2D(pool_size=(2,2)))
		net.add(Dropout(0.25))
		
		net.add(Conv2D(64,(3,3), activation='relu'))
		net.add(Conv2D(64,(3,3), activation='relu'))
		net.add(MaxPooling2D(pool_size=(2,2)))
		net.add(Dropout(0.25))
		
		net.add(Flatten())
		net.add(Dense(256, activation='relu'))
		net.add(Dropout(0.5))
		net.add(Dense(output_dim, activation='softmax'))
		
		sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
		net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	elif net_type == 'resnet18':
		block_func = identity_block
		stage_list = [2,2,2,2]
		filter_num_list = [64,128,256,512]
		net = resnet(net_type, input_dim, output_dim, block_func, stage_list, filter_num_list)
	elif net_type == 'resnet50':
		block_func = bottleneck_block
		stage_list = [3,4,6,3]
		filter_num_list = [64,128,256,512]
		net = resnet(net_type, input_dim, output_dim, block_func, stage_list, filter_num_list)
	else:
		print('unsupported net_type ' + net_type)
		
	if net != None:
		net.summary()
		plot_model(net, to_file='network.png', show_shapes=True)
	return net

def preview(net, x_train, y_train, sample_num=100):
	stype_lookup = ['rectangle', 'circle', 'triangle']
	sel_idx_list = [np.random.randint(len(y_train)-1) for i in range(sample_num)]
	y_pred = net.predict(x_train)
	for idx in sel_idx_list:
		img = x_train[idx]*255
		stype_s = stype_lookup[np.argmax(y_train[idx])]
		stype_pred_s = stype_lookup[np.argmax(y_pred[idx])]
		name = '%s_%d-predict_%s_%.2f.png' % (stype_s, idx, stype_pred_s, np.max(y_pred))
		cv2.imwrite(name, img)

def train():
	N = 1000
	input_dim = (48,48,3)
	output_dim = 3
	epochs = 200
	bsize = 128
	lr = 0.002
	print('generating dataset')
	x_train, y_train, x_test, y_test = create_dataset(N, input_dim)
	print('building network')
	net = create_net('resnet50', input_dim, output_dim, learning_rate = lr)
	if net != None:
		print('solving network')
		print(x_train.shape)
		print(y_train.shape)
		net.fit(x_train, y_train, epochs=epochs, batch_size = bsize)
		print('evaluating network')
		score = net.evaluate(x_test, y_test, batch_size = bsize)
		print('test score = ', end='')
		print(score)
		preview(net, x_train, y_train, sample_num=100)

if __name__ == '__main__':
	train()