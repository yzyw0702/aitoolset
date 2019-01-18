from __future__ import print_function
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import copy

class TwoLabelDataSrc:
	def __init__(self, name, feature_sel_list, dataset, pos_tm_table, divide_strategy = 'simple'):
		# init class-global parameters
		col_sel_list = [0]
		[col_sel_list.extend([idx*3+1, idx*3+2, idx*3+3]) for idx in feature_sel_list]
		self.name = name
		self.dataset = dataset[:,:,col_sel_list]
		self.clip_len = len(self.dataset)
		np.random.seed(6)
		
		# time table -> time list
		self.pos_tm_list = []
		for tm in self._getTmIdxTable(pos_tm_table):
			self.pos_tm_list.extend(tm)
		self.all_tm_list = range(self.clip_len)
		self.neg_tm_list = list(set(self.all_tm_list).difference(set(self.pos_tm_list)))
		
		# dataset -> pos/neg_dataset
		self.pos_dataset_raw = self.dataset[self.pos_tm_list]
		self.pos_dataset = self._cleanAndNormalize(self.pos_dataset_raw)
		pos_mir = self._mirrorEnhance(self.pos_dataset)
		self.pos_dataset = np.concatenate([self.pos_dataset, pos_mir], axis=0)
		self.neg_dataset_raw = self.dataset[self.neg_tm_list]
		self.neg_dataset = self._cleanAndNormalize(self.neg_dataset_raw)
		neg_mir = self._mirrorEnhance(self.neg_dataset)
		self.neg_dataset = np.concatenate([self.neg_dataset, neg_mir], axis=0)
		
		# init 3 set and init allocation strategy
		self.trainset = None
		self.evalset = None
		self.testset = None
		self.trainlabel = None
		self.evallabel = None
		self.testlabel = None
		self.strategy = divide_strategy
		self._divideDataset()
	
	def _getTmIdxTable(self, time_table):
		"""
		[                    [
		 [start stop]         [start, start+1, ..., stop]
		 [start stop]  ===>   [start, start+1, ..., stop]
		 [start stop]         [start, start+1, ..., stop]
		]                    ]
		"""
		tm_idx_table = []
		for rg in time_table:
			tm_idx_table.append(range(rg[0], rg[1]))
		return tm_idx_table
	
	def getMiniBatch(self, type, batch_size):
		if type == 'trainset':
			return self._getBatch(self.trainset, self.trainlabel, batch_size)
		elif type == 'evalset':
			return self._getBatch(self.evalset, self.evallabel, batch_size)
		elif type == 'testset':
			return self._getBatch(self.testset, self.testlabel, batch_size)
		else:
			print('currently supported type: trainset, evalset, testset')
	
	def _getBatch(self, dataset, datalabel, batch_size):
		idx_list = []
		for v in range(dataset.shape[2]):
			if (v!=0 and v%3!=0):
				idx_list.append(v)
		start = int(np.random.random() * (len(dataset) - batch_size))
		win_size = dataset.shape[1]
		coor_sel_num = len(idx_list)
		return dataset[start:(start + batch_size),:,idx_list].reshape((batch_size, win_size*coor_sel_num)), datalabel[start:(start + batch_size)]
	
	def _rotAndScaled(self, pt_vec):
		"""
			rotate and rescale all points in vector pt_vec
			[pt_vec.format] idx, x1, y1, p1, ...
			where pt[0] == nose, pt[-1] == tail
			[return.pt_rotscal_vec] idx, x1', y1', p1, ... 
		"""
		pt_num = len(pt_vec)
		nose = pt_vec[1:3].reshape((2,1))
		tail = pt_vec[-3:-1].reshape((2,1))
		# compute rotation matrix
		R = None
		if nose[1] != 0:
			theta = np.arctan(nose[0] / nose[1])
		elif nose[0] < 0:
			theta = 1.5 * np.pi
		elif nose[0] > 0:
			theta = 0.5 * np.pi
		else:
			theta = 0
			print('[warning] nose coord == origin')
		R = np.array([
			[math.cos(theta), -math.sin(theta)],
			[math.sin(theta),  math.cos(theta)]
		])
		coord_vec = np.array([pt_vec[1::3], pt_vec[2::3]])
		coord_rot_vec = np.matmul(R, coord_vec)
		# compute scale vector
		k = coord_rot_vec[1, 0]
		coord_scaled_vec = coord_rot_vec / k
		# reshape the vector
		pt_rotscal_vec = [pt_vec[0]]
		prob_vec = pt_vec[3::3]
		for (xy, p) in zip(coord_scaled_vec.T, prob_vec):
			pt_rotscal_vec.extend(xy)
			pt_rotscal_vec.append(p)
		return pt_rotscal_vec
	
	def _cleanAndNormalize(self, dataset, valid_prob = 0.9):
		"""
			return subset with all probabilities > valid_prob
			[dataset.format] idx, x1, y1, p1, x2, y2, p2...
			where: (x1,y1) = nose and (x2,y2) = tailroot
			[return.normalized_valid_dataset]
		"""
		# remove invalid subset
		data_set = np.array(dataset)
		prob_set = data_set[:, :, 3::3]
		fr_num, win_size, pt_num = prob_set.shape
		valid_tresh = win_size * pt_num
		isvalid_list = [(mat>valid_prob).sum() == valid_tresh for mat in prob_set]
		valid_idx_list = np.argwhere(isvalid_list)
		valid_idx_list = valid_idx_list.reshape(len(valid_idx_list))
		valid_data_set = data_set[valid_idx_list]
		# set the center of body axis as origin
		print(valid_data_set.shape)
		fr_num_valid = valid_data_set.shape[0]
		X = valid_data_set[:, :, 1::3]
		Y = valid_data_set[:, :, 2::3]
		X0 = X[:,:,[0,-1]].mean(axis=2, keepdims=True)
		Y0 = Y[:,:,[0,-1]].mean(axis=2, keepdims=True)
		X_norm = (X - X0) / fr_num_valid
		Y_norm = (Y - Y0) / fr_num_valid
		# reconstruct the data stack
		idx_set_valid = valid_data_set[:, :, 0].reshape((fr_num_valid, win_size, 1))
		prob_set_valid = valid_data_set[:, :, 3::3]
		data_set_norm = []
		for (Id_win, X_win, Y_win, p_win) in zip(idx_set_valid, X_norm, Y_norm, prob_set_valid):
			data_samp_norm = []
			for (idx, X_vec, Y_vec, p_vec) in zip(Id_win, X_win, Y_win, p_win):
				data_vec_norm = [idx[0]]
				[data_vec_norm.extend([x,y,p]) for (x,y,p) in zip(X_vec, Y_vec, p_vec)]
				data_vec_norm = self._rotAndScaled(np.array(data_vec_norm)) # rotate and rescale
				data_samp_norm.append(data_vec_norm)
			data_set_norm.append(data_samp_norm)
		# return numpy version of preprocessed data_set
		return np.array(data_set_norm)
	
	def _mirrorEnhance(self, dataset):
		"""
			get mirror copy of normalized dataset
			[dataset.format] idx, x1, y1, p1, x2, y2, p2, ...
			pt[0] = nose = (0,1), pt[-1] = tail = (0,-1)
			[return.dataset_mir] mirror copy of dataset
		"""
		dataset_mir = copy.deepcopy(dataset)
		dataset_mir[:,:,0] = -dataset[:,:,0]
		dataset_mir[:,:,4] = -dataset[:,:,4]
		dataset_mir[:,:,7] = -dataset[:,:,7]
		dataset_mir[:,:,10] = -dataset[:,:,10]
		dataset_mir[:,:,13] = -dataset[:,:,13]
		return dataset_mir
	
	def _divideDataset(self):
		if self.strategy == 'simple':
			ratio_list = np.array([0.6, 0.2, 0.2])
			pos_count = len(self.pos_dataset)
			neg_count = len(self.neg_dataset)
			subset_pos_list = ratio_list * pos_count
			subset_pos_list = subset_pos_list.astype(np.int32)
			subset_neg_list = ratio_list * neg_count
			subset_neg_list = subset_neg_list.astype(np.int32)
			lSubset = []
			prev_start = [0, 0]
			for i in range(3):
				subset = []
				subset.extend(self.pos_dataset[prev_start[0]:prev_start[0]+subset_pos_list[i]])
				subset.extend(self.neg_dataset[prev_start[1]:prev_start[1]+subset_neg_list[i]])
				lSubset.append(subset)
				prev_start[0] += subset_pos_list[i]
				prev_start[1] += subset_neg_list[i]
			self.trainset, self.evalset, self.testset = np.array(lSubset[0]), np.array(lSubset[1]), np.array(lSubset[2])
			lSublabel = []
			for i in range(3):
				sublabel = list(np.zeros(subset_pos_list[i]) + 1)
				sublabel.extend(list(np.zeros(subset_neg_list[i])))
				lSublabel.append(sublabel)
			self.trainlabel, self.evallabel, self.testlabel = np.array(lSublabel[0]), np.array(lSublabel[1]), np.array(lSublabel[2])
		else:
			print('[warning] invalid dataset allocation strategy.')

	def getPos2NegRatio(self):
		return float(len(self.pos_dataset)) / (len(self.neg_dataset) + len(self.pos_dataset))

	def visualize(self, out_file = 'dataset_distribution.png', dataset_type = 'all'):
		# target choices
		data_set = None
		data_set_list = []
		is_plot_all = False
		if dataset_type == 'trainset':
			data_set = self.trainset
		elif dataset_type == 'evalset':
			data_set = self.evalset
		elif dataset_type == 'testset':
			data_set = self.testset
		elif dataset_type == 'pos_dataset':
			data_set = self.pos_dataset
		elif dataset_type == 'neg_dataset':
			data_set = self.neg_dataset
		elif dataset_type == 'all':
			is_plot_all = True
			data_set_list = [self.pos_dataset, self.neg_dataset]
		else:
			print('[warning] No currently only support: trainset, evalset, testset, pos_dataset, neg_dataset')
			return
		# draw on pyplot
		if is_plot_all:
			name_list = ['positive_dataset', 'negative_dataset']
			fig = plt.figure(1)
			for i, subset in enumerate(data_set_list):
				fr_num, win_num, feature_num = subset.shape
				sample_num = fr_num * win_num
				pt_num = (feature_num - 1) / 3
				coord_mat = subset.reshape((sample_num, feature_num))[:, 1:]
				X_mat = coord_mat[:sample_num/2, 0::3].T
				Y_mat = coord_mat[:sample_num/2, 1::3].T
				color = ['r', 'y', 'orange', 'g', 'cyan', 'k', 'blue', 'brown', 'orange', 'grey', 'cyan', '#2ca02c']
				iPt = 0
				#fig = plt.figure(name_list[i])
				ax = plt.subplot(1,2,i+1)
				ax.set_title(name_list[i])
				for (X_vec, Y_vec) in zip(X_mat, Y_mat):
					if iPt == 0:
						plt.plot([0, 0], [1, -1], 'k-', linewidth=2)
					clr = color[iPt]
					plt.scatter(X_vec, Y_vec, s=2, color=clr)
					iPt += 1
			plt.savefig(out_file)
		else:
			fr_num, win_num, feature_num = data_set.shape
			sample_num = fr_num * win_num
			pt_num = (feature_num - 1) / 3
			coord_mat = data_set.reshape((sample_num, feature_num))[:, 1:]
			X_mat = coord_mat[:sample_num/2, 0::3].T
			Y_mat = coord_mat[:sample_num/2, 1::3].T
			color = ['r', 'y', 'k', 'g', 'm', 'purple', 'blue', 'brown', 'orange', 'grey', 'cyan', '#2ca02c']
			iPt = 0
			plt.figure()
			for (X_vec, Y_vec) in zip(X_mat, Y_mat):
				clr = color[iPt]
				plt.scatter(X_vec, Y_vec, s=2, color=clr)
				iPt += 1
			plt.savefig(out_file)

def getTraceData(fInput):
	"""
	<return.scorer>
	<return.type_list>
	<return.data_mat> fr_idx, x, y, prob, x, y, prob ...
	"""
	scorer = None
	type_list = []
	data_mat = []
	with open(fInput, 'r') as hIn:
		for i, line in enumerate(hIn.readlines()):
			sLine = line.rstrip(', \r\n')
			if i == 0: # parse scorer name
				scorer = sLine.split(',')[1]
				continue
			elif i == 1: # parse bodypart type list
				type_list = sLine.split(',')[1::3]
				continue
			elif i == 2 or len(sLine) < 1: # omit notes or blank line
				continue
			data_mat.append([float(v) for v in sLine.split(',')])
		data_mat = np.array(data_mat)
	return scorer, type_list, data_mat

def getFrIdx(minsec_s, fps):
	tm_min, tm_sec = minsec_s.rstrip('sec').split('min-')
	return int((int(tm_min) * 60 + int(tm_sec)) * fps)

def getEventTableDict(fInput, win_size):
	"""
		<Return> event_table_dict: (key: clip file name; value: a tuple) value_dims = (fps, time_table)
		<Return> time_table: (np-mat) dims = [event_number x 2] (which provides start and stop time for each event)
		
	"""
	event_table_dict = {}
	with open(fInput, 'r') as hIn:
		for i, line in enumerate(hIn.readlines()):
			sLine = line.rstrip(', \t\r\n').lstrip()
			if len(sLine) < 1 or sLine[0] == '#':
				continue # omit notes or blank line
			clip_file, fps, time_list = sLine.split('\t')
			fps = float(fps)
			time_table = []
			for time_range in time_list.split('; '):
				begin, end = time_range.split(', ')
				begin_tm = getFrIdx(begin, fps) + int(fps)
				end_tm = getFrIdx(end, fps)-int(fps) - win_size
				if begin_tm > end_tm:
					print('\tomit event %s in %s' % (time_range, clip_file))
					continue
				time_table.append([begin_tm, end_tm])
			time_table = np.array(time_table)
			event_table_dict[clip_file] = (fps, time_table)
	return event_table_dict

def getFeatureSpectrum(mat, win_size):
	spectrum = []
	samp_num = len(mat) - win_size
	for i in range(samp_num):
		spectrum.append(mat[i:i+win_size])
	return spectrum

def createDataset(root_path, name, feature_sel_list = None, win_size = 5, divide_strategy = 'simple'):
	event_list_path = os.path.join(root_path, 'event_list.txt')
	event_table_dict = getEventTableDict(event_list_path, win_size)
	prev_stop_tm = 0
	all_pos_tm_table = []
	all_dataset = []
	for k in event_table_dict.keys():
		fps, time_table = event_table_dict[k]
		time_table += prev_stop_tm
		all_pos_tm_table.extend(list(time_table))
		trace_file = k.rstrip('.avi') + '.csv'
		trace_path = os.path.join(root_path, trace_file)
		scorer, type_list, data_mat = getTraceData(trace_path)
		data_mat[:, 0] += prev_stop_tm
		data_spectrum = getFeatureSpectrum(data_mat, win_size)
		all_dataset.extend(data_spectrum)
		prev_stop_tm = len(data_mat)
	all_pos_tm_table = np.array(all_pos_tm_table)
	all_dataset = np.array(all_dataset)
	return TwoLabelDataSrc(name, feature_sel_list, all_dataset, all_pos_tm_table, divide_strategy)

######## uint test func ########
def printDims(name, np_arr):
	print('np.array <name = %s> dims = ' % name, end='')
	print(np_arr.shape)

def printList(l, prefix='', col_num=5):
	for i, v in enumerate(l):
		print(prefix, end='')
		print(v, end='')
		print(', ', end='')
		if (i+1) % col_num == 0:
			print('')

def printDict(name, dt):
	print('###### <start>dict (%s)' % name)
	for k,v in dt.items():
		print('#### %s -> ' % k, end='')
		if isinstance(v, list) or isinstance(v, tuple):
			print('')
			printList(v, prefix='', col_num=1)
		else:
			print(v)
	print('###### <stop>dict (%s)' % name)

def test_getTraceData():
	fInput = '20170829-KI 28 after drug 1.csv'
	scorer, type_list, data_mat = getTraceData(fInput)
	print('scorer = ' + scorer)
	print(type_list)
	printDims('data_mat', data_mat)

def test_getFrIdx():
	minsec_s = '35min-17sec'
	fps = float('27.4')
	# should be 58005
	print(getFrIdx(minsec_s, fps))

def test_getEventTableDict():
	fInput = 'event_list.txt'
	event_table_dict = getEventTableDict(fInput)
	printDict('event_table_dict', event_table_dict)

def run_createDataset(root_path = '.'):
	name = 'mouse_scratching'
	strategy = 'simple'
	win_size = 5 # unit: frame
	batch_size = 10
	feature_name_list = ['nose','face1','face2','hand1','hand2','foot1','foot2','tailroot']
	feature_sel_list = [0,3,4,5,6,7] # selected col_num should be 3*6+1 = 19
	feature_name_sel_list = [feature_name_list[i] for i in feature_sel_list]
	print('selected feature name list:')
	print(feature_name_sel_list)
	data_src = createDataset(root_path, name, feature_sel_list,win_size=win_size, divide_strategy = strategy)
	print('dataset size = ', end='')
	print(data_src.dataset.shape)
	print('trainset size = ', end='')
	print(data_src.trainset.shape)
	print('evalset size = ', end='')
	print(data_src.evalset.shape)
	print('testset size = ', end='')
	print(data_src.testset.shape)
	print('pos_dataset ratio = ', end='')
	print('%.2f%%' % (data_src.getPos2NegRatio()*100))
	for i in range(4):
		X_feed, Y_feed = data_src.getMiniBatch('trainset', batch_size)
		print(X_feed[0, 0])
		print(Y_feed)
	dataset_vis_path = os.path.join(root_path, 'dataset_profile.png')
	data_src.visualize(dataset_vis_path)

def main():
	# version: 20190108 1855
	root_path = '/home/youngway/data/mouse_scratch'
	#test_getTraceData()
	#test_getFrIdx()
	#test_getEventTableDict()
	run_createDataset(root_path)
	#run_replayLabelingVideo()

if __name__ == '__main__':
	main()