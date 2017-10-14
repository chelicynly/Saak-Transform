
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from skimage.util.shape import view_as_windows
from sklearn.cluster import KMeans
import cPickle, gzip
def generate_sample(imgs, label, img_cnt, class_cnt):
    cnt, w, h, d = imgs.shape
    each_class = img_cnt / class_cnt
    sample_data = np.zeros((img_cnt, w, h, d))
    sample_label = np.zeros(img_cnt)
    taken_img = 0
    dictionary = {}
    for i in range(class_cnt):
        dictionary[i] = each_class
    for index, img in enumerate(imgs):
        if taken_img == img_cnt:
            break
        if (dictionary[label[index]] > 0):
            sample_data[taken_img] = img
            sample_label[taken_img] = label[index]
            taken_img += 1
            dictionary[label[index]] -= 1
    return sample_data, sample_label

def print_shape(train, test):
	print "train's shape is %s" % str(train.shape)
	print "test's shape is %s" % str(test.shape)

def patch_filter(sample, sample_label, patch_threshold = 10000):
	sample_var = np.std(sample, axis = 1)
	max_std = np.max(sample_var)
	print np.max(sample_var)
	sample_filter = sample[sample_var > max_std/patch_threshold]
	sample_label = sample_label[sample_var > max_std/patch_threshold]
	return sample_filter, sample_label

def KMeans_resample(sample, sample_label, k = -1, floor_val = 1000, components_cnt = 100):
	if k != -1:
		components_cnt = int(sample.shape[1]*0.5)
		# Perform kmeans clustering and patch filtering based on distribution
		kmeans = KMeans(n_clusters = k, precompute_distances = True, random_state = 10, n_jobs = -1)
		sample_kmeans_data = sample[:,:components_cnt]
		kmeans.fit(sample_kmeans_data)
		# calculate the distribution
		distribution_dict = {}
		for i in range(k):
			distribution_dict[i] = float(sum(kmeans.labels_ == i))
		min_val = min(floor_val, 0.5 * min(distribution_dict.itervalues()))
		for i in range(k):
			distribution_dict[i] = min_val / distribution_dict[i]
		print 'floor val for resample is %d' % min_val
		sample_cnt = sample.shape[0]
		random_num = np.random.randn(sample_cnt)
		boolean_take = np.zeros(sample_cnt, dtype = bool)
		cnt = 0
		for i in xrange(sample_cnt):
			if random_num[i] < distribution_dict[kmeans.labels_[i]]:
				boolean_take[i] = True
				cnt += 1
		sample = sample[boolean_take]
		sample_label = sample_label[boolean_take]
	return sample, sample_label

def window_process(sample, train, test, sample_label, train_label, test_label):
	sample_shape = sample.shape
	train_shape = train.shape
	test_shape = test.shape

	sample_cnt, train_cnt, test_cnt = sample_shape[0], train_shape[0], test_shape[0]
	w, h, d = sample_shape[1], sample_shape[2], sample_shape[3]

	sample_label = np.repeat(sample_label, (w-1) * (h-1))
	train_label = np.repeat(train_label, w/2 * h/2)
	test_label = np.repeat(test_label, w/2 * h/2)

	sample_window = view_as_windows(sample, (1,2,2,d), step = (1,1,1,d)).reshape(sample_cnt * (w-1) * (h-1), -1)
	train_window = view_as_windows(train, (1,2,2,d), step = (1,2,2,d)).reshape(train_cnt * w/2 * h/2, -1)
	test_window = view_as_windows(test, (1,2,2,d), step = (1,2,2,d)).reshape(test_cnt * w/2 * h/2, -1)

	return sample_window, train_window, test_window, sample_label, train_label, test_label

def convolution(train, test, train_label, test_label, k, stage, per):
	# generate sample data and label
	sample, sample_label = generate_sample(train, train_label, 60000, 10)
	#sample = train
	#sample_label = train_label

	sample_shape = sample.shape
	train_shape = train.shape
	test_shape = test.shape
	sample_cnt, train_cnt, test_cnt = sample_shape[0], train_shape[0], test_shape[0]
	w, h, d = sample_shape[1], sample_shape[2], sample_shape[3]
	# use sample to do the DC, AC substraction
	sample_window, train_window, test_window, sample_label, train_label, test_label = window_process(sample, train, test, sample_label, train_label, test_label)
	print 'before filtering training sample size: %d' % (sample_window.shape[0] )
	sample_filter, sample_label = patch_filter(sample_window, sample_label)
	sample_window, sample_label = KMeans_resample(sample_filter, sample_label, k = k)
	print 'PCA training sample size: %d' % (sample_window.shape[0] )
	# pca training

	d = sample_window.shape[-1]

	train_dc = (np.mean(train_window, axis = 1)*(d**0.5)).reshape(-1,1).reshape(train_cnt, w/2, h/2, 1)
	test_dc = (np.mean(test_window, axis = 1)*(d**0.5)).reshape(-1,1).reshape(test_cnt, w/2, h/2, 1)

	mean = np.mean(sample_window, axis = 1).reshape(-1,1)

	# PCA weight training
	pca = PCA(n_components = int(d*per)-1)
	pca.fit(sample_window - mean)
	np.save('/data/orcs/yueru/Saak/PCA/lossy_cent_'+str(stage) + '_v4.npy',pca.components_)
	train = pca.transform(train_window).reshape(train_cnt, w/2, h/2, -1)
	test = pca.transform(test_window).reshape(test_cnt, w/2, h/2, -1)

	shape = train.shape
	w, h, d = shape[1], shape[2], shape[3]

	train_data = np.zeros((train_cnt, w, h, 1 + d*2))
	test_data = np.zeros((test_cnt, w, h, 1 + d*2))

	train_data[:,:,:,:1] = train_dc[:,:,:,:]
	test_data[:,:,:,:1] = test_dc[:,:,:,:]
	train_data[:,:,:,1:d+1] = train[:,:,:,:].copy()
	train_data[:,:,:,d+1:] = -train[:,:,:,:].copy()
	test_data[:,:,:,1:d+1] = test[:,:,:,:].copy()
	test_data[:,:,:,d+1:] = -test[:,:,:,:].copy()
	train_data[train_data < 0] = 0
	test_data[test_data < 0] = 0

	return train_data, test_data



#f = gzip.open('/media/mcl418-2/New Volume/Yueru Cifar/MNIST/mnist.pkl.gz', 'rb')
f = gzip.open('./mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train = np.concatenate((train_set[0],valid_set[0]),0)#np.load('train_data.npy')
train_label = np.concatenate((train_set[1],valid_set[1]))#np.load('train_label.npy')
test = test_set[0]#np.load('test_data.npy')
test_label = test_set[1]#np.load('test_label.npy')

train_cnt, test_cnt = train.shape[0], test.shape[0]
train = train.reshape((train_cnt,28,28,1))
train = np.lib.pad(train, ((0, 0),(2, 2), (2, 2),(0, 0)), 'constant', constant_values=0)
test = test.reshape((test_cnt,28,28,1))
test = np.lib.pad(test, ((0, 0),(2, 2), (2, 2),(0, 0)), 'constant', constant_values=0)
train_data = np.zeros((train_cnt, 0))
test_data = np.zeros((test_cnt, 0))

print('start training')
stage = 1
Percentage = [0.5, 0.25, 0.25, 0.25, 0.25]
for k in [-1, -1, -1, -1, -1]:
	print 'k value is :%d' % k
	train, test = convolution(train, test, train_label, test_label, k, stage, Percentage[stage-1])
	#train_data = np.append(train_data, train.copy().reshape(train_cnt, -1), axis = 1)
	#test_data = np.append(test_data, test.copy().reshape(test_cnt, -1), axis = 1)
	print_shape(train, test)
	#print_shape(train_data, test_data)

	np.save('/data/orcs/yueru/Saak/feature/train_before_f_test_'+str(stage)+'_v4.npy', train)
	np.save('/data/orcs/yueru/Saak/feature/test_before_f_test_'+str(stage)+'_v4.npy', test)
	stage += 1

 

