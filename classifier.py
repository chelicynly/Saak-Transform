from sklearn.decomposition import PCA
import numpy as np
import cPickle, gzip
from sklearn import svm
#import matplotlib.pyplot as plt
#from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
#
f = gzip.open('/media/mcl418-2/New Volume/Yueru Cifar/MNIST/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
train_label = np.concatenate((train_set[1],valid_set[1]))
test_label = test_set[1]

train_data = np.load('train_before_f.npy')
test_data = np.load('test_before_f.npy' )
#train_data = train_data[:50000,:]
#train_data = train_set[0]
#test_data = test_set[0]
def evac_ftest(rep2,label):              
    F,_ = f_classif(rep2,label)
    where_are_NaNs = np.isnan(F)
    F[where_are_NaNs] = 0
    return F
    
Eva = evac_ftest(train_data, train_label)
np.save('./eva_ftest.npy',Eva)
idx = Eva > np.sort(Eva)[::-1][2000]
train_data = train_data[:,idx]
test_data = test_data[:,idx]

for n_components in [64,128]:
    pca = PCA(n_components)
    pca.fit(train_data)
    traindata = pca.transform(train_data) 
    testdata = pca.transform(test_data)
            
    clf = svm.SVC()#RandomForestClassifier(n_estimators = 1000)#
    clf.fit(traindata, train_label)
    pre  = clf.predict(testdata)
    print 'reduce dim to %d:' % n_components
    print np.count_nonzero(pre == test_label)

