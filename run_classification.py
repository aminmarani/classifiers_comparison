import sys
from mnist import MNIST
from sklearn import svm
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier, XGBRFClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, \
							precision_score, f1_score
import matplotlib.pyplot as plt
from scipy import stats



def extract_data(data_path):
	#please define the way to read your very own dataset, this function only read Mnist and Iyre
	train = []; test=[]; train_labels = []; test_labels = []
	if(data_path == 'mnist'):
		# source = MNIST('./mnist')
		# train,train_labels = source.load_training()
		# test,test_labels = source.load_testing()
		(train,train_labels),(test,test_labels) = tf.keras.datasets.mnist.load_data()
		#print(source.display(train[0]))
	elif(data_path == 'iyer'):
		data = pd.read_csv('./iyer.txt',sep='\t',header=None).values
		y = data[:,1]
		y[y==-1] = 0 #re-assign noise to class 0 to have 0-10 classes
		x = np.delete(data,1,1)
		train,test,train_labels,test_labels = train_test_split(x,y,test_size = 0.3, random_state = 12345)
	else:
		print('please correct your data-set name and re-run the program')
		exit()

	return(train,train_labels,test,test_labels)



def classification(classifier,train,train_labels,test,test_labels,dataset,pca_flag):
	if(classifier == 'svm'):
		return svm_classify(train,train_labels,test,test_labels,pca_flag)
	elif(classifier == 'dnn'):
		return dnn_classify(train,train_labels,test,test_labels,dataset,pca_flag)
	elif(classifier == 'xgboost'):
		return xgboost_classify(train,train_labels,test,test_labels,pca_flag)



def svm_classify(X,Xlabels,Y,Ylabels,pca_flag):
	#you can also use decision_function_shape='ovo' for one vs one, but the result won't change
	#I used different kernels (poly with deg 9 0.8581, 4degree 0.9698, rbf = 0.9792, rbf with pca_70 is 0.9846, sigmoid with 0.7759 acc)
	model = svm.SVC(decision_function_shape='ovr',kernel='rbf') #define a support vector classifier
	
	#if mnist-3d data comes, make them vector
	if(len(X.shape)==3):
		X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
		Y = Y.reshape(Y.shape[0],Y.shape[1]*Y.shape[2])

	########## Applying PCA (Optional)
	if(pca_flag and len(X[0])>13):
		pca = PCA(n_components = 70)
		pca.fit(X) 
		X = pca.transform(X)
		Y = pca.transform(Y)
	if(pca_flag and len(X[0])==13):
		#with PCA 0.78, without 0.76
		pca = PCA(n_components = 2)
		pca.fit(X) 
		X = pca.transform(X)
		Y = pca.transform(Y)
	########## Applying PCA (Optional)
	
	model.fit(X,Xlabels)
	output = model.predict(Y)
	
	return output

def dnn_classify(X,Xlabels,Y,Ylabels,dataset,pca_flag):
	model = models.Sequential()
	#(X,Xlabels),(Y,Ylabels) = tf.keras.datasets.mnist.load_data()
	#reshape the data
	if(dataset == 'mnist'):
		#X = np.asfarray(X)/255
		X = X.reshape(X.shape[0],28,28,1)
		X = X.astype('float32')
		X = X / 255
		Xlabels = np.asfarray(Xlabels)
		#Y = np.asfarray(Y)/255
		Y = Y.reshape(Y.shape[0],28,28,1)
		Y = Y.astype('float32')
		Y = Y / 255
		Ylabels = np.asfarray(Ylabels)
		input_shape = (28,28,1)
		class_num = 10
		epochs = 20
		#building the model
		#drop out 0.2 0.9920
		model.add(layers.Conv2D(28,(3,3), activation='relu', input_shape=input_shape ))
		model.add(layers.MaxPooling2D(2,2))
		model.add(layers.Conv2D(64,(3,3), activation='relu'))
		model.add(layers.MaxPooling2D(2,2))
		# model.add(layers.Conv2D(64,(3,3),activation='relu'))
		model.add(layers.Flatten())
		model.add(layers.Dense(128,activation='relu'))
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(class_num,activation='softmax'))

	elif(dataset == 'iyer'):
		#0.88, 0.93 with PCA
		######### Applying PCA (Optional)
		features = 13
		if(pca_flag):
			features = 6
			pca = PCA(n_components = features)
			pca.fit(X) 
			X = pca.transform(X)
			Y = pca.transform(Y)
		########## Applying PCA (Optional)
		X = X.reshape(X.shape[0],features,1)
		Y = Y.reshape(Y.shape[0],features,1)
		input_shape = (features,1)
		class_num = 11
		epochs = 50

		#building the model
		model.add(layers.Conv1D(filters = 1024,kernel_size = 3, activation='relu', input_shape=input_shape ))
		model.add(layers.MaxPooling1D(pool_size = 2))
		# model.add(layers.Conv1D(filters = 2048,kernel_size = 3, activation='relu'))
		# model.add(layers.MaxPooling1D(pool_size = 2))
		model.add(layers.Flatten())
		model.add(layers.Dense(2048,activation='relu'))
		model.add(layers.Dropout(0.2))
		
		model.add(layers.Dense(class_num,activation='softmax'))

		

	
	print(model.output_shape)


	# model.compile(optimizer='adam',loss = 'categorical_crossentropy',
	# 	metrics=['accuracy'])
	model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy',
		metrics=['accuracy'])
	history = model.fit(X,Xlabels,epochs=epochs,validation_data=(Y,Ylabels))
	output = model.predict(Y)
	output = np.argmax(output,axis=1)
	return(output)
	#test_loss,test_acc = model.evaluate(Y,Ylabels,verbose=2)


def xgboost_classify(X,Xlabels,Y,Ylabels,pca_flag):

	if(len(X.shape)==3):
		X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
		Y = Y.reshape(Y.shape[0],Y.shape[1]*Y.shape[2])

	########## Applying PCA (Optional)
	if(pca_flag and len(X[0])==13):
		pca = PCA(n_components = 3)
		pca.fit(X) 
		X = pca.transform(X)
		Y = pca.transform(Y)
	elif(pca_flag and len(X[0])>13):
		pca = PCA(n_components = 50)
		pca.fit(X) 
		X = pca.transform(X)
		Y = pca.transform(Y)
	########## Applying PCA (Optional)

	#MNIST : gbtree, gblinear, dart 0.978
	#IYER: 0.96153, 0.9935 with PCA (3)
	model = XGBClassifier()
	#MNIST : acc 0.9572 (n_estimators=50,max_depth=50, with no PCA)
	#IYER: 0.9487 (n_estimators=500,max_depth=50)
	#model = XGBRFClassifier(n_estimators=50,max_depth=50)
	model.fit(np.asfarray(X),np.asfarray(Xlabels))

	output = model.predict(np.asfarray(Y))

	return output

def metrics_calc(target,output,out_mat):
	#computing accuracy = number of correct classified / all samples
	acc = (np.asfarray(target) == output).sum()/len(target)
	#computing area under curve with one-vs-rest and macro averaging (average over all samples)
	auc = roc_auc_score(target,out_mat,multi_class='ovr',average = 'macro')
	#computing recall with macro averaging
	rpf = []
	rpf.append(recall_score(target,output,average='macro'))
	rpf.append(precision_score(target,output,average='macro'))
	rpf.append(f1_score(target,output,average='macro'))
	return(acc,auc,rpf)




#main part of the program
if len(sys.argv)==1:
	print('run the program similar to what shown below')
	print('python classification.py classifier-name dataset-name')
	exit()

data_path = sys.argv[2]
classifier = sys.argv[1]

pca_flag = False
run_all = False
if(len(sys.argv)>3 and sys.argv[3] == 'pca'):
	pca_flag = True

# if(len(sys.argv)>4 and sys.argv[4] == 'all'):
# 	run_all = True

if(classifier != 'all'):
	#description
	print('result of running ',classifier,' on ',data_path)
	#get data from data path
	train, train_labels, test,test_labels = extract_data(data_path)
	#call classifiers' function
	output = classification(classifier,train,train_labels,test,test_labels,data_path,pca_flag)
	#calculate confusion matrix
	print(confusion_matrix(test_labels,output))
	#compute other metrics
	output_matrix = np.zeros((test_labels.size,int(np.max(test_labels)-np.min(test_labels))+1))
	output_list = [int(i) for i in output.tolist()]
	output_matrix[np.arange(len(output_list)),output_list] = 1.0

	accuracy,auc,recalls = metrics_calc(test_labels,output,output_matrix)
	print('accruracy: ',accuracy,' - area under curveL ',auc)
	print('recall: ',recalls[0],' - precision: ',recalls[1],' - F1: ',recalls[2])
else:
	print('running all models on: ',data_path)
	runs_ = 5 #number of runs
	ensemble = 1
	#get data from data path
	train, train_labels, test,test_labels = extract_data(data_path)
	acc_all = np.zeros((3+ensemble,runs_))
	auc_all = np.zeros((3+ensemble,runs_))
	recall_all = np.zeros((3+ensemble,runs_))

	#run each model 5 times
	for run in range(0,runs_):
		#run svm and compute metrics
		svm_out = classification('svm',train,train_labels,test,test_labels,data_path,pca_flag)
		output_matrix = np.zeros((test_labels.size,int(np.max(test_labels)-np.min(test_labels))+1))
		output_list = [int(i) for i in svm_out.tolist()]
		output_matrix[np.arange(len(output_list)),output_list] = 1.0
		accuracy,auc,recalls = metrics_calc(test_labels,svm_out,output_matrix)
		acc_all[0,run] = accuracy; auc_all[0,run] = auc; recall_all[0,run] = recalls[0]
		
		#run deep learing and compute metrics
		dnn_out = classification('dnn',train,train_labels,test,test_labels,data_path,pca_flag)
		output_matrix = np.zeros((test_labels.size,int(np.max(test_labels)-np.min(test_labels))+1))
		output_list = [int(i) for i in dnn_out.tolist()]
		output_matrix[np.arange(len(output_list)),output_list] = 1.0
		accuracy,auc,recalls = metrics_calc(test_labels,dnn_out,output_matrix)
		acc_all[1,run] = accuracy; auc_all[1,run] = auc; recall_all[1,run] = recalls[0]

		#run XGboost and compute metrics
		xg_out = classification('xgboost',train,train_labels,test,test_labels,data_path,pca_flag)
		output_matrix = np.zeros((test_labels.size,int(np.max(test_labels)-np.min(test_labels))+1))
		output_list = [int(i) for i in xg_out.tolist()]
		output_matrix[np.arange(len(output_list)),output_list] = 1.0
		accuracy,auc,recalls = metrics_calc(test_labels,xg_out,output_matrix)
		acc_all[2,run] = accuracy; auc_all[2,run] = auc; recall_all[2,run] = recalls[0]

		#Do the ensemble learning with majority vote
		if(ensemble>0):
			ensemble_maj_out = stats.mode([svm_out,dnn_out,xg_out]).mode[0]
			output_matrix = np.zeros((test_labels.size,int(np.max(test_labels)-np.min(test_labels))+1))
			output_list = [int(i) for i in ensemble_maj_out.tolist()]
			output_matrix[np.arange(len(output_list)),output_list] = 1.0
			accuracy,auc,recalls = metrics_calc(test_labels,ensemble_maj_out,output_matrix)
			acc_all[3,run] = accuracy; auc_all[3,run] = auc; recall_all[3,run] = recalls[0]
		#Do the ensemble learning with weighting vote
		if(ensemble>1):
			ensemble_W_out = np.round( (svm_out*(acc_all[0,run]/np.sum(acc_all[0:3,run]))+ \
								dnn_out*(acc_all[1,run]/sum(acc_all[0:3,run]))+ \
								xg_out*(acc_all[2,run]/sum(acc_all[0:3,run])) ))
			#exit()
			output_matrix = np.zeros((test_labels.size,int(np.max(test_labels)-np.min(test_labels))+1))
			output_list = [int(i) for i in ensemble_W_out.tolist()]
			output_matrix[np.arange(len(output_list)),output_list] = 1.0
			accuracy,auc,recalls = metrics_calc(test_labels,ensemble_W_out,output_matrix)
			acc_all[4,run] = accuracy; auc_all[4,run] = auc; recall_all[4,run] = recalls[0]

	#show plots of different metrics
	bar_labels = ['SVM','DNN','XGBOOST','Ensemble majority','Ensemble Weighted']
	#accuracy

	fig, ax = plt.subplots()
	ax.bar(np.arange(3+ensemble),np.mean(acc_all,axis=1),yerr = np.std(acc_all,axis=1),align='center')
	ax.set_ylabel('Accuracy %')
	ax.set_xticks(np.arange(3+ensemble))
	ax.set_xticklabels(bar_labels[0:3+ensemble])
	ax.set_title(str('Accuracy Comparison of different models over '+data_path))
	plt.show()

	#AUC
	fig, ax = plt.subplots()
	ax.bar(np.arange(3+ensemble),np.mean(auc_all,axis=1),yerr = np.std(auc_all,axis=1),align='center')
	ax.set_ylabel('AUC %')
	ax.set_xticks(np.arange(3+ensemble))
	ax.set_xticklabels(bar_labels[0:3+ensemble])
	ax.set_title(str('AUC Comparison of different models over '+data_path))
	plt.show()

	#recall
	fig, ax = plt.subplots()
	ax.bar(np.arange(3+ensemble),np.mean(recall_all,axis=1),yerr = np.std(recall_all,axis=1),align='center')
	ax.set_ylabel('Recall %')
	ax.set_xticks(np.arange(3+ensemble))
	ax.set_xticklabels(bar_labels[0:3+ensemble])
	ax.set_title(str('Recall Comparison of different models over '+data_path))
	plt.show()


