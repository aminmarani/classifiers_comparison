import sys
from mnist import MNIST
from sklearn import svm
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras import layers,models
import numpy as np



def extract_data(data_path):
	#please define the way to read your very own dataset, this function only read Mnist and Iyre
	train = []; test=[]; train_labels = []; test_labels = []
	if(data_path == 'mnist'):
		source = MNIST('./mnist')
		train,train_labels = source.load_training()
		test,test_labels = source.load_testing()
		#print(source.display(train[0]))



	return(train,train_labels,test,test_labels)





def classification(classifier,train,train_labels,test,test_labels,dataset):
	if(classifier == 'svm'):
		svm_classify(train,train_labels,test,test_labels)
	elif(classifier == 'dnn'):
		dnn_classify(train,train_labels,test,test_labels,dataset)



def svm_classify(X,Xlabels,Y,Ylabels):
	#you can also use decision_function_shape='ovo' for one vs one, but the result won't change
	#I used different kernels (poly with deg 9 0.8581, 4degree 0.9698, rbf = 0.9792, rbf with pca_70 is 0.9846, sigmoid with 0.7759 acc)
	model = svm.SVC(decision_function_shape='ovr',kernel='rbf') #define a support vector classifier
	
	########## Applying PCA (Optional)
	pca = PCA(n_components = 70)
	pca.fit(X) 
	X = pca.transform(X)
	Y = pca.transform(Y)
	########## Applying PCA (Optional)
	
	model.fit(X,Xlabels)
	output = model.predict(Y)
	#acc = len(output==Ylabels)/len(Ylabels)
	acc = 0
	for i,j in zip(output,Ylabels):
		if(i==j):
			acc += 1

	print(acc/len(Ylabels))


def dnn_classify(X,Xlabels,Y,Ylabels,dataset):
	#reshape the data
	if(dataset == 'mnist'):
		lr = np.arange(10)

		Xt = np.asfarray(np.zeros((len(X),28,28)))
		Xl = np.asfarray(np.zeros((len(X),10)))
		Yt = np.asfarray(np.zeros((len(Y),28,28)))
		Yl = np.asfarray(np.zeros((len(Y),10)))
		for i in range(0,len(X)):
			Xt[i,:,:] = np.asfarray(X[i]).reshape(28,28)
			Xl[i] = (lr==Xlabels[i]).astype(np.float)

		for i in range(0,len(Y)):
			Yt[i,:,:] = np.asfarray(Y[i]).reshape(28,28)
			Yl[i] = (lr==Ylabels[i]).astype(np.float)
		
		Xt = np.asfarray(X).reshape(len(X),28,28)
		Yt = np.asfarray(Y).reshape(len(Y),28,28)


		#X = Xt/255
		X = np.asfarray(X)/255
		Xlabels = Xl
		#Y = Yt/255
		Y = np.asfarray(Y)/255
		Ylabels = Yl

	#print(X[0]); exit()

	model = models.Sequential()


	# model.add(layers.Conv2D(28,(3,3), activation='relu', input_shape=(28,28,1) ))
	# model.add(layers.MaxPooling2D(2,2))
	# model.add(layers.Conv2D(64,(3,3), activation='relu'))
	# model.add(layers.MaxPooling2D(2,2))
	# model.add(layers.Conv2D(64,(3,3),activation='relu'))
	# model.add(layers.Flatten())
	# model.add(layers.Dense(64,activation='relu'))
	# model.add(layers.Dense(10))

	model.add(layers.Dense(20,input_dim=784,activation='relu'))
	print(model.output_shape)
	model.add(layers.Dense(40,activation='relu'))
	print(model.output_shape)
	model.add(layers.Dense(60,activation='relu'))
	print(model.output_shape)
	model.add(layers.Dense(80,activation='relu'))
	model.add(layers.Dense(100,activation='relu'))
	model.add(layers.Dense(120,activation='relu'))
	model.add(layers.Dense(120,activation='relu'))
	model.add(layers.Dense(240,activation='relu'))
	model.add(layers.Dense(10,activation='softmax'))
	print(model.output_shape)


	model.compile(optimizer='adam',loss = 'categorical_crossentropy',
		metrics=['accuracy'])
	history = model.fit(X,Xlabels,epochs=20,validation_data=(Y,Ylabels),batch_size=200)

	test_loss,test_acc = model.evaluate(Y,Ylabels,verbose=2)


#main part of the program
data_path = sys.argv[2]
classifier = sys.argv[1]

#get data from data path
train, train_labels, test,test_labels = extract_data(data_path)

classification(classifier,train,train_labels,test,test_labels,data_path)
