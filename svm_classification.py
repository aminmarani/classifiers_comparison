import sys
from mnist import MNIST
from sklearn import svm



def extract_data(data_path):
	#please define the way to read your very own dataset, this function only read Mnist and Iyre
	train = []; test=[]; train_labels = []; test_labels = []
	if(data_path == 'mnist'):
		source = MNIST('./mnist')
		train,train_labels = source.load_training()
		test,test_labels = source.load_testing()
		#print(source.display(train[1]))



	return(train,train_labels,test,test_labels)





def classification(classifier,train,train_labels,test,test_labels):
	if(classifier == 'svm'):
		svm_classify(train,train_labels,test,test_labels)



def svm_classify(X,Xlabels,Y,Ylabels):
	#you can also use decision_function_shape='ovo' for one vs one, but the result won't change
	#I used different kernels (poly with deg 9 0.8581, 4degree 0.9698, rbf = 0.9792 acc, sigmoid with 0.7759 acc)
	model = svm.SVC(decision_function_shape='ovr',kernel='rbf') #define a support vector classifier
	model.fit(X,Xlabels)
	output = model.predict(Y)
	#acc = len(output==Ylabels)/len(Ylabels)
	acc = 0
	for i,j in zip(output,Ylabels):
		if(i==j):
			acc += 1

	print(acc/len(Ylabels))





#main part of the program
data_path = sys.argv[2]
classifier = sys.argv[1]

#get data from data path
train, train_labels, test,test_labels = extract_data(data_path)

classification(classifier,train,train_labels,test,test_labels)
