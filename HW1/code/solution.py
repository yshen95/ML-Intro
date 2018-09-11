import numpy as np 
from helper import *

'''
Homework1: perceptron classifier
'''
def sign(x):
	return 1 if x > 0 else -1

#-------------- Implement your code Below -------------#

def show_images(data):
	'''
	This function is used for plot image and save it.

	Args:
	data: Two images from train data with shape (2, 16, 16). The shape represents total 2
	      images and each image has size 16 by 16. 

	Returns:
		Do not return any arguments, just save the images you plot for your report.
	'''
	
	'''
    data = mpimg.imread load_data in helper.py
    '''
	
	for loop in range(2):
                plt.title('Training Data Image')
                imgplot = plt.imshow(data[loop])
                plt.show()
                

def show_features(data, label):
	'''
	This function is used for plot a 2-D scatter plot of the features and save it. 

	Args:
	data: train features with shape (1561, 2). The shape represents total 1561 samples and 
	      each sample has 2 features.
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the 2-D scatter plot of the features you plot for your report.
	'''
	fig, ax = plt.subplots()
	# zip in python reutn an iterator of tuples, in python 2 use itertools.izip instead
	for x,y,pll in zip(data[:,0],data[:,1],label):
		if pll == 1:
			ax.scatter(x,y,marker='*',c='red',s=70)
		elif pll == -1:
			ax.scatter(x,y,marker='+',c='blue',s=70)
	plt.title('2-D Scatter Plot For Training Data')
	plt.show()


def perceptron(data, label, max_iter, learning_rate):
	'''
	The perceptron classifier function.

	Args:
	data: train data with shape (1561, 3), which means 1561 samples and 
		  each sample has 3 features.(1, symmetry, average internsity)
	label: train data's label with shape (1561,1). 
		   1 for digit number 1 and -1 for digit number 5.
	max_iter: max iteration numbers
	learning_rate: learning rate for weight update
	
	Returns:
		w: the seperater with shape (1, 3). You must initilize it with w = np.zeros((1,d))
	'''
	w = np.zeros(len(data[0])) 
	t = 0
	for t in range(max_iter):
		for index, val in enumerate(data):
			'''
			two cases for missclassified:
			1) XTw(t)<0 and Y>0 result<0
			2) XTw(t)>0 and Y<0 result<0
			'''
			if (sign(np.dot(data[index],w)*label[index])) < 0:
				w = w + learning_rate * label[index] * val	
	return w
	

def show_result(data, label, w):
	'''
	This function is used for plot the test data with the separators and save it.
	
	Args:
	data: test features with shape (424, 2). The shape represents total 424 samples and 
	      each sample has 2 features.
	label: test data's label with shape (424,1). 
		   1 for digit number 1 and -1 for digit number 5.
	
	Returns:
	Do not return any arguments, just save the image you plot for your report.
	'''
	fig, ax = plt.subplots()	

	for x,y,pll in zip(data[:,0],data[:,1],label):
		if pll == 1:
			ax.scatter(x,y,marker='*',c='red',s=70)
		elif pll == -1:
			ax.scatter(x,y,marker='+',c='blue',s=70)

	x1 = np.linspace(np.amin(data[:,0]),np.amax(data[:,0]))
	x2 = (-w[1]/w[2])*x1+(-w[0]/w[2])
	ax.plot(x1,x2)
	
	plt.title('2-D Scatter Plot For Test Data')
	plt.show()


#-------------- Implement your code above ------------#
def accuracy_perceptron(data, label, w):
	n, _ = data.shape
	mistakes = 0
	for i in range(n):
		if sign(np.dot(data[i,:],np.transpose(w))) != label[i]:
			mistakes += 1
	return (n-mistakes)/n


def test_perceptron(max_iter, learning_rate):
	#get data
	traindataloc,testdataloc = "../data/train.txt", "../data/test.txt"
	train_data,train_label = load_features(traindataloc)
	test_data, test_label = load_features(testdataloc)
	#train perceptron
	w = perceptron(train_data, train_label, max_iter, learning_rate)
	train_acc = accuracy_perceptron(train_data, train_label, w)	
	#test perceptron model
	test_acc = accuracy_perceptron(test_data, test_label, w)
	return w, train_acc, test_acc


