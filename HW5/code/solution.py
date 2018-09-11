import sys
sys.path.append('libsvm/python/')
from svmutil import svm_train, svm_predict
'''
Homework5: support vector machine classifier

You need to use two functions 'svm_train' and 'svm_predict'
from libsvm library to start your homework. Please read the 
readme.txt file carefully to understand how to use these 
two functions.

'''

'''
2. Train a svm classifier:
Command:
model = svm_train(train_label, train_data, libsvm_options)

- train_label: an m by 1 list of training labels. m represents total training data samples.
- train_data: an m by n two dimension list. m represents total training data samples and n represents number of features for each data sample.
- libsvm_options: a string format of training options. You will using following options in your homework:
-c cost: set the parameter C of C-SVC, epsilon-SVR and nu-SVR (default 1)
-t kernel: set type of kernel function (default 3). 0: linear kernel; 1: polynomial kernel; 2: radial basis function kernel.

Here is an example to use svm_train:

suppose you have train_data and train_label, then set up libsvm options as:
libsvm_options = '-c 2 -t 1' and code:
model = svm_train(train_label, train_data, libsvm_options)
'''

'''
3. Predict label on test data
Command:
predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)

- test_label: an m by 1 list of prediction labels
- test_data: an m by n two dimension list. m represents total testing data samples and n represents number of features for each data sample.
- model: the output of svm_train function.
This function will return three values: predicted label, test accuracy and decision values with classify accuracy printing out.
'''


def svm_with_diff_c(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different value 
	of cost c to train a svm classify model. Then apply the trained model
	on testing label and data.
	
	The value of cost c you need to try is listing as follow:
	c = [0.01, 0.1, 1, 2, 3, 5]
	Please keep other parameter options as default.
	No return value is needed
	'''
	c = [0.01, 0.1, 1, 2, 3, 5]
	for cost in c:
                libsvm_options = '-c ' + str(cost)
                model = svm_train(train_label, train_data, libsvm_options)
                predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)
	
def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
	'''
	Use 'svm_train' function with training label, data and different kernel
	to train a svm classify model. Then apply the trained model
	on testing label and data.
	
	The kernel  you need to try is listing as follow:
	1. linear kernel
	2. polynomial kernel
	3. radial basis function kernel
	Please keep other parameter options as default.
	No return value is needed
	'''

	'''
         -t kernel: set type of kernel function (default 3). 0: linear kernel; 1: polynomial kernel; 2: radial basis function kernel.
        '''
	k = [0, 1, 2]
	for kernel in k:
                libsvm_options = '-t ' + str(kernel)
                model = svm_train(train_label, train_data, libsvm_options)
                predicted_label, test_acc, decision_values = svm_predict(test_label, test_data, model)
