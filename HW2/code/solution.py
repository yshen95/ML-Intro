import numpy as np 
from helper import *
'''
Homework2: logistic regression classifier
'''

def logistic_regression(data, label, max_iter, learning_rate):
        '''
        The logistic regression classifier function. 
        Args:
        data: train data with shape (1561, 3), which means 1561 samples and each sample has 3 features.(1, symmetry, average internsity)
        label: train data's label with shape (1561,1). 1 for digit number 1 and -1 for digit number 5.
        max_iter: max iteration numbers
        learning_rate: learning rate for weight update
        
        Returns:
        w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
        '''
        # Initialize 
        w = np.zeros((data.shape[1], 1))
        t = 0
        grad = np.zeros((data.shape[1], 1)) 

        for t in range(max_iter):
                for index in np.arange(len(data)):
                        np.transpose(grad)[0] += (label[index]*data[index])/(1+np.exp(label[index]*np.dot(data[index],w))[0])
                grad = -1/len(data) * grad
                w = w - (learning_rate * grad)
        return w

def thirdorder(data):
	'''
	This function is used for a 3rd order polynomial transform of the data.
	
	Args:
	data: input data with shape (:, 3) the first dimension represents total samples (training: 1561; testing: 424) and the second dimesion represents total features.
	
	Return:
	result: A numpy array format new data with shape (:,10), which using a 3rd order polynomial transformation to extend the feature numbers from 3 to 10. 
	The first dimension represents total samples (training: 1561; testing: 424) and the second dimesion represents total features
	'''
	trans = np.zeros((len(data), 10))
	#deal with degree 3 of Z space
	#I(w)=(1,x1,x2,x1^2,x1x2,x2^2,x1^3,x1^2x2,x1x2^2,x2^3)
	for index in np.arange(len(data)):
                trans[index][0] = 1
                trans[index][1] = data[index][0]
                trans[index][2] = data[index][1]
                trans[index][3] = (data[index][0])**2 #** -> ^
                trans[index][4] = data[index][0] * data[index][1]
                trans[index][5] = (data[index][1])**2
                trans[index][6] = (data[index][0])**3
                trans[index][7] = ((data[index][0])**2)*data[index][1]
                trans[index][8] = ((data[index][1])**2)*data[index][0]
                trans[index][9] = (data[index][1])**3
	return trans


def accuracy(x, y, w):
        '''
        This function is used to compute accuracy of a logsitic regression model.
        
        Args:
        x: input data with shape (n, d), where n represents total data samples and d represents total feature numbers of a certain data sample.
        y: corresponding label of x with shape(n, 1), where n represents total data samples.
        w: the seperator learnt from logistic regression function with shape (d, 1), where d represents total feature numbers of a certain data sample.
        
        Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5, which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
        '''
        corr = 0
        for index in range(len(x)):
                if np.exp(np.dot(x[index],w))/(1+np.exp(np.dot(x[index],w))) > 0.5: #if θ(wTx) > 0.5 -> 1
                        val = 1
                        if val == y[index]:
                                corr += 1
                else: #if θ(wTx) < 0.5 -> -1
                        val = -1
                        if val == y[index]:
                                corr += 1
        return float(corr) / len(x)

		
