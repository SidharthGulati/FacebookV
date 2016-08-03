#!/usr/local/bin/python3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from six.moves import cPickle as pickle
#from xgboost.sklearn import XGBClassifier
"""
	Prepare the data in form of features and doing feature engineering based on weights.
	Private LB Score = 0.5531
"""
def prepare_data(rawdata, n_grid_x, n_grid_y):
	# Computing grids for the dataFrame.
	minX = min(rawdata.x.values)
	maxX = max(rawdata.x.values)
	minY = min(rawdata.y.values)
	maxY = max(rawdata.y.values)
	grid_size_x = (maxX - minX) / n_grid_x
	grid_size_y = (maxY - minY) / n_grid_y
	epsilon = 0.00001
	xs = np.where(rawdata.x.values < epsilon, 0, rawdata.x.values - epsilon)
	ys = np.where(rawdata.y.values < epsilon, 0, rawdata.y.values - epsilon)
	pos_x = (xs/grid_size_x).astype(np.int)
	pos_y = (ys/grid_size_y).astype(np.int) 
	rawdata['grid_num'] = n_grid_x * pos_y + pos_x

	# Feature Engineering 
	weights = [500, 1000, 4, 3, 1./22., 2, 10]  # weights for the features.
	initial_date = np.datetime64('2015-01-01T00:01', dtype='datetime64')
	dates = pd.DatetimeIndex(initial_date + np.timedelta64(int(d)) for d in rawdata.time.values)
	rawdata.x = rawdata.x * weights[0]
	rawdata.y = rawdata.y * weights[1]
	#rawdata.accuracy = rawdata.accuracy * weights[2]
	rawdata['hour'] = dates.hour * weights[2]
	rawdata['weekday'] = ((dates.weekday + 1) * weights[3]).astype(np.int)
	rawdata['year'] = ((dates.year -2014) * weights[6]).astype(np.int)
	rawdata['day'] = (dates.dayofyear * weights[4]).astype(int)
	rawdata['month'] = (dates.month * weights[5]).astype(int)

	rawdata = rawdata.drop(['time'], axis=1)
	return rawdata

"""
	Classification of  data from single grid and predicting the labels for test data. 
"""
def process_1_grid(df_train, df_test, grid, threshold):

	# Creating data with the particular grid id.
	df_train_1_grid = df_train.loc[df_train.grid_num == grid]
	df_test_1_grid = df_test.loc[df_test.grid_num == grid]
	place_counts = df_train_1_grid.place_id.value_counts()
	mask = (place_counts[df_train_1_grid.place_id.values] >= threshold).values
	df_train_1_grid = df_train_1_grid.loc[mask]
	# Label Encoding
	le = LabelEncoder()
	labels = le.fit_transform(df_train_1_grid.place_id.values)
	
	# Computing train and test feature data for grid grid.
	X = df_train_1_grid.drop(['place_id','grid_num'], axis=1).values.astype(int)
	X_test = df_test_1_grid.drop(['grid_num'], axis=1).values.astype(int)
	row_id = df_test_1_grid.index
	
	# KNN Classifier 
	clf = KNeighborsClassifier(n_neighbors=20, weights= 'distance', metric='manhattan')
	#clf = GaussianNB()
	# Training of the classifier
	#clf = XGBClassifier(max_depth=10, learning_rate=0.5, n_estimators=25,objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)                  
	clf.fit(X,labels)

	
	# Predicting probabilities for each of the label for test data.
	prob_y = clf.predict_proba(X_test)
	
	# Transforming back to labels from One hot Encoding
	pred_labels = le.inverse_transform(np.argsort(prob_y, axis=1)[:,::-1][:,:3])
	return pred_labels, row_id
"""
	Processing of the test and train data for classification and prediction.
"""
def process_data(df_train, df_test, grids, threshold):

	predictions = np.zeros((df_test.shape[0], 3), dtype=int)
	for iteration in range(grids) :
		if iteration % 50 ==0:
			print("Processing for Grid Number = {0}").format(iteration)
		pred_labels, row_id = process_1_grid(df_train, df_test, iteration, threshold)
		predictions[row_id] = pred_labels

	print('Generating submission file ...')
	# Auxilary Dataframe with the 3 best predictions for each sample
	df_aux = pd.DataFrame(predictions, dtype=str, columns=['p1', 'p2', 'p3'])  

	#Concatenating the 3 predictions for each sample as per the submission file
	df_final = df_aux.p1.str.cat([df_aux.p2, df_aux.p3], sep=' ')

	#Writting to csv
	df_final.name = 'place_id'
	df_final.to_csv('submission_knn.csv', index=True, header=True, index_label='row_id')  


def main():
	
	print("Reading data from data file.......")
	train_file = pd.read_csv('./dataset/train.csv', index_col=0,usecols = ['row_id','x','y','accuracy','time','place_id'])
	test_file = pd.read_csv('./dataset/test.csv', index_col=0,usecols = ['row_id','x','y','accuracy','time'])

	# Specifying number of grids.
	n_grid_x = 20
	n_grid_y = 40
	threshold = 5
	print("Preparing data.......")
	df_train = prepare_data(train_file, n_grid_x, n_grid_y)
	df_test = prepare_data(test_file,n_grid_x,n_grid_y)
	#df_train.to_csv('train_data_processed.csv', index=True, header=True, index_label='row_id')
	#df_test.to_csv('test_data_processed.csv', index=True, header=True, index_label='row_id')
	print "Processing data......."
	process_data(df_train,df_test, n_grid_y * n_grid_x, threshold)

if __name__ == '__main__':
	main()