import os
import sys
import csv
from random import shuffle
from scipy import stats
import numpy as np
import math

# numCells = 40
numRows = 40
numCols = 40
base_dir = 'test40/'
tests_frames = [[[10,0],[10,1]]
, [[10,0],[10,2]]
, [[10,0],[10,3]]
,[[10,0,10],[10,0,20]],[[10,0,10],[10,0,40]]]

def manhanton(i, j):
	return abs(i[0]-j[0])+abs(i[1]-j[1])

def euclidean(i,j):
	return math.sqrt((i[0]-j[0])**2+(i[1]-j[1])**2)

def folder_check(time_frame):
	folder_name = base_dir+'raster'
	for time in time_frame:
		for timepart in time:
			folder_name += str(timepart)
	return folder_name

def rowcol_to_index(row, col):
	'''2D coordinates to 1D index'''
	return row * numCols + col

def index_to_rowcol(i):
	row = i/numCols
	col = i%numCols
	return row, col 

def inflow(row,col,matrix):
	'''inflow of cell(row, col)'''
	result = 0
	size = numRows*numCols
	dindex = rowcol_to_index(row,col)
	for i in range(size):
		if i != dindex:
			result += matrix[i][dindex]
	return result

def outflow(row,col,matrix):
	'''outflow of cell(row, col)'''
	result = 0
	size = numRows*numCols
	oindex = rowcol_to_index(row,col)
	for i in range(size):
		if i != oindex:
			result += matrix[oindex][i]
	return result


def read_rasterserface(file_name):
	'''read raster data from asc file (start snapshot or end snapshot) return matrix of object counts'''
	matrix = []
	with open(file_name,'r') as f:
		for line in f:
			if line[0].isdigit():
				matrix.append(map(lambda x: int(x),filter(lambda x : not x.isspace(),line.split("\t"))))
	return matrix

def calc_selfflow(start_raster, end_raster, odmatrix):
	'''calculate selfflow info for a given ODmatrix'''
	for i in range(numRows):
		for j in range(numCols):
			odindex = rowcol_to_index(i,j)
			odmatrix[odindex][odindex] = min(start_raster[i][j]-outflow(i,j,odmatrix),end_raster[i][j]-inflow(i,j,odmatrix))



def make_ODmatrix(file_name):
	'''read ODmatrix from csv file'''
	size = numCols * numRows
	matrix = [[0 for i in range(size)] for j in range(size)]
	with open(file_name,'rb') as f:
		table = csv.reader(f)
		for row in table:
			from_index = rowcol_to_index(int(row[0]),int(row[1]))
			to_index = rowcol_to_index(int(row[2]),int(row[3]))
			matrix[from_index][to_index] += int(float(row[4]))
	return matrix

# intuitive evaluation
def intuitive_contingency_table(estimate,groundtruth):
	'''return contingency table that 
	hits = number of instances that predicated movement and true movement end up at the same cell but not the start cell;
	misses = number of instances that predicated stationary but did move in reality;
	false alarm = number of instances that the object is predicated to move, but in reality it didn't move or move to a different place;
	correct negatives = number of instances that object is predicated to be stationary and is also stationary in reality'''
	size = numRows * numCols
	total_moves = 0
	hits = 0
	false_alarm = 0
	misses = 0
	correct_negatives = 0
	for i in range(size):
		for j in range(size):
			total_moves += groundtruth[i][j]
			if i == j:
				correct_negatives += min(estimate[i][j],groundtruth[i][j])
				if estimate[i][j]>groundtruth[i][j]:
					misses += estimate[i][j] - groundtruth[i][j]
			else:
				hits += min(estimate[i][j],groundtruth[i][j])
	false_alarm = total_moves - correct_negatives - hits - misses
	return hits, false_alarm, misses, correct_negatives, total_moves

def intuitive_accuracy(contingency_table):
	hits, false_alarm, misses, correct_negatives, total_moves = contingency_table
	return float(hits+correct_negatives)/total_moves



# dichotomous forecasts: event, whether an object moved
def dichotomous_contingency_table(estimate,groundtruth):
	'''return contingency table that 
	hits = predicated move and did move indeed;
	false alarm = predicated move but didn't move in reality;
	misses = predicated didn't move but did move in reality;
	correct negatives = predicated didn't move and didn't move in reality;

	THIS ANALYSIS IS OBJECT-BASED

	'''
	size = numRows*numCols
	total_moves = 0
	hits = 0
	false_alarm = 0
	misses=0
	correct_negatives=0
	for i in range(size):
		prediction_possitive = 0
		prediction_negative = 0
		true_possitive = 0
		true_negative = 0
		for j in range(size):
			total_moves += groundtruth[i][j]
			if i!=j:
				prediction_possitive += estimate[i][j]
				true_possitive += groundtruth[i][j]
			else:
				prediction_negative += estimate[i][j]
				true_negative += groundtruth[i][j]
		if true_possitive>prediction_possitive:
			hits += prediction_possitive
			misses += true_possitive - prediction_possitive
			correct_negatives += true_negative
		else:
			hits += true_possitive
			false_alarm += prediction_possitive - true_possitive
			correct_negatives += prediction_negative
	return hits, false_alarm, misses, correct_negatives, total_moves

def dichotomous_accuracy(contingency_table):
	hits, false_alarm, misses, correct_negatives, total_moves = contingency_table
	return float(hits+correct_negatives)/total_moves

def dichotomous_bias(contingency_table):
	hits, false_alarm, misses, correct_negatives, total_moves = contingency_table
	return float(hits+false_alarm)/(hits+misses)

def dichotomous_ets(contingency_table):
	hits, false_alarm, misses, correct_negatives, total_moves = contingency_table
	hits_random = (hits+misses)*(hits+false_alarm)/float(total_moves)
	return (hits - hits_random)/(hits + misses + false_alarm - hits_random)

def dichotomous_hk(contingency_table):
	hits, false_alarm, misses, correct_negatives, total_moves = contingency_table
	return float(hits)/(hits+misses) - float(false_alarm)/(false_alarm+correct_negatives)

def dichotomous_hss(contingency_table):
	hits, false_alarm, misses, correct_negatives, total_moves = contingency_table
	ec = float((hits+misses)*(hits+false_alarm)+(correct_negatives+misses)*(correct_negatives+false_alarm))/total_moves
	return (hits+correct_negatives-ec)/(total_moves-ec)

def dichotomous_orss(contingency_table):
	hits, false_alarm, misses, correct_negatives, total_moves = contingency_table
	return float(hits*correct_negatives - misses*false_alarm)/(hits*correct_negatives + misses*false_alarm)

def dichotomous_accuracy_direct(estimate,groundtruth):
	'''CELL-BASED: return a list of figures representing accuracy for each cell'''
	accu_sum = []
	size = numRows*numCols
	for i in range(size):
		total_moves = 0
		predict_move = 0
		predict_static =0 
		true_move = 0
		true_static = 0
		for j in range(size):
			total_moves += groundtruth[i][j]
			if i != j:
				predict_move += estimate[i][j]
				true_move += groundtruth[i][j]
			else:
				predict_static += estimate[i][j]
				true_static += groundtruth[i][j]
		if total_moves:
			accu_sum.append(float(min(predict_static, true_static)+min(predict_move, true_move))/total_moves)
	return accu_sum


# multi-category forecasts
def calculate_accuracy(estimate, groundtruth):
	'''accuracy as proportion of objects that with the right destination.
	return list of accuracies, one for each cell;'''
	accu_sum = []
	size = numCols*numRows
	for i in range(size):
		total_moves = 0
		correct_moves = 0
		for j in range(size):
			total_moves += groundtruth[i][j]
			correct_moves += min(estimate[i][j],groundtruth[i][j])
		if total_moves:
			accu_sum.append(float(correct_moves)/total_moves)
	return accu_sum

def calculate_flow_accuracy(estimate, groundtruth):
	'''accuracy as proportion of objects that with the right destination.
	return one single value for the entire space;'''
	total = 0
	correct = 0
	size = numCols*numRows
	for i in range(size):
		for j in range(size):
			total += groundtruth[i][j]
			correct += min(estimate[i][j],groundtruth[i][j])
	return float(correct)/total


def calculate_hss(estimate, groundtruth):
	'''return a list of HSS, one for each cell'''
	size = numRows*numCols
	hss_sum = []
	for i in range(size):
		total_moves = 0.0
		correct_moves = 0.0
		product =0.0
		for j in range(size):
			correct_moves += min(estimate[i][j],groundtruth[i][j])
			product += estimate[i][j] * groundtruth[i][j]
			total_moves += groundtruth[i][j]
		if total_moves >= 1:
			try:
				hss_sum.append((correct_moves/total_moves - product/(total_moves*total_moves))/(1-product/(total_moves*total_moves)))
			except Exception, e:
				pass		
	return hss_sum

def calculate_flow_hss(estimate, groundtruth):
	'''return one single value of HSS for the entire space'''
	size = numRows*numCols
	total_moves = 0.0
	correct_moves = 0.0
	product = 0.0
	for i in range(size):
		for j in range(size):
			correct_moves += min(estimate[i][j],groundtruth[i][j])
			product += estimate[i][j] * groundtruth[i][j]
			total_moves += groundtruth[i][j]
	return (correct_moves/total_moves - product/(total_moves*total_moves))/(1-product/(total_moves*total_moves))

# grouped by movement distance
def calculate_groupped_accuracy(estimate,groundtruth):
	'''accuracies grouped by the length of movement'''
	total_moves = {}
	hits = {}
	size = numRows * numCols
	for i in range(size):
		orowcol = index_to_rowcol(i)
		for j in range(size):
			if groundtruth[i][j]!=0:
				drowcol = index_to_rowcol(j)
				dist = manhanton(orowcol,drowcol)
				if dist in total_moves.keys():
					total_moves[dist] += groundtruth[i][j]
					hits[dist] += min(estimate[i][j],groundtruth[i][j])
				else:
					total_moves[dist] = groundtruth[i][j]
					hits[dist] = min(estimate[i][j], groundtruth[i][j])
	accuracy = {}
	for key in total_moves.keys():
		accuracy[key] = float(hits[key])/total_moves[key]
	return accuracy


#multi-category predicate: based on distance
def distance_accuracy(estimate,groundtruth,distance_func):
	'''accuracy as the proportion of objects that get length of movement right;
	return a list of accuracies, one for each cell'''
	accu_sum =[]
	size = numRows * numCols
	for i in range(size):
		orowcol = index_to_rowcol(i)
		e = {}
		g = {}
		for j in range(size):
			drowcol = index_to_rowcol(j)
			dist = distance_func(orowcol,drowcol)
			if estimate[i][j]:
				if dist in e.keys():
					e[dist] += estimate[i][j]
				else:
					e[dist] = estimate[i][j]
			if groundtruth[i][j]:
				if dist in g.keys():
					g[dist] += groundtruth[i][j]
				else:
					g[dist] = groundtruth[i][j]
		total_moves = 0
		moves_with_correct_distance = 0
		for key in g.keys():
			total_moves += g[key]
			if key in e:
				moves_with_correct_distance += min(g[key],e[key])
		if total_moves:
			accu_sum.append(float(moves_with_correct_distance)/total_moves)
	return accu_sum

def distance_flow_accuracy(estimate,groundtruth,distance_function):
	'''accuracy as the proportion of objects that get length of movement right;
	return a single value for the entire space;'''
	size = numRows * numCols
	total =0.0
	correct = 0.0
	# stat_dist_total =0.0
	# stat_dists = []
	for i in range(size):
		orowcol = index_to_rowcol(i)
		e = {}
		g = {}
		for j in range(size):
			drowcol = index_to_rowcol(j)
			dist = distance_function(orowcol,drowcol)
			# stat_dist = euclidean(orowcol,drowcol)
			# stat_dist_total += stat_dist * groundtruth[i][j]
			if estimate[i][j]:
				if dist in e.keys():
					e[dist] += estimate[i][j]
				else:
					e[dist] = estimate[i][j]
			if groundtruth[i][j]:
				# stat_dists.append(stat_dist)
				if dist in g.keys():
					g[dist] += groundtruth[i][j]
				else:
					g[dist] = groundtruth[i][j]
		total_moves = 0
		moves_with_correct_distance = 0
		for key in g.keys():
			total_moves += g[key]
			if key in e:
				moves_with_correct_distance += min(g[key],e[key])
		total += total_moves
		correct += moves_with_correct_distance
	# print stat_dist_total/total
	# print max(stat_dists)
	return correct/total


def distance_hss(estimate, groundtruth,distance_function):
	'''return a list of HSS values relating to length of movement, one for each cell'''
	hss_sum = []
	size = numRows * numCols
	for i in range(size):
		orowcol = index_to_rowcol(i)
		e = {}
		g = {}
		for j in range(size):
			drowcol = index_to_rowcol(j)
			dist = distance_function(orowcol,drowcol)
			if estimate[i][j]:
				if dist in e.keys():
					e[dist] += estimate[i][j]
				else:
					e[dist] = estimate[i][j]
			if groundtruth[i][j]:
				if dist in g.keys():
					g[dist] += groundtruth[i][j]
				else:
					g[dist] = groundtruth[i][j]
		total_moves = 0.0
		moves_with_correct_distance =0.0
		product = 0.0
		for key in g.keys():
			total_moves += g[key]
			if key in e:
				moves_with_correct_distance += min(g[key],e[key])
				product += g[key]*e[key]
		if total_moves:
			try:
				hss_sum.append((moves_with_correct_distance/total_moves-product/total_moves/total_moves)/(1-product/total_moves/total_moves))
			except Exception, e:
				pass
	return hss_sum



def baseline_solution(start_matrix, end_matrix):
	'''return OD values based on random assignment;
	really naive baseline solution;'''
	size = numRows * numCols
	consumers = []
	providers = []
	odvaules = [[0 for i in range(size)] for j in range(size)]
	for i in range(numCells):
		for j in range(numCells):
			if start_matrix[i][j] > end_matrix[i][j]:
				providers.append({'row':i,'col':j,'q':start_matrix[i][j]-end_matrix[i][j]})
			elif end_matrix[i][j] > start_matrix[i][j]:
				consumers.append({'row':i,'col':j,'q':end_matrix[i][j]-start_matrix[i][j]})
			odindex = rowcol_to_index(i,j)
			odvaules[odindex][odindex] = min(start_matrix[i][j],end_matrix[i][j])
	shuffle(providers)
	shuffle(consumers)
	p_index = len(providers)-1
	c_index = len(consumers)-1
	while p_index!=0 and c_index!=0:
		provider = providers[p_index]
		consumer = consumers[c_index]
		if provider['q'] > consumer['q']:
			odvaules[rowcol_to_index(provider['row'],provider['col'])][rowcol_to_index(consumer['row'],consumer['col'])] = consumer['q']
			provider['q'] -= consumer['q']
			c_index -= 1
		elif provider['q']<consumer['q']:
			odvaules[rowcol_to_index(provider['row'],provider['col'])][rowcol_to_index(consumer['row'],consumer['col'])] = provider['q']
			consumer['q'] -= provider['q']
			p_index -= 1
		else:
			odvaules[rowcol_to_index(provider['row'],provider['col'])][rowcol_to_index(consumer['row'],consumer['col'])] = provider['q']
			c_index -= 1
			p_index -= 1
	return 	odvaules

def greedy_baseline(start_matrix,end_matrix):
	'''Baseline solution based on greedy solution;
	return ODmatrix;'''
	size = numRows * numCols
	consumers = []
	providers = []
	odvaules = [[0 for i in range(size)] for j in range(size)]
	for i in range(numRows):
		for j in range(numCols):
			if start_matrix[i][j] > end_matrix[i][j]:
				providers.append({'row':i,'col':j,'q':start_matrix[i][j]-end_matrix[i][j]})
			elif end_matrix[i][j] > start_matrix[i][j]:
				consumers.append({'row':i,'col':j,'q':end_matrix[i][j]-start_matrix[i][j]})
			odindex = rowcol_to_index(i,j)
			odvaules[odindex][odindex] = min(start_matrix[i][j],end_matrix[i][j])
	shuffle(providers)
	while len(consumers) and len(providers):
		provider = providers[0]
		distance_list = [manhanton((provider['row'],provider['col']),(consumer['row'],consumer['col'])) for consumer in consumers]
		min_dist = min(distance_list)
		consumer_index = distance_list.index(min_dist)
		consumer = consumers[consumer_index]
		oi = rowcol_to_index(provider['row'],provider['col'])
		di = rowcol_to_index(consumer['row'],consumer['col'])
		quantity=min(provider['q'],consumer['q'])
		odvaules[oi][di] = quantity
		provider['q'] -= quantity
		if provider['q'] == 0:
			providers.pop(0)
		consumer['q'] -= quantity
		if consumer['q'] == 0:
			consumers.pop(consumer_index)
	return odvaules

def clear_selfflow(matrix):
	'''Delete selfflow cases when stationary objects are not of interest'''
	size = numRows * numCols
	for i in range(size):
		for j in range(size):
			if i == j:
				matrix[i][j] = 0
	return matrix


def read_groundtruth(file_name):
	'''read groundtruth from an ODmatrix in a csv file'''
	matrix = []
	with open(file_name,'r') as f:
		table = csv.reader(f)
		for row in table:
			matrix.append([int(x) for x in row])
	return matrix

def read_groundtruth_list(file_name):
	'''read groundtruth from a list of flow records in a csv file '''
	matrix = [[[] for i in range(numCols)] for j in range(numRows)]
	with open(file_name,'r') as f:
		table = csv.reader(f)
		for row in table:
			row_n = int(row[0])
			col_n = int(row[1])
			true_coords = (float(row[2]),float(row[3]))
			matrix[row_n][col_n].append(true_coords)
	return matrix


# def frame_test(t):
# 	folder = folder_check(t)
# 	print "Test on " + folder
# 	if not os.path.exists(folder):
# 		print "Folder not found."
# 		return False
# 	else:
# 		groundtruth_file = folder + '/groundtruth.csv'
# 		withnetwork_file = folder + '/odMatrix_network.csv'
# 		withoutnetwork_file = folder + '/odMatrix_without.csv'
# 		start_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[0]))+".asc"
# 		start_raster = read_rasterserface(start_raster_file)
# 		end_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[1]))+".asc"
# 		end_raster = read_rasterserface(end_raster_file)
# 		groundtruth_matrix = read_groundtruth(groundtruth_file)
# 		withnetwork_matrix = make_ODmatrix(withnetwork_file)
# 		calc_selfflow(start_raster,end_raster,withnetwork_matrix)
# 		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
# 		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
# 		withnetwork_acc = calculate_accuracy(withnetwork_matrix,groundtruth_matrix)
# 		withoutnetwork_acc = calculate_accuracy(withoutnetwork_matrix, groundtruth_matrix)
# 		print "Accuracy with network info is " + str(withnetwork_acc)
# 		print "Accuracy without network info is " + str(withoutnetwork_acc)
# 	return withnetwork_acc, withoutnetwork_acc

# def statistics(t):
# 	'''performed without network constraint'''
# 	folder = folder_check(t)
# 	print "Test on " + folder
# 	if not os.path.exists(folder):
# 		print "Folder not found."
# 		return False
# 	else:
# 		groundtruth_file = folder + '/groundtruth.csv'
# 		withoutnetwork_file = folder + '/odMatrix_without.csv'
# 		start_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[0]))+".asc"
# 		start_raster = read_rasterserface(start_raster_file)
# 		end_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[1]))+".asc"
# 		end_raster = read_rasterserface(end_raster_file)
# 		groundtruth_matrix = read_groundtruth(groundtruth_file)
# 		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
# 		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
# 		baseline = baseline_solution(start_raster,end_raster)
# 		# dichotomous test
# 		print "dichotomous tests:"
# 		baseline_contingency = dichotomous_contingency_table(baseline,groundtruth_matrix)
# 		my_contingency = dichotomous_contingency_table(withoutnetwork_matrix,groundtruth_matrix)
# 		print "Accuracy: " + str(dichotomous_accuracy(my_contingency)) + '    ' + str(dichotomous_accuracy(baseline_contingency))
# 		print "Bias: " + str(dichotomous_bias(my_contingency)) + '    ' + str(dichotomous_bias(baseline_contingency))
# 		print "ETS: " + str(dichotomous_ets(my_contingency)) + '    ' + str(dichotomous_ets(baseline_contingency))
# 		print "HK: " +str(dichotomous_hk(my_contingency))+ '    ' + str(dichotomous_hk(baseline_contingency))
# 		print "HSS: " +str(dichotomous_hss(my_contingency))+ '    ' + str(dichotomous_hss(baseline_contingency))
# 		print "ORSS: " + str(dichotomous_orss(my_contingency)) + '    ' + str(dichotomous_orss(baseline_contingency))
# 		# multi-category test
# 		print "multi-category test:"
# 		print "Accuracy: " + str(calculate_accuracy(withoutnetwork_matrix,groundtruth_matrix)) + '    ' + str(calculate_accuracy(baseline,groundtruth_matrix))
# 		print "HSS: " + str(calculate_hss(withoutnetwork_matrix,groundtruth_matrix)) + '    ' + str(calculate_hss(baseline,groundtruth_matrix))
# 		# intuitive test
# 		print "intuitive test:"
# 		baseline_contingency = intuitive_contingency_table(baseline,groundtruth_matrix)
# 		my_contingency = intuitive_contingency_table(withoutnetwork_matrix,groundtruth_matrix)
# 		print "Accuracy: " + str(intuitive_accuracy(my_contingency)) + '    ' + str(intuitive_accuracy(baseline_contingency))
# 		# grouped accuracy test
# 		print "accuracy statistics grouped by distance:"
# 		print "Accuracy: " + str(calculate_groupped_accuracy(withoutnetwork_matrix,groundtruth_matrix))
# 		# multi-category test: based on distance
# 		print "Accuracy distance category:"
# 		print "Accuracy: " + str(distance_accuracy(withoutnetwork_matrix,groundtruth_matrix))



def diff_in_sec(t1, t2):
	secs = ((t2[0]-t1[0])*60 + (t2[1]-t1[1]))*60
	if len(t1) == 3:
		secs += t2[2]-t1[2]
	return secs

def manhanton_index(i1,i2):
	row1, col1 = index_to_rowcol(i1)
	row2, col2 = index_to_rowcol(i2)
	return manhanton((row1,col1),(row2,col2))

def euclidean_index(i1,i2):
	row1, col1 = index_to_rowcol(i1)
	row2, col2 = index_to_rowcol(i2)
	return euclidean((row1,col1),(row2,col2))

def degree_of_flow_wrongness(estimate,groundtruth):
	'''object-based target error, take the average out of all objects'''
	size = numRows*numCols
	wrongness =[]
	for i in range(size):
		wrongness += degree_of_wrongness_cell(estimate[i],groundtruth[i],euclidean_index)
	return np.mean(wrongness)

def degree_of_wrongness(estimate, groundtruth):
	'''object-based target errors, return a list of errors one for each object'''
	size = numRows*numCols
	wrongness =[]
	for i in range(size):
		wrongness += degree_of_wrongness_cell(estimate[i],groundtruth[i],manhanton_index)
	return wrongness

def degree_of_wrongness_cell(estimate_row, groundtruth_row, dist_func):
	'''estimate_row: movements start from one cell according to estimation;
	groundtruth_row: movements start from one cell according to groundtruth;
	return a list of errors that representing the distances between estimated movements and groundtruth movements all starting from one particular cell'''
	estimate = [x for x in estimate_row]
	groundtruth = [x for x in groundtruth_row]
	wrongness = []
	a = []
	b = []
	size = numRows * numCols
	for i in range(size):
		if estimate[i] and groundtruth[i]:
			num_correct = min(estimate[i],groundtruth[i])
			for j in range(num_correct):
				wrongness.append(0)
			estimate[i] -= num_correct
			groundtruth[i] -= num_correct
		if estimate[i] > 0:
			a.append([i,estimate[i]])
		if groundtruth[i] > 0:
			b.append([i,groundtruth[i]])
	while len(a)>0 and len(b)>0:
		a0 = a[0]
		dist_list = [dist_func(a0[0],x[0]) for x in b]
		min_dist = min(dist_list)
		index = dist_list.index(min_dist)
		b0 = b[index]
		quantity = min(a0[1],b0[1])
		a0[1] -= quantity
		if a0[1] == 0:
			a.pop(0)
		b0[1] -= quantity
		if b0[1] == 0:
			b.pop(index)
		for j in range(quantity):
			wrongness.append(min_dist)
	return wrongness



def deviate_direction(estimate, groundtruth):
	'''calculate a list of direction difference, one for each cell that has movements originated from that cell'''
	degrees = []
	size = numRows*numCols
	for i in range(size):
		total = sum(groundtruth[i])
		if total>0:
			estimate_direction = average_normalised_direction(estimate[i],i)
			groundtruth_direction = average_normalised_direction(groundtruth[i],i)
			degrees.append(getAngle(estimate_direction,groundtruth_direction))
	return degrees


def average_normalised_direction(row, i):
	'''the sum of all vectors starting from one cell'''
	size = numRows*numCols
	vector_sum = [0,0]
	for j in range(size):
		if row[j]>0 and i!=j:
			vector = unit_vector(i, j)
			vector_sum[0] += row[j]*vector[0]
			vector_sum[1] += row[j]*vector[1]
	return vector_sum

def unit_vector(start,end):
	start_row, start_col = index_to_rowcol(start)
	end_row, end_col = index_to_rowcol(end)
	distance = math.sqrt((end_row-start_row)**2+(end_col-start_col)**2)
	return float(end_row-start_row)/distance, float(end_col-start_col)/distance

def getAngle(vector1,vector2):
	'''calculate direction difference between two vectors'''
	if vector1 == vector2:
		return 0
	if vector1[0]==0.0 and vector1[1]==0.0 or vector2[0] == 0.0 and vector2[1]==0.0:
		return 10000.0 
	degree1 = math.atan2(vector1[1],vector1[0])
	degree2 = math.atan2(vector2[1],vector2[0])
	delta_radian = degree2 - degree1
	while delta_radian <= 0:
		delta_radian += 2* math.pi 
	while delta_radian > 2*math.pi:
		delta_radian -= 2*math.pi 
	if delta_radian > math.pi:
		delta_radian = 2*math.pi - delta_radian
	return delta_radian 

def degree_accuracy(degrees,benchmark):
	'''apply benchmark to get the proportion of cells that has a direction deviation less than the benchmark'''
	size = len(degrees)
	correct = 0
	for degree in degrees:
		if degree < benchmark:
			correct += 1
	return float(correct)/size


def output_accuracy(t):
	'''performed without network constraint'''
	folder = folder_check(t)
	print "Test on " + folder
	if not os.path.exists(folder):
		print "Folder not found."
		return False
	else:
		groundtruth_file = folder + '/groundtruth.csv'
		withoutnetwork_file = folder + '/odMatrix_without2.csv'

		start_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[0]))+".asc"
		start_raster = read_rasterserface(start_raster_file)
		end_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[1]))+".asc"
		end_raster = read_rasterserface(end_raster_file)
		groundtruth_matrix = read_groundtruth(groundtruth_file)

		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
		baseline = greedy_baseline(start_raster,end_raster)
		lp_accuracy = calculate_accuracy(withoutnetwork_matrix,groundtruth_matrix)
		bl_accuracy = calculate_accuracy(baseline,groundtruth_matrix)
		lp_dist_accuracy = distance_accuracy(withoutnetwork_matrix,groundtruth_matrix,euclidean)
		bl_dist_accuracy = distance_accuracy(baseline,groundtruth_matrix,euclidean)
		lp_hss = calculate_hss(withoutnetwork_matrix,groundtruth_matrix)
		bl_hss = calculate_hss(baseline,groundtruth_matrix)
		dist_hss = distance_hss(withoutnetwork_matrix,groundtruth_matrix,euclidean)

		lp_wrongness = degree_of_wrongness(withoutnetwork_matrix,groundtruth_matrix)
		bl_wrongness = degree_of_wrongness(baseline,groundtruth_matrix)
		lp_degrees = deviate_direction(withoutnetwork_matrix,groundtruth_matrix)
		lp_deg_acc = degree_accuracy(lp_degrees,30.0/180*math.pi)
		bl_degrees = deviate_direction(baseline,groundtruth_matrix)
		bl_deg_acc = degree_accuracy(bl_degrees,30.0/180*math.pi)

		flow_accu = calculate_flow_accuracy(withoutnetwork_matrix,groundtruth_matrix)
		bl_flow_accu = calculate_flow_accuracy(baseline,groundtruth_matrix)
		flow_hss = calculate_flow_hss(withoutnetwork_matrix,groundtruth_matrix)
		bl_flow_hss = calculate_flow_hss(baseline,groundtruth_matrix)
		lp_flow_dist_accuracy = distance_flow_accuracy(withoutnetwork_matrix,groundtruth_matrix,euclidean)
		bl_flow_dist_accuracy = distance_flow_accuracy(baseline,groundtruth_matrix,euclidean)
		flow_wrongness = degree_of_flow_wrongness(withoutnetwork_matrix,groundtruth_matrix)
		bl_flow_wrongness =degree_of_flow_wrongness(baseline,groundtruth_matrix)
		# return (np.mean(lp_accuracy),np.mean(bl_accuracy),stats.ttest_ind(lp_accuracy,bl_accuracy)[-1],
		# 	np.mean(lp_hss),np.mean(bl_hss),stats.ttest_ind(lp_hss,bl_hss)[-1],
		# 	np.mean(lp_wrongness),np.mean(bl_wrongness),stats.ttest_ind(lp_wrongness,bl_wrongness)[-1],
		# 	lp_deg_acc,bl_deg_acc,np.mean(lp_dist_accuracy),np.mean(bl_dist_accuracy),
		# 	flow_accu,bl_flow_accu)
		return (flow_accu, bl_flow_accu,0,
			flow_hss, bl_flow_hss, 0,
			flow_wrongness, bl_flow_wrongness,0,
			lp_deg_acc,bl_deg_acc,
			lp_flow_dist_accuracy, bl_flow_dist_accuracy
			)

def revised_accuracy_test(t,type):
	'''performed without network constraint'''
	folder = folder_check(t)
	print "Test on " + folder
	if not os.path.exists(folder):
		print "Folder not found."
		return False
	else:
		groundtruth_file = folder + '/groundtruth.csv'
		groundtruth = read_groundtruth(groundtruth_file)
		groundtruth_list_file = folder + '/groundtruth_list.csv'
		true_groundtruth = read_groundtruth_list(groundtruth_list_file)

		start_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[0]))+".asc"
		start_raster = read_rasterserface(start_raster_file)
		end_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[1]))+".asc"
		end_raster = read_rasterserface(end_raster_file)

		withoutnetwork_file = folder + '/odMatrix_without.csv'
		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
		baseline = greedy_baseline(start_raster,end_raster)

		if type == "cell":
			pass
			


def output_distance_accuracy(t):
	'''performed without network constraint'''
	folder = folder_check(t)
	print "Test on " + folder
	if not os.path.exists(folder):
		print "Folder not found."
		return False
	else:
		groundtruth_file = folder + '/groundtruth.csv'
		withoutnetwork_file = folder + '/odMatrix_without2.csv'
		start_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[0]))+".asc"
		start_raster = read_rasterserface(start_raster_file)
		end_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[1]))+".asc"
		end_raster = read_rasterserface(end_raster_file)
		groundtruth_matrix = read_groundtruth(groundtruth_file)
		clear_selfflow(groundtruth_matrix)
		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
		baseline = greedy_baseline(start_raster,end_raster)
		clear_selfflow(baseline)
		return calculate_groupped_accuracy(withoutnetwork_matrix,groundtruth_matrix)


def output_dichotomous_accuracy(t):
	'''performed without network constraint'''
	folder = folder_check(t)
	print "Test on " + folder
	if not os.path.exists(folder):
		print "Folder not found."
		return False
	else:
		groundtruth_file = folder + '/groundtruth.csv'
		withoutnetwork_file = folder + '/odMatrix_without2.csv'
		# withnetwork_file = folder + '/odMatrix_network.csv'
		start_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[0]))+".asc"
		start_raster = read_rasterserface(start_raster_file)
		end_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[1]))+".asc"
		end_raster = read_rasterserface(end_raster_file)
		groundtruth_matrix = read_groundtruth(groundtruth_file)
		clear_selfflow(groundtruth_matrix)
		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
		# withnetwork_matrix = make_ODmatrix(withnetwork_file)
		# calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
		# calc_selfflow(start_raster,end_raster,withnetwork_matrix)
		# baseline = baseline_solution(start_raster,end_raster)
		baseline = greedy_baseline(start_raster,end_raster)
		clear_selfflow(baseline)
		baseline_accuracy = dichotomous_accuracy_direct(baseline,groundtruth_matrix)
		withoutnetowrk_accuracy = dichotomous_accuracy_direct(withoutnetwork_matrix,groundtruth_matrix)
		# withnetwork_accuracy = dichotomous_accuracy_direct(withnetwork_matrix,groundtruth_matrix)
		return np.mean(withoutnetowrk_accuracy), np.mean(baseline_accuracy)



if __name__ == '__main__':
	with open(base_dir+'accuracy2.csv','w') as f:
		for t in tests_frames:
			line = str(diff_in_sec(*t)) + ','
			x = output_accuracy(t)
			line += ','.join([str(a) for a in x])+'\n'
			f.write(line)
	with open(base_dir+'grouped_accuracy2.csv','w') as f:
		t = tests_frames[-1]
		accu_dict = output_distance_accuracy(t)
		for key in accu_dict.keys():
			if accu_dict[key] != 0:
				f.write(str(key)+','+str(accu_dict[key])+'\n')
	with open(base_dir + 'dichotomous_acc2.csv','w') as f:
		for t in tests_frames:
			line = str(diff_in_sec(*t)) + ','
			x = output_dichotomous_accuracy(t)
			line += ','.join([str(a) for a in x])+'\n'
			f.write(line)

# if __name__ == '__main__':
# 	frame_test("raster1001010020")

# def road_accuracy(folder):
# 	start_raster_file = folder+'/start.asc'
# 	start_raster = read_rasterserface(start_raster_file)
# 	end_raster_file = folder+'/end.asc'
# 	end_raster = read_rasterserface(end_raster_file)
# 	groundtruth_file = folder+'/groundtruth.csv'
# 	withoutnetwork_file = folder+'/odMatrix_without.csv' 
# 	withnetwork_file =folder+'/odMatrix_network.csv'
# 	groundtruth_matrix = read_groundtruth(groundtruth_file)
# 	clear_selfflow(groundtruth_matrix)
# 	withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
# 	# calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
# 	withnetwork_matrix = make_ODmatrix(withnetwork_file)
# 	# calc_selfflow(start_raster,end_raster,withnetwork_matrix)
# 	accuracy = calculate_flow_accuracy(withoutnetwork_matrix,groundtruth_matrix)
# 	network_accuracy = calculate_flow_accuracy(withnetwork_matrix,groundtruth_matrix)
# 	return accuracy, network_accuracy	

# steps = (3,5,7,9)
# sets = 10
# if __name__ == '__main__':
# 	with open('roads_accu.csv','w') as f:
# 		for step in steps:
# 			without_accu =[]
# 			with_accu = []
# 			for i in range(sets):
# 				folder = 'roads/backup/set'+str(i)+'/'+str(step)
# 				x, y = road_accuracy(folder)
# 				without_accu.append(x)
# 				with_accu.append(y)
# 			f.write(str(step*10)+','+str(np.mean(without_accu))+','+str(np.mean(with_accu))+','+str(stats.ttest_ind(without_accu,with_accu)[-1])+'\n')







