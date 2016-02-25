import os
import sys
import csv
from random import shuffle
from scipy import stats
import numpy as np

numCells = 20
base_dir = 'simulation/'
object_nums = [5,10,15,20]

def manhanton(i, j):
	return abs(i[0]-j[0])+abs(i[1]-j[1])

def folder_check(time_frame):
	folder_name = base_dir+'raster'
	for time in time_frame:
		for timepart in time:
			folder_name += str(timepart)
	return folder_name

def rowcol_to_index(row, col):
	return row * numCells + col

def index_to_rowcol(i):
	row = i/numCells
	col = i%numCells
	return row, col 

def inflow(row,col,matrix):
	result = 0
	size = numCells*numCells
	dindex = rowcol_to_index(row,col)
	for i in range(size):
		if i != dindex:
			result += matrix[i][dindex]
	return result

def outflow(row,col,matrix):
	result = 0
	size = numCells*numCells
	oindex = rowcol_to_index(row,col)
	for i in range(size):
		if i != oindex:
			result += matrix[oindex][i]
	return result


def read_rasterserface(file_name):
	matrix = []
	with open(file_name,'r') as f:
		for line in f:
			if line[0].isdigit():
				matrix.append(map(lambda x: int(x),filter(lambda x : not x.isspace(),line.split("\t"))))
	return matrix

def calc_selfflow(start_raster, end_raster, odmatrix):
	for i in range(numCells):
		for j in range(numCells):
			odindex = rowcol_to_index(i,j)
			odmatrix[odindex][odindex] = min(start_raster[i][j]-outflow(i,j,odmatrix),end_raster[i][j]-inflow(i,j,odmatrix))



def make_ODmatrix(file_name):
	size = numCells * numCells
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
	size = numCells * numCells
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
	size = numCells*numCells
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



# multi-category forecasts
def calculate_accuracy(estimate, groundtruth):
	accu_sum = []
	size = numCells*numCells
	for i in range(size):
		total_moves = 0
		correct_moves = 0
		for j in range(size):
			total_moves += groundtruth[i][j]
			correct_moves += min(estimate[i][j],groundtruth[i][j])
		if total_moves:
			accu_sum.append(float(correct_moves)/total_moves)
	return accu_sum


def calculate_hss(estimate, groundtruth):
	size = numCells*numCells
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

# grouped by movement distance
def calculate_groupped_accuracy(estimate,groundtruth):
	total_moves = {}
	hits = {}
	size = numCells * numCells
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
def distance_accuracy(estimate,groundtruth):
	accu_sum =[]
	size = numCells * numCells
	for i in range(size):
		orowcol = index_to_rowcol(i)
		e = {}
		g = {}
		for j in range(size):
			drowcol = index_to_rowcol(j)
			dist = manhanton(orowcol,drowcol)
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

def distance_hss(estimate, groundtruth):
	hss_sum = []
	size = numCells * numCells
	for i in range(size):
		orowcol = index_to_rowcol(i)
		e = {}
		g = {}
		for j in range(size):
			drowcol = index_to_rowcol(j)
			dist = manhanton(orowcol,drowcol)
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
	'''return OD values based on random assignment'''
	size = numCells * numCells
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
	size = numCells * numCells
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




def read_groundtruth(file_name):
	matrix = []
	with open(file_name,'r') as f:
		table = csv.reader(f)
		for row in table:
			matrix.append([int(x) for x in row])
	return matrix

def frame_test(t):
	folder = folder_check(t)
	print "Test on " + folder
	if not os.path.exists(folder):
		print "Folder not found."
		return False
	else:
		groundtruth_file = folder + '/groudtruth.csv'
		withnetwork_file = folder + '/odMatrix_network.csv'
		withoutnetwork_file = folder + '/odMatrix_without.csv'
		start_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[0]))+".asc"
		start_raster = read_rasterserface(start_raster_file)
		end_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[1]))+".asc"
		end_raster = read_rasterserface(end_raster_file)
		groundtruth_matrix = read_groundtruth(groundtruth_file)
		withnetwork_matrix = make_ODmatrix(withnetwork_file)
		calc_selfflow(start_raster,end_raster,withnetwork_matrix)
		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
		withnetwork_acc = calculate_accuracy(withnetwork_matrix,groundtruth_matrix)
		withoutnetwork_acc = calculate_accuracy(withoutnetwork_matrix, groundtruth_matrix)
		print "Accuracy with network info is " + str(withnetwork_acc)
		print "Accuracy without network info is " + str(withoutnetwork_acc)
	return withnetwork_acc, withoutnetwork_acc

def statistics(t):
	'''performed without network constraint'''
	folder = folder_check(t)
	print "Test on " + folder
	if not os.path.exists(folder):
		print "Folder not found."
		return False
	else:
		groundtruth_file = folder + '/groundtruth.csv'
		withoutnetwork_file = folder + '/odMatrix_without.csv'
		start_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[0]))+".asc"
		start_raster = read_rasterserface(start_raster_file)
		end_raster_file = folder+"/" +"".join(map(lambda x: "%02d" % (x), t[1]))+".asc"
		end_raster = read_rasterserface(end_raster_file)
		groundtruth_matrix = read_groundtruth(groundtruth_file)
		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
		baseline = baseline_solution(start_raster,end_raster)
		# dichotomous test
		print "dichotomous tests:"
		baseline_contingency = dichotomous_contingency_table(baseline,groundtruth_matrix)
		my_contingency = dichotomous_contingency_table(withoutnetwork_matrix,groundtruth_matrix)
		print "Accuracy: " + str(dichotomous_accuracy(my_contingency)) + '    ' + str(dichotomous_accuracy(baseline_contingency))
		print "Bias: " + str(dichotomous_bias(my_contingency)) + '    ' + str(dichotomous_bias(baseline_contingency))
		print "ETS: " + str(dichotomous_ets(my_contingency)) + '    ' + str(dichotomous_ets(baseline_contingency))
		print "HK: " +str(dichotomous_hk(my_contingency))+ '    ' + str(dichotomous_hk(baseline_contingency))
		print "HSS: " +str(dichotomous_hss(my_contingency))+ '    ' + str(dichotomous_hss(baseline_contingency))
		print "ORSS: " + str(dichotomous_orss(my_contingency)) + '    ' + str(dichotomous_orss(baseline_contingency))
		# multi-category test
		print "multi-category test:"
		print "Accuracy: " + str(calculate_accuracy(withoutnetwork_matrix,groundtruth_matrix)) + '    ' + str(calculate_accuracy(baseline,groundtruth_matrix))
		print "HSS: " + str(calculate_hss(withoutnetwork_matrix,groundtruth_matrix)) + '    ' + str(calculate_hss(baseline,groundtruth_matrix))
		# intuitive test
		print "intuitive test:"
		baseline_contingency = intuitive_contingency_table(baseline,groundtruth_matrix)
		my_contingency = intuitive_contingency_table(withoutnetwork_matrix,groundtruth_matrix)
		print "Accuracy: " + str(intuitive_accuracy(my_contingency)) + '    ' + str(intuitive_accuracy(baseline_contingency))
		# grouped accuracy test
		print "accuracy statistics grouped by distance:"
		print "Accuracy: " + str(calculate_groupped_accuracy(withoutnetwork_matrix,groundtruth_matrix))
		# multi-category test: based on distance
		print "Accuracy distance category:"
		print "Accuracy: " + str(distance_accuracy(withoutnetwork_matrix,groundtruth_matrix))


def output_accuracy(t):
	'''performed without network constraint'''
	folder = base_dir + str(t)
	print "Test on " + folder
	if not os.path.exists(folder):
		print "Folder not found."
		return False
	else:
		groundtruth_file = folder + '/groudtruth.csv'
		withoutnetwork_file = folder + '/odMatrix_without.csv'
		start_raster_file = folder+"/start.asc"
		start_raster = read_rasterserface(start_raster_file)
		end_raster_file = folder+"/end.asc"
		end_raster = read_rasterserface(end_raster_file)
		groundtruth_matrix = read_groundtruth(groundtruth_file)
		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
		# baseline = baseline_solution(start_raster,end_raster)
		baseline = greedy_baseline(start_raster,end_raster)
		lp_accuracy = calculate_accuracy(withoutnetwork_matrix,groundtruth_matrix)
		print lp_accuracy
		bl_accuracy = calculate_accuracy(baseline,groundtruth_matrix)
		dist_accuracy = distance_accuracy(withoutnetwork_matrix,groundtruth_matrix)
		# lp_accuracy = calculate_hss(withoutnetwork_matrix,groundtruth_matrix)
		# bl_accuracy = calculate_hss(baseline,groundtruth_matrix)
		# dist_accuracy = distance_hss(withoutnetwork_matrix,groundtruth_matrix)
		return np.mean(lp_accuracy),np.mean(bl_accuracy),np.mean(dist_accuracy),stats.ttest_ind(lp_accuracy,bl_accuracy)[-1],stats.ttest_ind(lp_accuracy,dist_accuracy)[-1]

def output_distance_accuracy(t):
	'''performed without network constraint'''
	folder = base_dir +str(t)
	print "Test on " + folder
	if not os.path.exists(folder):
		print "Folder not found."
		return False
	else:
		groundtruth_file = folder + '/groudtruth.csv'
		withoutnetwork_file = folder + '/odMatrix_without.csv'
		start_raster_file = folder+"/start.asc"
		start_raster = read_rasterserface(start_raster_file)
		end_raster_file = folder+"/end.asc"
		end_raster = read_rasterserface(end_raster_file)
		groundtruth_matrix = read_groundtruth(groundtruth_file)
		withoutnetwork_matrix = make_ODmatrix(withoutnetwork_file)
		calc_selfflow(start_raster,end_raster,withoutnetwork_matrix)
		return calculate_groupped_accuracy(withoutnetwork_matrix,groundtruth_matrix)





def diff_in_sec(t1, t2):
	secs = ((t2[0]-t1[0])*60 + (t2[1]-t1[1]))*60
	if len(t1) == 3:
		secs += t2[2]-t1[2]
	return secs

if __name__ == '__main__':
	# with open('result.csv','w') as f:
	# 	for t in tests_frames:
	# 		line = str(diff_in_sec(*t)) + ','
	# 		a,b = frame_test(t)
	# 		line += str(a) + ',' + str(b) +'\n'
	# 		f.write(line)
	# print "success."
	with open(base_dir+'accuracy.csv','w') as f:
		for num in object_nums:
			line = str(num) + ','
			x = output_accuracy(num)
			line += ','.join([str(a) for a in x])+'\n'
			f.write(line)
	with open(base_dir+'grouped_accuracy.csv','w') as f:
		t = object_nums[-1]
		accu_dict = output_distance_accuracy(t)
		for key in accu_dict.keys():
			if accu_dict[key] != 0:
				f.write(str(key)+','+str(accu_dict[key])+'\n')

# if __name__ == '__main__':
# 	frame_test("raster1001010020")


