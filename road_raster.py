import psycopg2
import math
import random
import os

upperRoad = [22761,18271]
lowerRoad = [1484, 22763]
numCols = 10
numRows = 3

def db_connection():
	main_dt = {
	'db' : 'gis',
	'ip' : 'localhost',
	'port' : '5432',
	'password' : 'peach',
	'user' : 'postgres'
	}
	conn = psycopg2.connect('''host={ip} dbname={db} user={user} password={password} port={port}'''.format(**main_dt))
	return conn 

def locate_point(pid, direction, conn):
	cur = conn.cursor()
	query = '''select st_linelocatepoint(keyroads.geom, ecourier.geom) from keyroads, ecourier where keyroads.direction='%s' and ecourier.id = %s; '''
	cur.execute(query % (direction,pid))
	fraction = cur.fetchone()[0]
	col = int(math.floor(fraction*10))
	if col>9:
		col = 9
	if direction == 'left':
		return [0,col]
	else:
		return [2, 9-col]


def gen_trajectory(records,conn):
	if records[0][1] > records[-1][1]:
		direction = "left"
	else:
		direction = "right"
	trajectory = [locate_point(record[0],direction,conn) for record in records]
	return trajectory


def rowcol_to_index(row, col):
	return row * numCols + col

def gen_trajectories(v_id,conn):
	trajectories = []
	cur = conn.cursor()
	query = '''select ecourier.id,ecourier.longitude, extract(epoch from time) from ecourier, keyroads 
	where st_distance(ecourier.geom, keyroads.geom) < 0.0005 and v_id = %s
	order by time;'''
	cur.execute(query % v_id)
	records = cur.fetchall()
	if len(records)>2:
		i =0 
		pre_time = records[0][2]
		j = 1 
		while j<len(records):
			if records[j][2] - pre_time < 60:
				j += 1 
			else:
				trajectories.append(gen_trajectory(records[i:j],conn))
				i =j
				pre_time = records[i][2]
				j += 1 
		trajectories.append(gen_trajectory(records[i:],conn))
	return  trajectories

def all_trajectories(conn):
	trajectories = []
	cur = conn.cursor()
	query = "select distinct v_id from ecourier;"
	cur.execute(query)
	records = cur.fetchall()
	ids = [x[0] for x in records]
	for i in ids:
		trajectories += gen_trajectories(i,conn)
	return trajectories

def simulate(trajectories, interval):
	size = 3*10
	start_raster = [[0 for i in range(10)] for j in range(3)]
	end_raster = [[0 for i in range(10)] for j in range(3)]
	ground_truth = [[0 for i in range(size)] for j in range(size)]
	for trajectory in trajectories:
		if len(trajectory) > interval:
			random_index = random.randint(0,len(trajectory)-interval-1)
			start_location = trajectory[random_index]
			end_location = trajectory[random_index+interval]
			# print start_location, end_location
			start_raster[start_location[0]][start_location[1]] += 1
			end_raster[end_location[0]][end_location[1]] += 1 
			ground_truth[rowcol_to_index(*start_location)][rowcol_to_index(*end_location)] += 1

	return start_raster, end_raster, ground_truth

def flush_raster(raster,file_name):
	with open(file_name,'w') as f:
		f.write("ncols\t%d\n" % (numCols))
		f.write("nrows\t%d\n" % (numRows))
		f.write("xllcorner\t0\n")
		f.write("yllcorner\t0\n")
		f.write("cellsize\t1\n")
		f.write("NODATA_value\t-99\n")
		for i in range(numRows):
			row = i
			for col in range(numCols):
				f.write(str(raster[row][col])+'\t')
			f.write('\n')

def flush_groundtruth(matrix,file_name):
	with open(file_name,'w') as f:
		for row in matrix:
			out_put = ''
			for cell in row:
				if out_put:
					out_put += ','
				out_put += str(cell)
			out_put += '\n'
			f.write(out_put)

def folder_check(folder_name):
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
		return folder_name
	return False

if __name__ == '__main__':
	conn = db_connection()
	trajectories = all_trajectories(conn)
	tests = 10
	for steps in (3,5,7,9,11):
		folder = 'roads/'+str(steps)
		folder_check(folder)
		for i in range(tests):
			start_raster, end_raster, ground_truth = simulate(trajectories, steps)
			folder = 'roads/'+str(steps)+'/test'+str(i)
			folder_check(folder)
			flush_raster(start_raster,folder+'/start.asc')
			flush_raster(end_raster,folder+'/end.asc')
			flush_groundtruth(ground_truth,folder+'/groundtruth.csv')






