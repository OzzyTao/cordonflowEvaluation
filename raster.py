import psycopg2
import math
import os

# boundary_ul = (-0.1348, 51.5544)
# boundary_ul = (-0.2483, 51.5576)
boundary_ul = (-0.3549,51.5749)
# boundary_ll = (0, 20)
boundary_size = 0.2
# boundary_size = 20
numCells = 40
# result_folder = "raster2"
# sample_time = [[8,0],[8,15],[8,20],[8,25],[8,30],[9,0],[9,30],[10,0],[11,0],[11,30],[11,45],[11,46],[11,47],[11,50],[11,55],[12,0]]
tests_frames = [[[10,0,10],[10,0,20]]
,[[10,0,10],[10,0,40]]
,[[10,0],[10,1]]
, [[10,0],[10,2]]
, [[10,0],[10,3]]]


tablename = 'ecourier'
data_of_id_sql = '''
	select * from {tablename}
	where v_id = {id} and date = {dayofyear}
	order by time;
'''

def compare_time(time1,time2):
	'''return 1 when time1 is after time2'''
	num_parts = len(time1)
	for i in range(num_parts):
		if time1[i] > time2[i]:
			return 1
		elif time1[i] < time2[i]:
			return -1 
	return 0

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

def cord_to_raster(longitude, latitude):
	# return row, col of a raster
	step = boundary_size / numCells
	if boundary_ul[0] <= longitude <= boundary_ul[0] + boundary_size and boundary_ul[1] - boundary_size <= latitude <= boundary_ul[1]:
		col = (longitude - boundary_ul[0])/step
		row = (boundary_ul[1]-latitude)/step
		return row, col
	return -1, -1

def process_one_trajectory(v_id, doy, conn, sample_time, rasters, trajectories):
	flags = [False]*len(sample_time)
	cells = [(-1,-1)]*len(sample_time)
	index = 0
	with conn.cursor() as cur:
		cur.execute(data_of_id_sql.format(tablename=tablename,id=v_id,dayofyear=doy))
		table = cur.fetchall()
		for line in table:
			if index >= len(sample_time):
				break
			time_read = (line[6],line[7]) if len(sample_time[index]) == 2 else (line[6],line[7],line[8]/10*10)
			if compare_time(time_read,sample_time[index])==0:
				if flags[index] == False: 
					row, col = cord_to_raster(line[2], line[3])
					cells[index]=(row,col)
					flags[index] = True
					index += 1
			elif compare_time(time_read,sample_time[index])==1:
				index += 1
	is_valid = (-1,-1) not in cells
	if is_valid:
		print cells
		trajectories.append(cells)
		for i in range(len(cells)):
			row, col = cells[i]
			if row >=0 and col >=0:
				rasters[i][int(row)][int(col)] += 1
	return is_valid


def rowcol_to_index(row, col):
	return row * numCells + col

def list_to_flowmatrix(list_stripped):
	size = numCells * numCells
	matrix = [[0 for i in range(size)] for j in range(size)]
	for record in list_stripped:
		from_index = rowcol_to_index(int(record[0][0]),int(record[0][1]))
		to_index = rowcol_to_index(int(record[1][0]),int(record[1][1]))
		matrix[from_index][to_index] += 1 
	return matrix

def flush_raster(time, raster,result_folder):
	file_name = result_folder+"/" +"".join(map(lambda x: "%02d" % (x), time))+".asc"
	with open(file_name,'w') as f:
		f.write("ncols\t%d\n" % (numCells))
		f.write("nrows\t%d\n" % (numCells))
		f.write("xllcorner\t0\n")
		f.write("yllcorner\t0\n")
		f.write("cellsize\t1\n")
		f.write("NODATA_value\t-99\n")
		for i in range(numCells):
			row = i
			for col in range(numCells):
				f.write(str(raster[row][col])+'\t')
			f.write('\n')
	
def flush_groundtruth(trajectories,result_folder):
	matrix = list_to_flowmatrix(trajectories)
	with open(result_folder+"/groundtruth.csv",'w') as f:
		for row in matrix:
			out_put = ''
			for cell in row:
				if out_put:
					out_put += ','
				out_put += str(cell)
			out_put += '\n'
			f.write(out_put)

def flush_true_groundtruth_list(trajectories,result_folder):
	with open(result_folder + "/groundtruth_list.csv",'w') as f:
		for record in trajectories:
			source, target = record
			out_put = ''
			out_put += str(source[0]) + ',' + str(source[1]) +','
			out_put += str(target[0]) + ',' + str(target[1]) + '\n'
			f.write(out_put)

def folder_check(time_frame):
	folder_name = 'test40_TG/raster'
	for time in time_frame:
		for timepart in time:
			folder_name += str(timepart)
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
		return folder_name
	return False


def perform_test(time_frame):
	folder = folder_check(time_frame)
	if folder:
		rasters = []
		for i in range(len(time_frame)):
			raster = []
			for j in range(numCells):
				row = []
				for k in range(numCells):
					row.append(0)
				raster.append(row)
			rasters.append(raster)
		trajectories = []
		vaild_instance = 0
		conn = db_connection()
		for vid in range(170):
			for day in range(1,367):
				if process_one_trajectory(vid,day,conn, time_frame,rasters,trajectories):
					vaild_instance += 1
		print str(vaild_instance) + "trajectories found."
		
		for i in range(len(time_frame)):
			flush_raster(time_frame[i],rasters[i],folder)
		flush_groundtruth(trajectories,folder)
		flush_true_groundtruth_list(trajectories,folder)

if __name__ == '__main__':
	for time_frame in tests_frames:
		perform_test(time_frame)






