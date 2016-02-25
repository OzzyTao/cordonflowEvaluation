numRows = 40
numCols = 40
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


if __name__ == '__main__':
	raster = [[0 for i in range(numCols)] for j in range(numRows)]
	flush_raster(raster,"test40_TG/pcells.asc")