import glob
import tfs
import os

files = glob.glob('*/*/*/*.tfs')

for f in files:
	data = tfs.read(f)
	data.to_pickle(f[:-3] + 'pkl')
	os.remove(f)