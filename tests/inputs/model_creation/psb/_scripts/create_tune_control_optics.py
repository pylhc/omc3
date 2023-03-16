# import tfs

# folder = 'operation/tune_control/'
# table = tfs.read(folder + 'tune_control.tfs')

# for index, row in table.iterrows():
# 	filename = 'psb_qx_' + str(row.Q1) + '_qy_' + str(row.Q2) + '.str'
# 	with open(folder + filename, 'w') as f:
# 		f.write('kbrqf = ' + str(row.KBRQF) + ';\n')
# 		f.write('kbrqd = ' + str(row.KBRQD) + ';')

from cpymad.madx import Madx
import tfs
import numpy as np
import glob
from shutil import copyfile

#--------- nominal tune control optics ------------

folder = 'operation/tune_control/'
table = tfs.read(folder + 'tune_control.tfs')

# create strengths files from tune control table
madx = Madx()
madx.call('_scripts/macros.madx')

for index, row in table.iterrows():

	filename = folder + 'psb_qx_{:0.3f}_qy_{:0.3f}.str'.format(row['Q1'], row['Q2'])

	for k in row.keys()[2:]:
	    madx.input('{} = {};'.format(k, row[k]))

	madx.input("exec, write_str_file('{}')".format(filename))

#--------- injection tune control optics ------------

folder = 'operation/tune_control/'
table = tfs.read(folder + 'tune_control_injection.tfs')

# create strengths files from tune control table
madx = Madx()
madx.call('_scripts/macros.madx')

for index, row in table.iterrows():

	filename = folder + 'psb_inj_qx_{:0.3f}_qy_{:0.3f}.str'.format(row['Q1'], row['Q2'])

	for k in row.keys()[2:]:
	    madx.input('{} = {};'.format(k, row[k]))

	madx.input("exec, write_str_file('{}')".format(filename))

# append tunes to ext str file names and copy them to tune control directory
# for injection files, only use the LHC and AD tunes to complete the tune control injection optics 
# (and remove the lhc/ad label for consistency)
for j, config in enumerate(['inj_ad', 'inj_lhc', 'ext']):
	str_files = sorted(glob.glob(f'scenarios/*/*/*{config}*.str'))
	tfs_files = sorted(glob.glob(f'scenarios/*/*/*{config}*.tfs'))

	if len(str_files) == 1:
		twiss = tfs.read(tfs_files[0])
		filename = folder + "psb_inj_qx_{:0.3f}_qy_{:0.3f}.str".format(twiss.MU1.iloc[-1], twiss.MU2.iloc[-1])
		copyfile(str_files[0], filename)
	else:
		for i,file_ in enumerate(str_files):
			twiss = tfs.read(tfs_files[i])
			filename = folder + "{}_qx_{:0.3f}_qy_{:0.3f}.str".format(file_[:-4].split('/')[-1], twiss.MU1.iloc[-1], twiss.MU2.iloc[-1])
			copyfile(file_, filename)
