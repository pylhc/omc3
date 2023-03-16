folder = '/afs/cern.ch/eng/acc-models/psb/'
branch = '2021/'
folder += branch
filename = folder + 'operation/knobs.txt'

with open(filename, 'w') as f:
	print('# MadX name, LSA name, scaling, test trim', file = f)
	print('# KSW bump', file = f)
	for i in range(4):
		print('bi{}ksw_x_mm, PSBBEAM{}/BIKSW_POSITION_X_MM#bump, 1.0, 1.0'.format(i+1,i+1), file = f)
		print('bi{}ksw_x_mm, PSBBEAM{}/BIKSW_POSITION_X_MM#injectionEnd, 1.0, 1.0'.format(i+1,i+1), file = f)
		print('bi{}ksw_x_mm, PSBBEAM{}/BIKSW_POSITION_X_MM#paintingEnd, 1.0, 1.0'.format(i+1,i+1), file = f)
	print('\n# Shaver bumps', file = f)
	for i in range(4):
		print('shaverr{}_x_mm, PSBBEAM{}/SHAVER_POSITION_X_MM, 1.0, 1.0'.format(i+1,i+1), file = f)
		print('shaverr{}_y_mm, PSBBEAM{}/SHAVER_POSITION_Y_MM, 1.0, 1.0'.format(i+1,i+1), file = f)
	print('\n# Extraction bump', file = f)
	print('bebsw_x_mm,    PSBBEAM/BEBSW_POSITION_X_MM,    1.0, 1.0', file = f)
	# print('bebsw_px_urad, PSBBEAM/BEBSW_ANGLE_X_URAD, 1.0, 100.', file = f)
	print('\n# Extraction bump correction', file = f)
	for i in range(4):
		print('be{}dhz_x_mm,    PSBBEAM{}/BEDHZ_POSITION_X_MM,    1.0, 1.0'.format(i+1,i+1), file = f)
		print('be{}dhz_px_urad, PSBBEAM{}/BEDHZ_ANGLE_X_URAD, 1.0, 100.'.format(i+1,i+1), file = f)
		print('be{}dvt_y_mm,    PSBBEAM{}/BEDVT_POSITION_Y_MM,    1.0, 1.0'.format(i+1,i+1), file = f)
		print('be{}dvt_py_urad, PSBBEAM{}/BEDVT_ANGLE_Y_URAD, 1.0, 100.'.format(i+1,i+1), file = f)