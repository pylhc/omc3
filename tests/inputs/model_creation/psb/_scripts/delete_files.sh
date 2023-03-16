#!/bin/bash

# remove HTCondor output files
find . -name "*.out" -type f -delete
find . -name "*.err" -type f -delete
find . -name "*.log" -type f -delete

# delete tfs and create pkl instead
# python3 ./_scripts/create_tfs_pkl.py

# create tune control optics files
python3 ./_scripts/tune_control_optics.py

# create JMAD configuration file
python3 ./_scripts/create_XML.py

# additional commands for reference
# find . -name "*.tfs" -type f -delete
# find . -name "*.html" -type f -delete
# find . -name "*.pdf" -type f -delete
# find . -name "debug.ptc" -type f -delete
# find . -name "*.tfs" -type f -delete
# find . -name "*.ipynb" -type f -delete
# find . -name "index.md" -type f -delete
# find . -name "internal_mag_pot.txt" -type f -delete

# find . -name "*.madx_job" -exec sed -i '' 's,"../../../,",' {} +
# find . -name "*.madx_job" -exec sed -i '' 's,"./,",' {} +
# find . -name "*.madx*" -exec sed -i '' 's, PC = 2.14, PC = 2.80,' {} +
# find . -name "*.madx*" -exec sed -i '' 's, PC = 2.80, PC = 2.794987,' {} +
# find . -name "*.madx*" -exec sed -i '' '#call, file="../../../ps.str";#d' {} +
# find . -name "*inj*.madx*" -exec sed -i '' 's, ../../2019, 16/03/2020,' {} +
# find . -name "*fb*.madx*" -exec sed -i '' 's, ../../2019, 16/03/2020,' {} +
# find . -name "*.madx*" -exec sed -i '' '/ps.str/d' {} +
# find . -name "*.madx_job" -exec sed -i '' 's,("ps,("./output/ps,' {} +

# insert 'system,  "mkdir output";' at the top of each file
# find . -iname "*.madx_job" -type f -exec sed -i '' '1 i\ 
# system, "mkdir output";\
# \
# ' {} \;

# find . -name "*.madx*" -exec sed -i '' 's"exec, write_str"/******************************************************************\
# * Knobs for bump at the shavers\
# ******************************************************************/\
# \
# exec, shaver_bump_knob_factors();\
# \
# exec, shaver_bump_knobs();\
# \
# exec, write_str"' {} +