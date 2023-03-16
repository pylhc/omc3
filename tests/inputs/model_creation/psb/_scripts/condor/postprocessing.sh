#!/bin/bash

cd /afs/cern.ch/eng/acc-models/psb/2021

# remove HTCondor output files
find . -name "*.out" -type f -delete
find . -name "*.err" -type f -delete
find . -name "*.log" -type f -delete

# delete tfs and create pkl instead
#/usr/bin/python3 ./_scripts/create_tfs_pkl.py

# create tune control optics files
rm operation/tune_control/*.str
/usr/bin/python3 ./_scripts/create_tune_control_optics.py

# create JMAD configuration file
/usr/bin/python3 ./_scripts/create_XML.py

# create knob file
/usr/bin/python3 ./_scripts/create_knob_file.py

git add -A .
git commit -m 'Automatic update of optics files created on HTCondor.'
git push origin HEAD
