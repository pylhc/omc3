#!/bin/bash

cd /afs/cern.ch/eng/acc-models/ps/2021

# remove HTCondor output files
find . -name "*.out" -type f -delete
find . -name "*.err" -type f -delete
find . -name "*.log" -type f -delete

# transform tfs to pkl instead
# /usr/bin/python3 ./_scripts/create_tfs_pkl.py

# create JMAD configuration file
/usr/bin/python3 ./_scripts/create_XML.py

git add -A .
git commit -m 'Automatic update of optics files created on HTCondor.'
git push origin HEAD
