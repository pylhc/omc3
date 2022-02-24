# OMC3 Changelog

#### 2022-02-23

- Fixed:
  - The coupling calculation now includes additional columns in the output dataframes, which were missing while being needed later on by the correction calculation.

#### 2022-01-10

- Fixed:
  - Sequences for K-Modulation are now included in PyPi package
  - Bug fixed where default errors in K-Modulation would not have been taken into account

#### 2021-07-14 
_by jdilly_

- Added:
  - global correction framework
    - global correction entrypoint
    - response matrix creation in madx and analytically
    - response read/write functions in hdf5
    - madx sequence evaluation
    - model appenders
    - model differences functions
  - script to fake measurement from model
  - more usages of constants for strings (e.g. column names)
  - introducing pathlib.Path in some places
  - output-paths in model job-files are relative
  
- Fixes:
  - Matplotlib warnings for `set_window_title`
  - excluded Windows and MacOS py3.9 from normal testing, due to installation issues of pyTables
  - model creation accepts relative and absolute paths


#### 2020-09-30

- Added:
  - script to merge kmod results and calculate imbalance
  - fixture for temporary/non-temporary test output folder
  - scripts to documentation

#### 2020-07-27

- Added:
  - tfs-plotter
  - optics-measurements plotter

#### 2020-03-04

- Added:
   - lin-file natural tune updater

#### 2020-02-24

- Added:
   - amplitude detuning analysis
   - amplitude detuning and bbq plotting
   - time tools
   - plotting helpers
- Distinction between `BasicTests` and `Extended Tests`

#### Before 2020-02

- Updated and moved main functionalities from python 2.7
    - Madx wrapper
    - Frequency Analysis of turn by turn
    - Optics measurement analysis scripts
    - Accelerator class and Model Creator
    - K-mod
    - Spectrum Plotting
    - Turn-by-Turn Converter

- `setup.py` and packaging functionality 
- Automated CI
    - Multiple versions of python
    - Accuracy tests
    - Unit tests
    - Release automation