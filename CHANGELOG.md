# OMC3 Changelog

#### 2022-06-21 - v0.4.1 - _jdilly_, _fesoubel_

- Fixed:
  - LHC ATS macros called for all years after 2018
  - LHC RunIII macros called for all years after 2022
  - Getting new BBQ data ended in a key-error.
  - Better KeyError Message for Exciter-BPM not found.

#### 2022-05-30 - v0.4.0 - _jdilly_

- Added: 
  - 2D amplitude detuning analysis and 3D plotting of the results
  - Converter for amp.det. analysis from bbs to omc3
  - general typehinting/doc/unification of entrypoint parameters/saving

- Fixed:
  - Switched action-tune planes in ampdet-kick file header-names
  - Deprecated way of pandas indexing (`.loc` for nearest points)
  - Allow different sized kick-files for amp.det. analysis

#### 2022-05-19 - v0.3.0 - _jdilly_

- Added:
  - Linfile cleaning script. 

#### 2022-04-25 - v0.2.7 - _awegshe_

- Added:
  - There is now an option, `coupling_pairing`, for the BPM pairing in coupling calculation, to let the user choose the number of BPMs instead of the usual "best candidate" method.

#### 2022-04-25 - v0.2.6 - _awegsche_

- Fixed:
  - Only perform index merging on the `NAME` column during coupling calculation. This solves an (at the moment) un-understood issue where some BPMs would have different `S` values in different files.

#### 2022-04-12 - v0.2.5 - _awegsche_

- Fixed:
  - An additional knob and macros definition file has been added to reflect the knobs used by OP in the LHC Run 3. This makes sure any `omc3.model_creator` run for the LHC with `year >= 2022` has correct knobs.

#### 2022-04-07 - v0.2.4 - _fsoubelet_

- Miscellaneous:
  - The jpype1 package is not a default dependency anymore, and is instead included as a dependency in the cern and test extras. Its import is mocked where needed in omc3.

#### 2022-02-23 - v0.2.3 - _awegsche_

- Fixed:
  - The coupling calculation now includes additional columns in the output dataframes, which were missing while being needed later on by the correction calculation.

#### 2022-01-10 - v0.2.2 - _mihofer_

- Fixed:
  - Sequences for K-Modulation are now included in PyPi package
  - Bug fixed where default errors in K-Modulation would not have been taken into account

#### 2022-01-10 - v0.2.1 - _fsoubelet_

- _Dummy Release for Zenodo_

#### 2021-07-14 - v0.2.0 - _jdilly_

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