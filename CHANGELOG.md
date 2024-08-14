# OMC3 Changelog

#### 2024-08-14 - v0.15.1 - _fesoubel_

- Fixed:
  - The package is now fully compatible with `numpy 2.x` on `Python >= 3.10` thanks to a `pytables` compatibility release.
  - The package still limits to `numpy < 2` on `Python 3.9` due to the lack of compatibility from `pytables` on this versions.

#### 2024-07-08 - v0.15.0 - _jdilly_

- PINNING NUMPY TO < 2.0.0

- Changed:
  - Model creation:
    - removed hard-coded `knobs.madx` from `lhc`
    - removed `corrections.madx` from `lhc` best-knowledge model
    - zip up log-output files in `response_madx.py`
    - keep 0th output file in `response_madx.py` for reference of the model setup
    - Sequence and modifiers use the acc-models symlink in madx-jobs where applicable.

#### 2024-06-05 - v0.14.1 - _jdilly_

- Fixed:
  - LHC Knobs: Fixed typo "MQSOR" to "MQSORG" in LHC Beam 2 coupling knobs.

- CI: Dropped python 3.8

#### 2024-03-18 - v0.14.0 - _jdilly_

- Added:
  - Linfile Updater: `keep`-flag to keep names and option to clean manually between limits.

#### 2024-03-08 - v0.13.1 - _jdilly_, _awegsche_, _mlegarre_, _fesoubel_

- Added:
  - Knob Extractor: Lumiscan Knob

- Fixed:
  - BBS converter: fixed closed orbit units
  - Optics: Pandas indexing error in DPP

#### 2023-12-07 - v0.13.0 - _awegsche_

- Added:
 - complete overhaul of model creation, uses now `acc-models` for LHC, PS and PSB and prints
   useful information about available model parameters. Can load from either a user defined path
   (`--path <PATH>`) or from the afs copy of acc-models (`--afs`)

#### 2023-11-29 - v0.12.1 - _jdilly_

- Fixed:
 - `tbt_converter` now also passes given output format to writer when running without noise.

#### 2023-11-29 - v0.12.0 - _jdilly_

- Added to harmonic analysis:
    - `suffix` input parameter: adds suffix to output files, which e.g. allows running the same file 
    with different parameters without overwriting it.
    - `bunch_ids` input parameter: in case of multibunch-files only analyse these bunches.
    If not given, all bunches will be analysed, as before.

#### 2023-09-20 - v0.11.4 - _fscarlier_, _awegsche_
- Fixed:
    - bug in beta from phase (3BPM method) that calculated too high errors for first and last BPMs

#### 2023-09-20 - v0.11.3 - _jdilly_, _awegsche_

- Fixed:
  - compatibility with matplotlib 3.8
  - skipping important phase advances for HL-LHC (as not defined yet)
  - allowing responses with delta_k < 1e-6 for full-response creation

#### 2023-09-01 - v0.11.2 - _jdilly_

- Fixed:
  - Plot Optics: making normalized dispersion plot a special case.

- Added:
 - Plot Optics: optional input "--labels" to manually set the legend-labels. 

#### 2023-06-16 - v0.11.1 - _jdilly_

- Fixed:
 - OptionalString: 'None' as input is converted to None.
 - Missing Kerberos config added to MANIFEST for packaging.
 - Plot Optics plots now correct error-column, e.g. for beta-beating.
 - Added warnings/errors for too few bpms in N-BPM/3-BPM methods. 
 - Added navbar to sphinx documentation.

- Tests:
 - Added test for the classes in omc3.plotting.utils.windows

#### 2023-06-05 - v0.11.0 - _jdilly_

- Added:
 - `omc3.plotting.utils.windows`: Qt-based windows and widgets for matplotlib-figure organization.
 - Using the new windows in `omc3.plotting.plot_checked_corrections` and `omc3.plotting.plot_tfs`

#### 2023-05-15 - v0.10.0 - _jdilly_

- Added:
  - `omc3.check_corrections`: A new feature to check the validity of corrections.
  - `omc3.plotting.plot_checked_corrections`: Function to plot the checked corrections.
  - Unified optics-columns naming in `omc3.definitions.optics`
    (but not yet propagated through the code)
  - Function to calculate RMS in `omc3.utils.stats.rms`.

- Fixed:
  - Some minor bugs with fake datatypes
  - Doc of `ArrayType` typehints

#### 2023-04-27 - v0.9.0 - _jdilly_

- Added:
  - RBAC token provider in omc3.utils.rbac
  - pylogbook wrapper in omc3.scripts.create_logbook_entry

#### 2023-04-20 - v0.8.0 - _jdilly_

- Fix:
  - Changed all `pandas`/`tfs-pandas` `append()` and `join()` to `concat()`
  - Moved `InputFiles` into separate module

#### 2023-03-16 - v0.7.2 - _jdilly_

- Fix: 
  - Added missing columns to coupling in BBS-OMC3 converter

#### 2023-01-20 - v0.7.1 - _jdilly_

- Added:
  - Amplitude Detuning plots: Switch to plot only with/without BBQ correction 

- Fix: 
  - Second Order Amplitude Detuning fit now working
  - Correct print/calculation of second order direct terms for forced 
    kicks in plot-labels.

#### 2022-11-08 - v0.7.0 - _jdilly_

- Added:
  - Tune error based on deviation of filtered BBQ data to the moving average
    (over moving average window)
  - Action error calculated from error on the spectral line
    (which in turn is the same as NOISE)
  
#### 2022-11-01 - v0.6.6

- Bugfixes 
  - correction: fullresponse is converted to Path.
  - fake measurement from model: dont randomize errors and values by default. 

#### 2022-10-15 - v0.6.5

- Added to `knob_extractor`:
  - proper state extraction. 
  - IP2 and IP8 separation/crossing variables.

#### 2022-10-12 - v0.6.4

- Fixed the phase filtering for coupling calculation to not forget columns.

#### 2022-09-27 - v0.6.3

- Pandafied `knob_extractor` internally and python output.

#### 2022-09-22 - v0.6.2

- Cleaned logging in `knob_extractor`

#### 2022-09-21 - v0.6.1

- Added: 
  - tbt output datatype for converter.

#### 2022-09-20 - v0.6.0

- Added:
  - The `knob_extractor` script to get LHC knob values from `NXCALS` at a given time

#### 2022-09-19 - v0.5.2

- Bugfix:
  - Correction Dataframe initialized as float (before as int)

- Added:
  - Plotting: Transposed legend order
  - Plotting: Create markers from any text

#### 2022-09-12 - v0.5.1

- Updated to turn_by_turn v0.4.0: Includes SPS reader

#### 2022-07-25 - v0.5.0 - _Mael-Le-Garrec_

- Added:
  - The resonance lines can now be sought and detected up to arbitrary order during the frequency analysis, with the `resonances` argument / flag of `harpy`.
  - The RDT components can now be calculated up to arbitrary order in the optics measurements with the `rdt_magnet_order` argument / flag of `optics`. Note that the relevant resonance lines for this order should have been detected by `harpy` beforehand.

#### 2022-06-21 - v0.4.1 - _jdilly_, _fesoubel_

- Fixed:
  - Fixed macros and knobs usage in model_creator for Run 3 optics
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
