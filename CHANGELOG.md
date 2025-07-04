# OMC3 Changelog

#### 2025-04-28 - v0.24.4 - _wvangoet_

- Fixed:
  - Correct element regex and fixes for SuperKEKB.

#### 2025-04-28 - v0.24.3 - _jdilly_

- Fixed:
  - `kmod_lumi_imbalance` added IPs to output names.

#### 2025-04-21 - v0.24.2 - _jdilly_

- Fixed:
  - `knob_extractor` added new knobs for 2025.

#### 2025-04-04 - v0.24.1 - _jdilly_

- Added:
  - Corrector lists for 2025.
  - Added `kqt13.l7b1` to the main `MQT` correctors list.
  - `lhc_corrector_list_check`: Checks LHC-sequence against the corrector lists.

#### 2025-04-03 - v0.24.0 - _jdilly_

- Added:
  - Additional Propagables: Dispersion (but not really working [#498](https://github.com/pylhc/omc3/issues/498)), Coupling (only forward working [#498](https://github.com/pylhc/omc3/issues/498])).

#### 2025-04-01 - v0.23.01 - _jdilly_, _jgray_

- Changed:
  - `twiss_elements` via macro now contain all elements in the sequence.

- Fixed:
  - `check_corrections` does not error when different markers in the model and the measurement.

- Added:
  - Bad BPMs summary: NANS reason.

#### 2025-03-13 - v0.23.0 - _jdilly_

- Added:
  - SPS model creation (Nominal, Segment and Correction).

#### 2025-03-06 - v0.22.1 - _jdilly_

- Fixed:
  - Error in list b2 error choices for lhc best knowledge models.

#### 2025-03-05 - v0.22.0 - _jdilly_

- Added:
  - Segment-by-Segment with phase-propagation.
  - Listing of entry-points on calling `python -m omc3`, `python -m omc3.scripts` and `python -m omc3.plotting`.
  - Measurement from model faker now also adds alpha (ALF) columns to the beta-output files.

- Updated:
  - MAD-X binaries to version 5.09.00.

#### 2025-03-03 - v0.21.1 - _fscarlier_

- Fixed:
  - Correct function call in `omc3.kmod_importer` when running as script.
  - Fixed k-mod import for single-IP imports.

#### 2025-01-23 - v0.21.0 - _jdilly_, _fscarlier_, _fesoubel_

- Fixed:
  - Plot Optics Measurements: Added extra mpl style for clearer plots.
  - LHC exciter BPM not found: Tells you which BPMs were searched for.
  - Plot Spectrum: Correct error handling for Single-Plane BPMs.

- Added:
  - A `generic` accelerator class, that can be used for non-specifc accelerators.
  - Global Correction: Total arc phase correction (`arc-by-arc`).
  - Cleaning: Filter BPMs with NaNs.
  - Cleaning: Log bad BPMs with reasons before raising errors.
  - MAD-X wrapper: Inform about failed twiss.

#### 2025-01-06 - v0.20.4 - _fsoubelet_

- Fixed:
  - Solved an issue in datetime operations occuring on `Python 3.13`.

- Changed:
  - Dropped support for `Python 3.9`.

#### 2024-11-21 - v0.20.3 - _jdilly_

- Fixed:
  - `analyse_dpp` issues in `beta_from_amplitude`, `chromatic_beating` and `dispersion`.
    Skips `dispersion` calculation if `NaN`s are present.

#### 2024-11-21 - v0.20.2 - _jdilly_, _awegsche_

- Added:
  - `bad_bpms_summary`: Also collect the reasons for the BPMs being bad.

#### 2024-11-14 - v0.20.1 - _jdilly_

- Fixed:
  - `kmod_import` fixed issue with too harsh filtering of BPM data.

#### 2024-11-14 - v0.20.0 - _jdilly_, _awegsche_

- Added:
  - `bad_bpms_summary` script: Collect and summarize the bad BPMs from GUI runs.

#### 2024-11-13 - v0.19.0 - _fscarlier_, _jdilly_

- Added K-Modulation tools:
  - `kmod_importer`: Main function to call the following scripts in succession.
  - `kmod_averages`: Combine k-modulation results into a single file.
  - `kmod_lumi_imbalance`: Calculate the luminosity imbalance IP1 / IP5 from k-modulation results.
  - `kmod_import`: Import k-modulation results into an `optics_measurements` directory as `beta_kmod_().tfs` file.

- Removed:
  - Old k-modulation scripts: Functionality is either in k-mod GUI or in newly added scripts.

#### 2024-11-11 - v0.18.1 - _jdilly_

- Fixed:
  - Setting sparkprops wrong in `knob_extractor` fixed.

#### 2024-10-29 - v0.18.0 - _jgray_

- Added:
  - Add the ability to calculate a deltap/p offset caused by a change in orbit.
  - Add the ability to use `response_madx` to calculate the updated response matrix for the global correction.
  - Tests for the calculation of the deltap/p and corresponding correction.
  - Tests for the calculation of the updated response matrix for the global correction.

- Fixed:
  - Fixed the `response_twiss` when updating the response matrix when calculating the global correction.

#### 2024-10-29 - v0.17.0 - _jdilly_

- Added:
  - Optics Measurements: `analyse_dpp` parameter, to analyse only files with a specific DPP for tune, phase and (C)RDTs.

- Fixed:
  - Uncompensated Phase calculations with `Equation` compensation are now actually uncompensated when passed on to the `coupling` module.

#### 2024-10-28 - v0.16.2 - _jdilly_

- Fixed:
  - Temporary hack to fix `knob_extractor` in CCC.

#### 2024-09-20 - v0.16.1 - _fsoubelet_

- Fixed:
  - Fixed `DepracationWarning`s related datetime operations.
  - Fixed `DeprecationWarning` occuring due to the use of old `numpy` functions.
  - Fixed `FutureWarning` happening during edge-cases of dataframe concatenation by performing checks ahead of time.
  - Fixed `FutureWarning`s occuring due to deprecated `pandas.Series` accesses.
  - Fixed `UserWarning` occuring when wrongly setting ticks and labels for correction plots.

- Changed:
  - Masked `NaturalNameWarning`s happening during HDF5 tables operations, as the use of names such as `kq4.r8b2` is not avoidable and `pandas` properly handles access operations for us.
  - Masked `UserWarning`s happening during plotting for operations that are explicitely demanded.
  - Intercept `RankWarning` which can happen during a `polyfit` of data and re-emit as log message.
  - Intercept `OptimizeWarning` happening when the covariance parameters could not be estimated in `kmod` analysis and re-emit as log message.
  - Intercept `OptimizeWarning` happening when the covariance parameters could not be estimated in `rdt` analysis and re-emit as log message.

#### 2024-09-18 - v0.16.0 - _jdilly_

- Added:
  - Global Correction for LHC:
    - The correction variables in the LHC accelerator class are now handled differently internally,
      allowing new variable classes to be added to each lhc-year and user-given files in the model-directory.
    - Variable categories `MQM_ALL` added to all LHC years.
    - Variable categories `MQM_INJ_2024` and `MQM_TOP_2024` added to LHC 2024.
    - Adding a "-" in front of a given correction variable name removes this variable from the correction. Does not work for whole variable categories.
  - Tests for running `global_correction` with `omp` and `pinv` correction methods.

- Fixed:
  - Orthogonal Matching Pursuit (`omp`) in global correction runs again ([#448](https://github.com/pylhc/omc3/issues/448))
  - Corrections Check window stop-iteration issue: fixed. Single-plane files, i.e. normalized dispersion, now accepted ([#447](https://github.com/pylhc/omc3/issues/447))
  - LHC Correction variables now logged, if not found in the variable categories ([#446](https://github.com/pylhc/omc3/issues/446))

#### 2024-09-16 - v0.15.4 - _jdilly_

- Fixed:
  - Measure optics skips now using the ERRAMP column, when not present, i.e. when cleaning deactivated ([#451](https://github.com/pylhc/omc3/issues/451))
  - `hole_in_one` now allows `pathlib.Path` objects in addition to `str` ([#452](https://github.com/pylhc/omc3/issues/452))
  - Pandas to numpy dtype conversions bug ([#453](https://github.com/pylhc/omc3/issues/453)).
  - Special phases writing now skipped when accelerator has no special phases ([#454](https://github.com/pylhc/omc3/issues/454)).
  - RDT/CRDT calculation now not crashing when also giving off-momentum files; but only calculated from on on-momentum files ([#456](https://github.com/pylhc/omc3/issues/456)).

- Added:
  - Tests for full runs `hole_in_one` with on-momentum and off-momentum files.

#### 2024-08-14 - v0.15.3 - _jdilly_

- Fixed:
  - Add DOROS BPMs to `twiss.dat`.
  - Some Pandas `FutureWarning`s, `DeprecationWarning`s and `PerformanceWarning`s

#### 2024-08-14 - v0.15.2 - _fesoubel_, _jdilly_

- Fixed:
  - Numpy's `ComplexWarning` was not part of main namespace in v2.0, so we import it directly

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

- CI:
  - Dropped python 3.8

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

- Fixed:
  - Changed all `pandas`/`tfs-pandas` `append()` and `join()` to `concat()`
  - Moved `InputFiles` into separate module

#### 2023-03-16 - v0.7.2 - _jdilly_

- Fixed:
  - Added missing columns to coupling in BBS-OMC3 converter

#### 2023-01-20 - v0.7.1 - _jdilly_

- Added:
  - Amplitude Detuning plots: Switch to plot only with/without BBQ correction.

- Fixed:
  - Second Order Amplitude Detuning fit now working
  - Correct print/calculation of second order direct terms for forced kicks in plot-labels.

#### 2022-11-08 - v0.7.0 - _jdilly_

- Added:
  - Tune error based on deviation of filtered BBQ data to the moving average
    (over moving average window)
  - Action error calculated from error on the spectral line
    (which in turn is the same as NOISE)
  
#### 2022-11-01 - v0.6.6

- Bugfixes:
  - correction: fullresponse is converted to Path.
  - fake measurement from model: dont randomize errors and values by default.

#### 2022-10-15 - v0.6.5

- Added to `knob_extractor`:
  - proper state extraction.
  - IP2 and IP8 separation/crossing variables.

#### 2022-10-12 - v0.6.4

- Fixed:
  - phase filtering for coupling calculation will not forget columns

#### 2022-09-27 - v0.6.3

- Added:
  - pandafied `knob_extractor` internally and python output

#### 2022-09-22 - v0.6.2

- Added:
  - cleaned logging in `knob_extractor`

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

- Added:
  - updated to turn_by_turn v0.4.0: Includes SPS reader

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
  
- Fixed:
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
