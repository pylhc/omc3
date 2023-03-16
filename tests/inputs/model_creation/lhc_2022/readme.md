General usage
====================

The madx user will then need to write

```
git clone -b 2022 https://gitlab.cern.ch/acc-models/acc-model-lhc acc-model-lhc
```
or
```
ln -s /afs/cern.ch/eng/acc-models/lhc/2022 acc-model-lhc
```
or
```
ln -s /eos/project/a/acc-models/lhc/2022 acc-model-lhc
```

A full model can be obtained for example by

```
call,file="acc-models-lhc/scenarios/pp_lumi/RAMP-SQUEEZE-6.8TeV-ATS-1.3m_V1/0/model.madx";
```

which uses:

```
call, file="acc-models-lhc/lhc.seq";
beam, ....
call,file="acc-models-lhc/operation/optics/R2022a_A11mC11mA10mL10m.madx";
call,file="acc-models-lhc/scenarios/pp_lumi/RAMP-SQUEEZE-6.8TeV-ATS-1.3m_V1/0/settings.madx";
```

All madx scripts internal or external uses  `acc-model-lhc` as prefix.


Metadata
==========

- `operation/knobs.txt` connection madx - lsa knobs
- `scenarios/cycles.txt` list of LHC cycles
- `scenarios/<cycle>/beam_processes.txt` list of beam processes considered of the scenarios








