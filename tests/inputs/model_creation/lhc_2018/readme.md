General usage
====================

The madx user will then need to write

```
git clone -b 2018 https://gitlab.cern.ch/acc-models/acc-model-lhc acc-model-lhc
```
or
```
ln -s /afs/cern.ch/eng/acc-models/lhc/RunII/2018 acc-model-lhc
```
or
```
ln -s /eos/project/a/acc-models/lhc/RunII/2018 acc-model-lhc
```

A full model can be obtained by

```
call,file="acc-model-lhc/lhc.seq";
call,file="acc-model-lhc/operation/optics/R2017a_A740.madx";
```

All madx scripts internal or external uses  `acc-model-lhc` as prefix.

<!-- Plan for the future!

call,file="acc-model-lhc/operation/settings/6125/RAMPSQUEEZE_595.madx";   

The list of optics used during a fill are listed in
```
operation/settings/<fill_number>/settings_list.txt
```

To get a full model, after having consulted `2018/operation/settings/595/settings_list.txt`
-->


Metadata
=====

- `operation/knobs.txt` connection madx - lsa knobs








