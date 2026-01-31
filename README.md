# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/pylhc/omc3/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                    |    Stmts |     Miss |   Cover |   Missing |
|-------------------------------------------------------- | -------: | -------: | ------: | --------: |
| omc3/\_\_init\_\_.py                                    |        8 |        0 |    100% |           |
| omc3/\_\_main\_\_.py                                    |       14 |       14 |      0% |      1-22 |
| omc3/amplitude\_detuning\_analysis.py                   |      150 |       32 |     79% |366, 406-424, 436, 443, 447, 452, 502-507, 514-519, 526-528, 535, 566 |
| omc3/check\_corrections.py                              |      176 |       18 |     90% |413, 419, 428, 433, 451, 456, 460, 470, 488-489, 531, 565-566, 715-720, 760, 778 |
| omc3/correction/\_\_init\_\_.py                         |        0 |        0 |    100% |           |
| omc3/correction/arc\_by\_arc.py                         |       65 |       17 |     74% |94-108, 142-146, 187-188, 199-200 |
| omc3/correction/constants.py                            |       17 |        0 |    100% |           |
| omc3/correction/filters.py                              |       86 |       13 |     85% |138, 170-177, 191, 226-231 |
| omc3/correction/handler.py                              |      189 |        8 |     96% |85, 95, 221, 236, 270, 282, 349, 356 |
| omc3/correction/model\_appenders.py                     |       62 |        0 |    100% |           |
| omc3/correction/model\_diff.py                          |       40 |        1 |     98% |        40 |
| omc3/correction/response\_io.py                         |       63 |        2 |     97% |    58, 86 |
| omc3/correction/response\_madx.py                       |      154 |       10 |     94% |83, 316-317, 324-330 |
| omc3/correction/response\_twiss.py                      |      396 |       55 |     86% |240, 244-246, 253, 288-290, 296-306, 401-402, 458-459, 493-535, 703, 707-715, 727, 739, 742, 745-747, 871 |
| omc3/correction/sequence\_evaluation.py                 |      149 |       11 |     93% |68, 137, 199-200, 220-226, 306 |
| omc3/definitions/\_\_init\_\_.py                        |        1 |        0 |    100% |           |
| omc3/definitions/constants.py                           |        9 |        0 |    100% |           |
| omc3/definitions/formats.py                             |        8 |        0 |    100% |           |
| omc3/definitions/optics.py                              |      152 |        6 |     96% |127, 136-139, 158 |
| omc3/global\_correction.py                              |       65 |        2 |     97% |  320, 387 |
| omc3/harpy/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| omc3/harpy/clean.py                                     |      151 |       15 |     90% |42, 90, 113, 188, 204-205, 215-216, 240-247 |
| omc3/harpy/constants.py                                 |       26 |        0 |    100% |           |
| omc3/harpy/frequency.py                                 |      196 |        7 |     96% |151, 238, 507, 524-526, 546 |
| omc3/harpy/handler.py                                   |      126 |        1 |     99% |       301 |
| omc3/harpy/kicker.py                                    |       27 |        0 |    100% |           |
| omc3/hole\_in\_one.py                                   |      194 |       10 |     95% |333, 335, 373, 456, 482, 498, 500, 502, 504, 881 |
| omc3/kmod\_importer.py                                  |       71 |        4 |     94% |287-293, 308 |
| omc3/knob\_extractor.py                                 |      160 |       20 |     88% |77-80, 390-391, 395-396, 400-401, 506-519, 577, 605, 609 |
| omc3/madx\_wrapper.py                                   |      103 |       12 |     88% |28, 30, 71, 96-97, 208-209, 212, 216-217, 223, 229 |
| omc3/model/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| omc3/model/accelerators/\_\_init\_\_.py                 |        1 |        0 |    100% |           |
| omc3/model/accelerators/accelerator.py                  |      193 |       24 |     88% |146, 184, 190-192, 203, 229, 245-252, 261, 271, 273, 289, 297, 303, 326, 339, 347, 355, 400 |
| omc3/model/accelerators/esrf.py                         |       10 |        3 |     70% |     70-72 |
| omc3/model/accelerators/generic.py                      |       10 |        0 |    100% |           |
| omc3/model/accelerators/iota.py                         |       23 |        8 |     65% |83-91, 94-97, 101 |
| omc3/model/accelerators/lhc.py                          |      167 |       23 |     86% |177, 190, 199, 239-242, 244, 246, 282-297, 315, 328, 334, 354-358 |
| omc3/model/accelerators/petra.py                        |       11 |        3 |     73% | 22-23, 27 |
| omc3/model/accelerators/ps.py                           |       47 |        9 |     81% |112, 124, 129-135 |
| omc3/model/accelerators/psbase.py                       |       20 |        0 |    100% |           |
| omc3/model/accelerators/psbooster.py                    |       44 |       11 |     75% |112, 119, 123-124, 127-133 |
| omc3/model/accelerators/skekb.py                        |       34 |        3 |     91% |112, 123-124 |
| omc3/model/accelerators/sps.py                          |       39 |        0 |    100% |           |
| omc3/model/constants.py                                 |       36 |        0 |    100% |           |
| omc3/model/manager.py                                   |       23 |        2 |     91% |    52, 63 |
| omc3/model/model\_creators/\_\_init\_\_.py              |        1 |        0 |    100% |           |
| omc3/model/model\_creators/abstract\_model\_creator.py  |      201 |       22 |     89% |96, 122, 130, 156, 262-267, 385-387, 392, 395, 400, 411, 524-537, 541 |
| omc3/model/model\_creators/lhc\_model\_creator.py       |      272 |       23 |     92% |98, 103, 177, 305, 358-359, 363-364, 375, 388-389, 395-396, 403, 407, 428-434, 450, 455, 500, 590 |
| omc3/model/model\_creators/manager.py                   |       27 |        4 |     85% |81-82, 86-87 |
| omc3/model/model\_creators/ps\_base\_model\_creator.py  |       39 |       11 |     72% |34, 39-55, 83-86, 89, 95-96 |
| omc3/model/model\_creators/ps\_model\_creator.py        |       33 |        3 |     91% | 62, 69-70 |
| omc3/model/model\_creators/psbooster\_model\_creator.py |       27 |        0 |    100% |           |
| omc3/model/model\_creators/sps\_model\_creator.py       |       78 |       14 |     82% |49, 56, 70-81, 91, 94-110, 115, 247 |
| omc3/model\_creator.py                                  |       59 |       12 |     80% |136-148, 185 |
| omc3/mqt\_extractor.py                                  |       38 |       18 |     53% |162-183, 188-191, 195 |
| omc3/nxcals/\_\_init\_\_.py                             |        1 |        0 |    100% |           |
| omc3/nxcals/knob\_extraction.py                         |      105 |       80 |     24% |102-153, 181-226, 243-247, 266-288, 304, 330-344 |
| omc3/nxcals/mqt\_extraction.py                          |       25 |       17 |     32% |46-51, 77-83, 96-103 |
| omc3/optics\_measurements/\_\_init\_\_.py               |        1 |        0 |    100% |           |
| omc3/optics\_measurements/beta\_from\_amplitude.py      |       63 |        0 |    100% |           |
| omc3/optics\_measurements/beta\_from\_phase.py          |      283 |        9 |     97% |129-132, 135, 246, 375, 397-398, 490, 601 |
| omc3/optics\_measurements/chromatic.py                  |       44 |       14 |     68% |     61-79 |
| omc3/optics\_measurements/constants.py                  |       90 |        0 |    100% |           |
| omc3/optics\_measurements/coupling.py                   |      160 |        4 |     98% |225, 241, 254-255 |
| omc3/optics\_measurements/crdt.py                       |       83 |        0 |    100% |           |
| omc3/optics\_measurements/data\_models.py               |      115 |       15 |     87% |58, 65, 86, 113, 119, 164, 169, 172, 205-219 |
| omc3/optics\_measurements/dispersion.py                 |      158 |       47 |     70% |79, 97, 112-113, 129-144, 162, 169-170, 176-177, 191-210, 244-253, 291-292 |
| omc3/optics\_measurements/dpp.py                        |       74 |       14 |     81% |102, 105, 118-130 |
| omc3/optics\_measurements/iforest.py                    |       67 |       46 |     31% |35-40, 44-47, 52-58, 62-63, 67-77, 81-90, 94-100, 104, 108-112 |
| omc3/optics\_measurements/interaction\_point.py         |       45 |        0 |    100% |           |
| omc3/optics\_measurements/kick.py                       |       72 |        4 |     94% |63-64, 130-131 |
| omc3/optics\_measurements/measure\_optics.py            |       86 |       14 |     84% |73, 110-114, 145, 175-183 |
| omc3/optics\_measurements/phase.py                      |      157 |       11 |     93% |95, 127, 217-220, 223-226, 228, 234 |
| omc3/optics\_measurements/rdt.py                        |      188 |        9 |     95% |172-179, 360, 389-393 |
| omc3/optics\_measurements/toolbox.py                    |       36 |        0 |    100% |           |
| omc3/optics\_measurements/tune.py                       |       38 |        0 |    100% |           |
| omc3/plotting/\_\_init\_\_.py                           |        0 |        0 |    100% |           |
| omc3/plotting/\_\_main\_\_.py                           |        8 |        8 |      0% |      1-12 |
| omc3/plotting/optics\_measurements/\_\_init\_\_.py      |        0 |        0 |    100% |           |
| omc3/plotting/optics\_measurements/constants.py         |        6 |        0 |    100% |           |
| omc3/plotting/optics\_measurements/utils.py             |       53 |        1 |     98% |        65 |
| omc3/plotting/plot\_amplitude\_detuning.py              |      312 |       26 |     92% |257, 288-290, 322-323, 367, 397, 400, 442-444, 472, 482-483, 526, 560, 581, 636, 643, 680, 716, 741, 744, 751, 778 |
| omc3/plotting/plot\_bbq.py                              |      104 |       19 |     82% |179-188, 197, 251, 254, 259-260, 263, 268, 272, 279-280, 299 |
| omc3/plotting/plot\_checked\_corrections.py             |      163 |       10 |     94% |271-272, 334, 405-406, 441-442, 453, 542, 548 |
| omc3/plotting/plot\_kmod\_results.py                    |      109 |        9 |     92% |156, 164, 174, 183, 212-216, 224, 241-242 |
| omc3/plotting/plot\_optics\_measurements.py             |      170 |       10 |     94% |387, 402, 414, 425, 433, 490, 532, 536, 585, 611 |
| omc3/plotting/plot\_spectrum.py                         |      122 |        3 |     98% |376, 381, 492 |
| omc3/plotting/plot\_tfs.py                              |      217 |       15 |     93% |347, 546-559, 621, 680 |
| omc3/plotting/spectrum/\_\_init\_\_.py                  |        0 |        0 |    100% |           |
| omc3/plotting/spectrum/stem.py                          |       60 |        2 |     97% |   52, 112 |
| omc3/plotting/spectrum/utils.py                         |      252 |       19 |     92% |177, 180, 259, 318, 321, 381-382, 385, 404, 500-509 |
| omc3/plotting/spectrum/waterfall.py                     |       59 |       10 |     83% |48, 58, 64, 75, 81-86, 104 |
| omc3/plotting/utils/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| omc3/plotting/utils/annotations.py                      |      166 |      122 |     27% |60-76, 86-88, 100-131, 136-143, 153-156, 169, 172-173, 184-189, 203-218, 231-239, 249-258, 271-279, 284, 290-304, 312, 315, 331-332, 349-351, 354, 357-359, 374-384, 402 |
| omc3/plotting/utils/colors.py                           |       32 |       10 |     69% |30, 36-41, 52, 75-77 |
| omc3/plotting/utils/lines.py                            |       44 |       14 |     68% |31, 44-46, 62-73, 86-87, 117 |
| omc3/plotting/utils/style.py                            |       21 |        2 |     90% |    43, 51 |
| omc3/plotting/utils/windows.py                          |       83 |       20 |     76% |46-53, 60, 139-140, 180, 197-205 |
| omc3/response\_creator.py                               |       39 |        1 |     97% |       160 |
| omc3/sbs\_propagation.py                                |      156 |       32 |     79% |172-175, 215, 218, 233-236, 266, 303, 309, 327, 332, 336-337, 340, 359-361, 384-389, 399, 431-432, 435, 450 |
| omc3/scripts/\_\_init\_\_.py                            |        0 |        0 |    100% |           |
| omc3/scripts/\_\_main\_\_.py                            |        8 |        8 |      0% |      1-12 |
| omc3/scripts/bad\_bpms\_summary.py                      |      185 |       13 |     93% |215-216, 220-221, 250, 258, 287, 327-329, 459-460, 470 |
| omc3/scripts/betabeatsrc\_output\_converter.py          |      141 |       19 |     87% |165-166, 204-205, 213-214, 249-250, 291-292, 331-332, 366-367, 400-401, 440-441, 471 |
| omc3/scripts/create\_logbook\_entry.py                  |       89 |       24 |     73% |72, 154-155, 163-186, 197, 217, 220, 226, 242, 253-268, 278 |
| omc3/scripts/fake\_measurement\_from\_model.py          |      256 |        3 |     99% |545, 568, 585 |
| omc3/scripts/kmod\_average.py                           |      104 |        7 |     93% |162, 165, 179, 290-291, 305, 321 |
| omc3/scripts/kmod\_import.py                            |      144 |        9 |     94% |248, 261-262, 308-312, 329, 351-352, 404 |
| omc3/scripts/kmod\_lumi\_imbalance.py                   |       97 |       11 |     89% |125, 127, 154-160, 163, 167-172, 177-181, 270 |
| omc3/scripts/lhc\_corrector\_list\_check.py             |       41 |       41 |      0% |    10-124 |
| omc3/scripts/linfile\_clean.py                          |      103 |        5 |     95% |164, 184, 228, 259, 301 |
| omc3/scripts/update\_nattune\_in\_linfile.py            |       70 |        1 |     99% |       195 |
| omc3/scripts/write\_madx\_macros.py                     |       30 |       30 |      0% |    19-104 |
| omc3/segment\_by\_segment/\_\_init\_\_.py               |        0 |        0 |    100% |           |
| omc3/segment\_by\_segment/constants.py                  |       13 |        0 |    100% |           |
| omc3/segment\_by\_segment/definitions.py                |       22 |       22 |      0% |      7-33 |
| omc3/segment\_by\_segment/math.py                       |      100 |        9 |     91% |30, 192-193, 209-210, 274-275, 291-292 |
| omc3/segment\_by\_segment/propagables/\_\_init\_\_.py   |        9 |        0 |    100% |           |
| omc3/segment\_by\_segment/propagables/abstract.py       |      136 |        6 |     96% |59, 82, 316, 326, 334, 341 |
| omc3/segment\_by\_segment/propagables/alpha.py          |       66 |        7 |     89% |51-52, 126-133 |
| omc3/segment\_by\_segment/propagables/beta.py           |       57 |        7 |     88% |51-52, 118-125 |
| omc3/segment\_by\_segment/propagables/coupling.py       |      133 |        9 |     93% |159-169, 271 |
| omc3/segment\_by\_segment/propagables/dispersion.py     |      108 |        8 |     93% |117-124, 208-211 |
| omc3/segment\_by\_segment/propagables/phase.py          |       75 |        6 |     92% |60-61, 137-141 |
| omc3/segment\_by\_segment/propagables/utils.py          |       57 |        0 |    100% |           |
| omc3/segment\_by\_segment/segments.py                   |       64 |        8 |     88% |51-53, 65-68, 73 |
| omc3/tbt\_converter.py                                  |       59 |        3 |     95% |110, 128, 182 |
| omc3/tune\_analysis/\_\_init\_\_.py                     |        0 |        0 |    100% |           |
| omc3/tune\_analysis/bbq\_tools.py                       |       76 |        6 |     92% |75, 81, 86, 150-151, 175 |
| omc3/tune\_analysis/constants.py                        |       90 |        1 |     99% |        37 |
| omc3/tune\_analysis/fitting\_tools.py                   |       62 |        5 |     92% |   206-213 |
| omc3/tune\_analysis/kick\_file\_modifiers.py            |      165 |       47 |     72% |237-238, 269, 277-282, 319, 330-332, 348-379, 383-399, 403-405 |
| omc3/tune\_analysis/timber\_extract.py                  |       67 |       45 |     33% |60-62, 83-135, 148-154, 170-171 |
| omc3/utils/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| omc3/utils/contexts.py                                  |       36 |        2 |     94% |     62-63 |
| omc3/utils/debugging.py                                 |        8 |        0 |    100% |           |
| omc3/utils/iotools.py                                   |      145 |       42 |     71% |32-39, 57-65, 79, 89-90, 100-108, 116-127, 134, 142, 152, 225, 244, 287-290 |
| omc3/utils/knob\_list\_manipulations.py                 |       26 |        2 |     92% |    28, 34 |
| omc3/utils/logging\_tools.py                            |      213 |      105 |     51% |53, 56, 70-98, 101, 104-113, 127-128, 131, 134-140, 153-157, 165-171, 177-184, 196-202, 205-208, 211, 215, 263-300, 323-329, 340, 350, 388-414, 419 |
| omc3/utils/math\_classes.py                             |       29 |       29 |      0% |      8-56 |
| omc3/utils/misc.py                                      |        5 |        0 |    100% |           |
| omc3/utils/mock.py                                      |       10 |        0 |    100% |           |
| omc3/utils/outliers.py                                  |       41 |       13 |     68% |81, 85, 108, 114, 121-138 |
| omc3/utils/parsertools.py                               |       34 |        6 |     82% |19-20, 66-67, 71-72 |
| omc3/utils/rbac.py                                      |       70 |        2 |     97% |  116, 120 |
| omc3/utils/stats.py                                     |       83 |        2 |     98% |   329-330 |
| omc3/utils/time\_tools.py                               |      135 |       28 |     79% |66-67, 79, 99-110, 139, 156, 182, 197, 202, 206, 210, 214, 224, 227, 232, 237, 247, 252, 257, 262 |
| **TOTAL**                                               | **11823** | **1659** | **86%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/pylhc/omc3/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/pylhc/omc3/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pylhc/omc3/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/pylhc/omc3/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fpylhc%2Fomc3%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/pylhc/omc3/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.