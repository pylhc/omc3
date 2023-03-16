# to create XML code
from yattag import Doc, indent
import glob

folder = '/afs/cern.ch/eng/acc-models/psb/'
branch = '2021/'
folder += branch
filename = folder + 'operation/psb.jmd.xml'

default_optic = 'psb_inj_qx_4.400_qy_4.450'
default_strength = default_optic + '.str'
default_sequence = 'psb1'

# strength_files = sorted(glob.glob(folder + 'scenarios/*/*/*.str'))
tune_control_files = sorted(glob.glob(folder + 'operation/tune_control/*.str'))

doc, tag, text = Doc().tagtext()

with tag('jmad-model-definition', name = 'PSB'):
    
    # define different optics via their strengths files
    with tag('optics'):
        # for file_ in strength_files:
            # with tag('optic', name = file_.split('/')[-1][:-4], overlay = 'false'):
            #     with tag('init-files'):
            #         doc.stag('call-file', path = file_.split(branch)[-1][:-3] + 'beam')
            #         doc.stag('call-file', path = file_.split(branch)[-1], parse='STRENGTHS')
        for file_ in tune_control_files:
            with tag('optic', name = file_.split('/')[-1][:-4], overlay = 'false'):
                with tag('init-files'):
                    doc.stag('call-file', path = "operation/tune_control/psb.beam" )
                    doc.stag('call-file', path = file_.split(branch)[-1], parse='STRENGTHS')
    doc.stag('default-optic ref-name="' + default_optic + '"')
    
    # define the sequence
    with tag('sequences'):
        for i in range(4):
            with tag('sequence', name='psb' + str(i+1)):
                with tag('ranges'):
                    with tag('range', name='ALL'):
                        with tag('twiss-initial-conditions', name='default-twiss'):
                            doc.stag('chrom', value='true')
                            doc.stag('closed-orbit', value='true')
                            doc.stag('centre', value='true')
                        # with tag('post-use-files'):
                        #     doc.stag('call-file', path = "_scripts/macros.madx" )
                        #     doc.stag('call-file', path = "_scripts/assign_injection_errors.madx" )
                doc.stag('default-range ref-name="ALL"')
    doc.stag('default-sequence ref-name="' + default_sequence + '"')
    
    with tag('init-files'):
        doc.stag('call-file', path='psb.seq')
        # doc.stag('call-file', path = "_scripts/macros.madx" )

    with tag('path-offsets'):
        doc.stag('repository-prefix', value='../')
        doc.stag('resource-prefix', value='./')

result = indent(doc.getvalue(), indentation = ' '*2, newline = '\r\n')

with open(filename, 'w') as f:
    print(result, file = f)
    
print('XML code written to file ' + filename + '.')