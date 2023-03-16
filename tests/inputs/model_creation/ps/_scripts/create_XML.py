# to create XML code
from yattag import Doc, indent
import glob

folder = '/afs/cern.ch/eng/acc-models/ps/'
branch = '2021/'
folder += branch
filename = folder + 'operation/ps.jmd.xml'

default_optic = 'ps_fb_lhc'
default_strength = default_optic + '.str'
default_sequence = 'ps'

strength_files = sorted(glob.glob(folder + 'scenarios/*/*/*.str'))

doc, tag, text = Doc().tagtext()

with tag('jmad-model-definition', name = 'PS'):
    
    # define different optics via their strengths files
    with tag('optics'):
        for file_ in strength_files:
            with tag('optic', name = file_.split('/')[-1][:-4], overlay = 'false'):
                with tag('init-files'):
                    doc.stag('call-file', path = file_.split(branch)[-1][:-3] + 'beam')
                    doc.stag('call-file', path = file_.split(branch)[-1], parse='STRENGTHS')
    doc.stag('default-optic ref-name="' + default_optic + '"')
    
    # define the sequence
    with tag('sequences'):
        with tag('sequence', name='ps'):
            with tag('ranges'):
                with tag('range', name='ALL'):
                    with tag('twiss-initial-conditions', name='default-twiss'):
                        doc.stag('chrom', value='true')
#                         doc.stag('closed-orbit', value='false')
                        doc.stag('centre', value='true')
            doc.stag('default-range ref-name="ALL"')
    doc.stag('default-sequence ref-name="' + default_sequence + '"')
    
    with tag('init-files'):
        doc.stag('call-file', path='ps_mu.seq')
        doc.stag('call-file', path='ps_ss.seq')

    with tag('path-offsets'):
        doc.stag('repository-prefix', value='../')
        doc.stag('resource-prefix', value='./')

result = indent(doc.getvalue(), indentation = ' '*2, newline = '\r\n')

with open(filename, 'w') as f:
    print(result, file = f)
    
print('XML code written to file ' + filename + '.')