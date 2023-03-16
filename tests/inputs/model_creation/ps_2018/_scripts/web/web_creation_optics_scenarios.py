import datetime
import tfs
import glob
import numpy as np
import matplotlib.pyplot as plt
import jinja2
import pandas as pnd
import httpimport 
with httpimport.remote_repo(['webtools'], 'https://gitlab.cern.ch/acc-models/acc-models-ps/-/raw/2021/_scripts/web/'):
    from webtools import webtools

year = datetime.datetime.now().year

with open('branch.txt', 'r') as f:
    branch = f.readlines()[0][:-1]

repo_directory = './'

scns = webtools.scenarios(repo_directory)

scns.loc['bare_machine']['short_desc'] = 'Combined function magnets only.'
scns.loc['bare_machine']['desc'] = 'The optics are determined by the combined function magnets only, i.e. LEQ or PFW are not in use.'

scns.loc['ad']['short_desc'] = 'Proton beams for the Antiproton Decelerator.'
scns.loc['ad']['desc'] = 'Operational scenario for the proton beams produced for the Antiproton Decelerator. Before extraction the PFW are modified to change tunes and chromaticities. The magnetic cycle is displayed in the interactive plot below.'

scns.loc['east']['short_desc'] = 'Proton beams for the EAST area.'
scns.loc['east']['desc'] = '''Operational scenario for the proton beams produced for the EAST area. At flat top, the beam is debunched and extracted using a resonant slow extraction. 
                              To do so, the current in the Figure-of-8-Loop is reduced to zero on the flat top and the PFW-D and PFW-F are used to control tunes and chromaticities. 
                              Subsequently, the XSE and QSE are powered to excite the third order resonance and move the horizontal tune to 6.33. Two orbit bumps are used to approach the beam to the electrostatic septum in SS23 and the magnetic septa in SS57 and SS61.
                              The magnetic cycle is displayed in the interactive plot below.'''

scns.loc['lhc_ion']['short_desc'] = 'Lead ion beams for the LHC physics programme.'
scns.loc['lhc_ion']['desc'] = 'Operational scenario for the lead ion beams produced for the LHC physics programme. The magnetic cycle is displayed in the interactive plot below.'

scns.loc['lhc_proton']['short_desc'] = 'Proton beams produced for the LHC physics programme.'
scns.loc['lhc_proton']['desc'] = 'Operational scenario for the proton beams produced for the LHC physics programme. The magnetic cycle is displayed in the interactive plot below.'

scns.loc['sftpro']['short_desc'] = 'Proton beams produced for the SPS fixed target physics programme.'
scns.loc['sftpro']['desc'] = 'Operational scenario for the proton beams produced for the SPS fixed target physics programme. At flat top, the Multi-Turn Extraction scheme is used to split the beam in the horizontal phase space into the core and four islands. Subsequently, the beam is extracted over five turns. The magnetic cycle is displayed in the interactive plot below.'

scns.loc['tof']['short_desc'] = 'Proton beams for the n_TOF fixed target physics programme.'
scns.loc['tof']['desc'] = 'Operational scenario for the proton beams produced for the n_TOF fixed target physics programme. The magnetic cycle is displayed in the interactive plot below.'

suppl = webtools.supplementary(repo_directory)

transition_repo_directory = './supplementary/transition_crossing/'
transition_scns = webtools.transition_scenarios(transition_repo_directory)

templateLoader = jinja2.FileSystemLoader( searchpath="./_scripts/web/templates/" )
templateEnv = jinja2.Environment(loader=templateLoader )

tmain = templateEnv.get_template("branch.template")
tscen = templateEnv.get_template("scenario.template")
tconf = templateEnv.get_template("configuration.template")
tmadx = templateEnv.get_template('MADX_example.template')
tyml = templateEnv.get_template('nav.yml.template')

# import content of global PS MAD-X files
file_content = pnd.DataFrame(columns = ['filename', 'content'])
file_content['filename'] = sorted(glob.glob(repo_directory + 'ps*'))
file_content['short_filename'] = file_content['filename'].apply(lambda x: x.split('/')[-1])
labels = ['ps_aper_content', 'ps_mu_content', 'ps_ss_content']
for j, l in enumerate(labels):
    with open(file_content['filename'].iloc[j]) as f:
        file_content['content'].iloc[j] = f.read()

rdata = {'date': datetime.datetime.now().strftime("%d/%m/%Y"), 
         'year': str(year), 'branch': branch, 'scenarios': scns, 
         'supplementary': suppl, 
         'ps_mu_seq': file_content['short_filename'].iloc[1], 
         'ps_mu_content':    file_content['content'].iloc[1],
         'ps_ss_seq': file_content['short_filename'].iloc[2],
         'ps_ss_content':    file_content['content'].iloc[2],
         'transition_scenarios': transition_scns
        }

print('\nCreating websites...\n')
webtools.renderfile([repo_directory], 'index.md', tmain, rdata)

# various parameters to be included in the tables for each configuration
beam_data = ['E<sub>kin</sub> [GeV]', 
             'E<sub>tot</sub> [GeV]', 
             '&gamma;<sub>rel</sub>', 
             '&beta;<sub>rel</sub>', 
             'p [GeV/c]', 
             'Q<sub>x</sub>', 
             'Q<sub>y</sub>', 
             'Q&prime;<sub>x</sub>', 
             'Q&prime;<sub>y</sub>']

# parameters of interest at the BI equipments
BI_data = ['s [m]', 
           '&beta;<sub>x</sub> [m]', 
           '&beta;<sub>y</sub> [m]', 
           'D<sub>x</sub> [m]']

for idx, scn in scns.iterrows():
    rdata['scn'] = scn
    rdata['beam_data'] = beam_data    
    rdata['BI_names'] = next(iter(scns.iloc[0]['configs'][0].values()))['BI_names']
    # Python 2 only
    # rdata['BI_names'] = [scns.loc['bare_machine']['configs'][0][k]['BI_names'] for k in scns.loc['bare_machine']['configs'][0].keys()[0:1]][0]
    rdata['BI_data'] = BI_data

    basedir = repo_directory + 'scenarios/'
    webtools.renderfile([basedir, scn.name], 'index.md', tscen, rdata)
    
    for config in scn['config_list']:
        conf = scn.configs[0][config]
        rdata['conf'] = conf
        
        # create configuration file
        webtools.renderfile([basedir, scn.name, config], 'index.md', tconf, rdata)
        # create SWAN example
        notebook_name = 'MADX_example_' + conf['madx'][:-5] + '.ipynb'
        webtools.renderfile([basedir, scn.name, config], notebook_name, tmadx, rdata)
        
# Create mkdocs navigation structure
webtools.renderfile(['./'], 'nav.yml', tyml, rdata)