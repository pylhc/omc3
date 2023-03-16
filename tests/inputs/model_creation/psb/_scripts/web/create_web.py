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

repo_directory = './public/'

scns = webtools.scenarios(repo_directory)

scns.loc['ad']['short_desc'] = 'Proton beams for the Antiproton Decelerator.'
scns.loc['ad']['desc'] = 'Operational scenario for the proton beams produced for the Antiproton Decelerator.'

scns.loc['east']['short_desc'] = 'Proton beams for the EAST area.'
scns.loc['east']['desc'] = 'Operational scenario for the proton beams produced for the EAST area.'

scns.loc['isolde']['short_desc'] = 'Proton beams sent to the Isotope mass Separator On-Line facility.'
scns.loc['isolde']['desc'] = 'Operational scenario for the proton beams sent to the Isotope mass Separator On-Line facility.'

scns.loc['lhc']['short_desc'] = 'Proton beams produced for the LHC physics programme.'
scns.loc['lhc']['desc'] = 'Operational scenario for the proton beams produced for the LHC physics programme.'

scns.loc['sftpro']['short_desc'] = 'Proton beams produced for the SPS fixed target physics programme.'
scns.loc['sftpro']['desc'] = 'Operational scenario for the proton beams produced for the SPS fixed target physics programme.'

scns.loc['tof']['short_desc'] = 'Proton beams for the neutron time-of-flight fixed target physics programme.'
scns.loc['tof']['desc'] = 'Operational scenario for the proton beams produced for the neutron time-of-flight fixed target physics programme.'

templateLoader = jinja2.FileSystemLoader( searchpath="./public/_scripts/web/templates/" )
templateEnv = jinja2.Environment(loader=templateLoader )

tmain = templateEnv.get_template("branch.template")
tscen = templateEnv.get_template("scenario.template")
tconf = templateEnv.get_template("configuration.template")
tmadx = templateEnv.get_template('MADX_example.template')
tyml = templateEnv.get_template('nav.yml.template')
tsurvey = templateEnv.get_template('survey.template')

# import content of global PSB MAD-X files
file_content = pnd.DataFrame(columns = ['filename', 'content'])
file_content['filename'] = sorted(glob.glob(repo_directory + 'ps*'))
file_content['short_filename'] = file_content['filename'].apply(lambda x: x.split('/')[-1])
labels = ['psb_aper_content', 'psb_seq_content']
for j, l in enumerate(labels):
    with open(file_content['filename'].iloc[j]) as f:
        file_content['content'].iloc[j] = f.read()

rdata = {'date': datetime.datetime.now().strftime("%d/%m/%Y"), 
         'year': str(year), 'branch': branch, 'scenarios': scns, 
         'psb_aper':       file_content['short_filename'].iloc[1], 
         'psb_aper_content':      file_content['content'].iloc[1],
         'psb_seq':        file_content['short_filename'].iloc[0], 
         'psb_seq_content':       file_content['content'].iloc[0],
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
webtools.renderfile([repo_directory], 'nav.yml', tyml, rdata)

# Create SURVEY content
survey = tfs.read(repo_directory + 'survey/psb_survey.tfs')
webtools.create_survey_plots(survey, repo_directory + 'survey/psb_survey.html' , 'file')

files = [repo_directory + 'survey/psb_survey.madx', repo_directory + 'survey/psb_survey.tfs']

for filename in files:        
    with open(filename) as f:
        content = f.read()
        
    short_filename = filename.split('/')[-1]
    file_content.loc[len(file_content) + 1] = [filename, content, short_filename]

rdata['psb_survey'] = file_content['short_filename'].iloc[-2]
rdata['psb_survey_content'] = file_content['content'].iloc[-2]
rdata['psb_survey_tfs'] = file_content['short_filename'].iloc[-1]
rdata['psb_survey_tfs_content'] = file_content['content'].iloc[-1]

webtools.renderfile([repo_directory + 'survey/'], 'index.md', tsurvey, rdata)
