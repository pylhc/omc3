import os
import glob
import sys
import re        # used to search numbers in strings

import pandas as pnd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
from matplotlib.colors import rgb2hex
import datetime
import scipy.constants as constants
import jinja2
import yaml
import collections

# import metaclass
# Python 3 version of the metaclass by the OMC team: pip install tfs-pandas
import tfs

from bokeh.plotting import figure, output_file, output_notebook, show, save, ColumnDataSource
from bokeh.models import Legend, LinearAxis, Range1d, CustomJS, Slider, Span, Panel, Tabs
from bokeh.models.glyphs import Rect
from bokeh.layouts import row, column, gridplot
import bokeh.palettes

plt.rcParams['xtick.direction']='in'
plt.rcParams['ytick.direction']='in'

plt.rcParams['xtick.labelsize']=14
plt.rcParams['ytick.labelsize']=plt.rcParams['xtick.labelsize']
plt.rcParams['axes.titlesize']=plt.rcParams['xtick.labelsize']

plt.rcParams['axes.labelsize']=plt.rcParams['xtick.labelsize']
plt.rcParams['legend.fontsize']=plt.rcParams['xtick.labelsize']
matplotlib.rcParams.update({'font.size': 8*2})
matplotlib.rc('font',**{'family':'serif'})
plt.rcParams["mathtext.fontset"] = "cm"

class webtools:

    def betarel(gamma):
        return np.sqrt(1-1/gamma**2)

    def E0_GeV():
        return constants.physical_constants['proton mass energy equivalent in MeV'][0]/1e3

    # def write_tfs(directory):
    #     basedir = directory + 'scenarios/'

    #     pkl_files = sorted(glob.glob(basedir + '*/*/*.pkl'))

    #     for f in pkl_files:
    #         twiss = pnd.read_pickle(f)
    #         twiss.drop_duplicates(subset='NAME', keep = 'last', inplace=True)
    #         tfs.write(f[:-3] + 'tfs', twiss)

    # def write_pkl(directory):
    #     basedir = directory + 'scenarios/'

    #     tfs_files = sorted(glob.glob(basedir + '*/*/*.tfs'))

    #     for f in tfs_files:
    #         twiss = tfs.read(f)
    #         twiss.drop_duplicates(subset='NAME', keep = 'last', inplace=True)
    #         twiss.to_pickle(f[:-3] + 'pkl')


    def write_tfs(pkl_files):

        for f in pkl_files:
            twiss = pnd.read_pickle(f)
            twiss.drop_duplicates(subset='NAME', keep = 'last', inplace=True)
            tfs.write(f[:-3] + 'tfs', twiss)

    def write_pkl(tfs_files):

        for f in tfs_files:
            twiss = tfs.read(f)
            twiss.drop_duplicates(subset='NAME', keep = 'last', inplace=True)
            twiss.to_pickle(f[:-3] + 'pkl')

    def scenarios(directory):
        
        basedir = directory + 'scenarios/'
        scns = [scn.split('/')[-2] for scn in sorted(glob.glob(basedir + '*/'))]
        
        if 'ps/' in directory:
            # make sure that bare machine scenario is first
            scns = [scns[1]] + [scns[0]] + scns[2:]
        
        scenarios = pnd.DataFrame(columns = ['label', 'short_desc', 'desc', 'config_list', 'configs', 'chroma_md', 'dir'], index = scns)
        
        for scn in scns:
            aux = scn.split('_')
            
            if len(aux) == 1:
                scenarios['label'].loc[scn] = scn.upper()
            elif 'bare' in aux:
                scenarios['label'].loc[scn] = aux[0][0].upper() + aux[0][1:] + ' ' + aux[1]
            elif len(aux) == 2:
                if aux[0] == '':
                    scenarios['label'].loc[scn] = aux[1]
                else:
                    scenarios['label'].loc[scn] = aux[0].upper() + ' ' + aux[1].upper()

            scenarios['dir'].loc[scn] = basedir.split('repository')[-1][1:] + scn
            configs = webtools.configurations(scn, directory)
            scenarios['config_list'].loc[scn] = configs.index.tolist()
            scenarios['configs'].loc[scn] = [configs.to_dict(orient="index")]
            try:
                scenarios['chroma_md'].loc[scn] = glob.glob(basedir + scn + '/*measurement.md')[0].split('/')[-1]
            except IndexError:
                scenarios['chroma_md'].loc[scn] = ''
            
        return scenarios.replace(np.nan, '', regex=True)

    def configurations(scn, directory):

        basedir = directory + 'scenarios/' + scn + '/'
        print('Importing data from scenario: ' + scn)

        # select only folders whose name starts with a number to filter folders for chromaticity measurements 
        configs = [config.split('/')[-2] for config in sorted(glob.glob(basedir + '*/'))  if config.split('/')[-2][0].isdigit()]
        
    #     configurations = pnd.DataFrame(columns = ['label', 'madx', 'madx_content', 'str', 'str_content', 'tfs', 'ps_str_content', 'ps_aper_content', 'ps_ss_content', 'ps_mu_content', 
    #                                               'twiss', 'gammarel', 'betarel', 'energy', 'kin_energy', 'momentum', 
    #                                               'Qx', 'Qy', 'Qpx', 'Qpy', 'BI_names', 's', 'betx', 'bety', 'dx', 
    #                                               'plot_pdf', 'plot_html', 'plot_height', 'directory'], 
    #                                    index = configs)
        configurations = pnd.DataFrame(columns = ['label', 'madx', 'madx_content', 'beam', 'beam_content', 'str', 'str_content',
                                                  'tfs', 'twiss_content', 'gammarel', 'betarel', 'energy', 'kin_energy', 'momentum', 
                                                  'Qx', 'Qy', 'Qpx', 'Qpy', 'BI_names', 's', 'betx', 'bety', 'dx', 
                                                  'plot_pdf', 'plot_html', 'plot_height', 'directory', 'closed_orbit', 'MTE'], 
                                       index = configs)
        
        for config in configs:
            aux = config.split('_')

            configurations['directory'].loc[config] = 'scenarios/' + scn + '/' + config + '/'
            
            if len(aux) == 2:
                configurations['label'].loc[config] = aux[1]
            elif len(aux) > 2:
                label = ''
                for i in range(len(aux)-1):
                    label = label  + aux[i+1] + ' '
                configurations['label'].loc[config] = label
            
            try:
                madx_file = glob.glob(basedir + config + '/*.madx')[0]
                beam_file = glob.glob(basedir + config + '/*.beam')[0]
                str_file = glob.glob(basedir + config + '/*.str')[0]
                tfs_file = sorted(glob.glob(basedir + config + '/*.tfs'))

                # if len(tfs_file) == 0:
                #     tfs_file = sorted(glob.glob(basedir + config + '/*.tfs'))
                                
                label = ['madx', 'beam', 'str', 'tfs']
                
                for j, file_ in enumerate([madx_file, beam_file, str_file]):
                   
                    configurations[label[j]].loc[config] = file_.split('/')[-1]
                    # import file content for local PS MAD-X files to be inserted in each configuration file
                    with open(file_) as f:
                        configurations[label[j] + '_content'].loc[config] = f.read()

                # plot_file = basedir + config + '/' + configurations['madx'].loc[config].split('.')[0] + '.pdf'
                plot_html_file = basedir + config + '/' + configurations['madx'].loc[config].split('.')[0] + '.html'
                
                digits = 2

                if 'ps_' in madx_file:
                    Q_int = ['6', '6']
                elif 'psb_' in madx_file:
                    Q_int = ['4', '4']
                elif 'leir_' in madx_file:
                    Q_int = ['1', '2']
                    digits = 3

                # if tfs_file[0].split('.')[-1] == 'pkl':
                #     twiss = pnd.read_pickle(tfs_file[0])
                # else:
                twiss = tfs.read(tfs_file[0])

                configurations['gammarel'].loc[config] = np.round(twiss.GAMMA, digits)
                configurations['betarel'].loc[config] = np.round(webtools.betarel(configurations['gammarel'].loc[config]), digits)
                configurations['energy'].loc[config] = np.round(twiss.GAMMA * webtools.E0_GeV(), digits)
                configurations['kin_energy'].loc[config] = np.round(configurations['energy'].loc[config] - webtools.E0_GeV(), digits)
                configurations['momentum'].loc[config] = np.round(twiss.PC, digits) 
                
                # to avoid rounding problems
                Qx = str(np.round(twiss.Q1, 3))
                Qy = str(np.round(twiss.Q2, 3))
                configurations['Qx'].loc[config] = Qx.replace('0', Q_int[0])
                configurations['Qy'].loc[config] = Qy.replace('0', Q_int[1])

                configurations['Qpx'].loc[config] = str(np.round(twiss.DQ1, 2))
                configurations['Qpy'].loc[config] = str(np.round(twiss.DQ2, 2))

                BI_names = [name for name in twiss.NAME.drop_duplicates() if 'BWS' in name] + [name for name in twiss.NAME.drop_duplicates() if 'BGI' in name] + [name for name in twiss.NAME.drop_duplicates() if 'MPI' in name]
                configurations['BI_names'].loc[config] = BI_names

                optics = webtools.get_optics_at_location(twiss, BI_names)
                for k in optics.keys():
                    configurations[k].loc[config] = np.round(optics[k], 3)

                configurations['plot_html'].loc[config] = plot_html_file.split('/')[-1]

                configurations['closed_orbit'].loc[config] = False
                configurations['MTE'].loc[config] = False
                if (('ext' in configurations['plot_html'].loc[config]) or ('_inj' in configurations['plot_html'].loc[config]) or ('ps_se_' in configurations['plot_html'].loc[config]) or ('leir_' in configurations['plot_html'].loc[config])) and not ('ps_ext_sftpro' in configurations['plot_html'].loc[config]):
                        configurations['closed_orbit'].loc[config] = True
                elif ('ps_hs_' in configurations['plot_html'].loc[config]) or ('ps_pr_' in configurations['plot_html'].loc[config]) or ('ps_ext_sftpro' in configurations['plot_html'].loc[config]):
                    configurations['MTE'].loc[config] = True

                webtools.create_optics_plots(twiss, plot_html_file, tfs_file, 'file', closed_orbit = configurations['closed_orbit'].loc[config], MTE = configurations['MTE'].loc[config])    

                webtools.write_pkl(tfs_file)
                pkl_files = [f[:-3] + 'pkl' for f in tfs_file]
                webtools.write_tfs(pkl_files)

                with open(tfs_file[0]) as f:
                    configurations['twiss_content'].loc[config] = f.read()

                # height of bokeh plots in pixels 
                if configurations['MTE'].loc[config] == True:
                    configurations['plot_height'].loc[config] = 850
                elif configurations['closed_orbit'].loc[config] == True:
                    configurations['plot_height'].loc[config] = 810
                else:
                    configurations['plot_height'].loc[config] = 650
            except IndexError:
                pass
        
        return configurations  

    def supplementary(directory):
        
        suppl = [suppl.split('/')[-1] for suppl in sorted(glob.glob(directory + 'supplementary/*'))]
        
        supplementary = pnd.DataFrame(columns = ['label'], index = suppl)
            
        for sup in suppl:
            aux = sup.split('_')
            
            if len(aux) == 1:
                supplementary['label'].loc[sup] = sup
            elif len(aux) == 2:
                if aux[0] == '':
                    pass
                else:
                    supplementary['label'].loc[sup] = aux[0] + ' ' + aux[1]
        
        return supplementary.replace(np.nan, '', regex=True)

    def transition_scenarios(directory):
        
        scns = [scn.split('/')[-2] for scn in sorted(glob.glob(directory + '*/'))]

        scenarios = pnd.DataFrame(columns = ['label', 'ipynb_dir'], index = scns)

        for scn in scns:
            aux = scn.split('_')

            if len(aux) == 1:
                scenarios['label'].loc[scn] = scn.upper()
            elif len(aux) == 2:
                if aux[0] == '':
                    scenarios['label'].loc[scn] = aux[1]
                else:
                    scenarios['label'].loc[scn] = aux[0].upper() + ' ' + aux[1].upper()

            scenarios['ipynb_dir'].loc[scn] = glob.glob(directory + scn + '/*.ipynb')[0]
        
        return scenarios

    def renderfile(dirnames, name, template, data):
        basedir=''
        
        for dirname in dirnames:
            basedir = os.path.join(basedir,dirname)

        fullname=os.path.join(basedir,name)
        
        with open(fullname,'w') as indexfile:
            indexfile.write(template.render(**data))
        
        print("Successfully created " + fullname)

    def get_optics_at_location(twiss, BI_names):
        optics = {'s': [], 'betx': [], 'bety': [], 'dx': []}
        for name in BI_names:
            optics['s'].append(twiss.S[twiss.NAME == name].values[0])
            optics['betx'].append(twiss.BETA11[twiss.NAME == name].values[0])
            optics['bety'].append(twiss.BETA22[twiss.NAME == name].values[0])
            optics['dx'].append(twiss.DISP1[twiss.NAME == name].values[0])
            
        return optics

    def create_optics_plots(twiss, filename, tfs_file, output, closed_orbit = False, MTE = False):

        # Load TWISS table as dataframe
        if type(twiss) == str:
            if twiss[-3:] == 'tfs':
                twiss = tfs.read(twiss)
            elif twiss[-3:] == 'pkl':
                twiss = pnd.read_pickle(twiss)
        elif type(twiss) == tfs.handler.TfsDataFrame:
            pass
        else: 
            raise TypeError("Twiss input is expected to be a string (either '*.tfs' for a MAD-X TFS table or '*.pkl' for a pickled tfs-pandas TfsDataFrame) or directly a TfsDataFrame.")

        # Extract Twiss Header
        twissHeader = dict(twiss.headers)

        # correction of wrong behaviour of PTC TWISS interpolation for active skew quadrupoles
        # keep only last element
        if twissHeader['SEQUENCE'] == 'LEIR':
            elem = [elem for elem in twiss.NAME.unique() if 'QSK' in elem]
            for e in elem:
                twiss.drop(index = twiss.index[twiss.NAME.str.contains(e)][:-1], inplace = True)

        # definition of parameters to be shown when hovering the mouse over the data points
        tooltips = [("parameter", "$name"), ("element", "@name"), ("value [m]", "$y")]
        tooltips_elements = [("element", "@name")]

        if output == 'file':
            # output to static HTML file
            output_file(filename, mode="inline")
        elif output == 'inline':
            output_notebook()

        # define the datasource
        data = pnd.DataFrame(columns = ['s', u"\u03B2x", u"\u03B2y", "Dx"])
        data['s'] = twiss.S
        data['name'] = twiss.NAME
        data['key'] = twiss.KEYWORD
        data['length'] = twiss.L

        if (('TYPE' in twissHeader) and (twissHeader['TYPE'] == 'PTC_TWISS')) or ('NAME' in twissHeader) and (twissHeader['NAME'] == 'PTC_TWISS'):
            # PTC TWISS
            data[u"\u03B2x"] = twiss.BETA11
            data[u"\u03B2y"] = twiss.BETA22
            data['Dx'] = twiss.DISP1
            data['x'] = twiss.X * 1e3
            y_data = False
            try:
                data['y'] = twiss.Y * 1e3
                y_data = True
            except AttributeError:
                data['y'] = twiss.X * 0.
        else:
            # MAD-X TWISS
            data[u"\u03B2x"] = twiss.BETX 
            data[u"\u03B2y"] = twiss.BETY
            data['Dx']       = twiss.DX * webtools.betarel(twissHeader['GAMMA'])
            try:
                data['Dy']       = twiss.DY * webtools.betarel(twissHeader['GAMMA'])
            except AttributeError:
                data['Dy']       = twiss.DX * 0.0    

        beamlets = ['core']
        tab = []

        # define data source for the MTE islands 
        if MTE == True:
            beamlets = ['core', 'island 1', 'island 2', 'island 3', 'island 4']
            #twiss_island = pnd.read_pickle(tfs_file[1])
            twiss_island = tfs.read(tfs_file[1])

        for i, beamlet in enumerate(beamlets):
            #redefine data source for the islands
            if i == 1:
                #---------------------------------------------------------
                # define the datasource for the islands
                data = pnd.DataFrame(columns = ['s', u"\u03B2x", u"\u03B2y", "Dx"])
                data['s'] = twiss_island.S
                data[u"\u03B2x"] = twiss_island.BETA11
                data[u"\u03B2y"] = twiss_island.BETA22
                data['Dx'] = twiss_island.DISP1
                data['x'] = twiss_island.X * 1e3
                data['name'] = twiss_island.NAME 
                #---------------------------------------------------------

            source = ColumnDataSource(data)

            # calculate plot limits based on data range
            # beta-functions:
            bmin = (np.floor(np.min([data[u"\u03B2x"], data[u"\u03B2y"]])/5)) * 5
            bmax = (np.floor(np.max([data[u"\u03B2x"], data[u"\u03B2y"]])/5) + 1) * 5  
            b_p2p = bmax - bmin
            y = Range1d(start = bmin - b_p2p/2, end = bmax)

            # dispersion-function:
            if twissHeader['SEQUENCE'] == 'PS':
                dxmin = (np.floor(np.min(data['Dx'])))
                dxmax = (np.floor(np.max(data['Dx'])) + 0.5)  
                dx_p2p = dxmax - dxmin
                dx = Range1d(start = dxmin, end = dxmax + dx_p2p*2)
            elif ('PSB' in twissHeader['SEQUENCE']) or (twissHeader['SEQUENCE'] == 'LEIR'):
                dxmin = (np.floor(np.max(data['Dx'])*10)/10)
                dxmax = (np.floor(np.min(data['Dx'])*10)/10)  
                dx_p2p = dxmax - dxmin
                dx = Range1d(start = dxmax, end = dxmin - dx_p2p*2)
            else:
                dxmin = (np.floor(np.min(data['Dx'])))
                dxmax = (np.floor(np.max(data['Dx'])) + 0.5)  
                dx_p2p = dxmax - dxmin
                dx = Range1d(start = dxmin, end = dxmax + dx_p2p*2)

            # create a new plot with a title and axis labels
            f = figure(title="", x_axis_label='s [m]', y_axis_label=u'\u03B2-functions [m]', width=1000, height=500, x_range=Range1d(0, twiss.LENGTH, bounds="auto"), y_range=y, tools="box_zoom, pan, reset, hover", active_drag = 'box_zoom', tooltips = tooltips)

            f.axis.major_label_text_font = 'times'
            f.axis.axis_label_text_font = 'times'
            f.axis.axis_label_text_font_style = 'normal'
            f.outline_line_color = 'black'
            f.sizing_mode = 'scale_width'

            # shift data to plot the different islands in the different tabs 
            if i > 0:
                source.data['s'] = source.data['s'] - 2 * np.pi *100 * (i-1)

            cols = ['darkblue', 'salmon']

            for j, col in enumerate(data.columns[1:3]):
                f.line('s', col, source=source, name = col, line_width=1.5, line_color = cols[j])

            # Setting the second y axis range name and range
            f.extra_y_ranges = {"disp": dx}

            # define tick locations for each SS or period
            ticks = twiss[['NAME', 'S']]
            ticks = ticks.drop_duplicates(subset='NAME', keep = 'last')
            ticks.set_index(ticks['S'], inplace=True)

            if (twissHeader['SEQUENCE'] == 'PS') or (twissHeader['SEQUENCE'] == 'LEIR'):
                ticks = ticks[ticks['NAME'].str.startswith('SS') & ticks['NAME'].str.endswith('START')]
                ticks['SS'] = ticks['NAME'].apply(lambda x: x[2:4])
                label = 'straight section'
            elif twissHeader['SEQUENCE'] == 'PSB1':
                ticks = ticks[ticks['NAME'].str.startswith('P') & ticks['NAME'].str.endswith('RING1$START')]
                ticks['SS'] = ticks['NAME'].apply(lambda x: x[1:3])
                label = 'period'

            # Setting the range name and range for top x-axis
            f.extra_x_ranges = {"sections": Range1d(start=0.0001, end=twiss.LENGTH, bounds="auto")}

            # Adding the second x axis to the plot.  
            f.add_layout(LinearAxis(x_range_name="sections", axis_label='Start of ' + label, axis_label_text_font = 'times', axis_label_text_font_style = 'normal', major_label_text_font = 'times'), 'above')
            tick_pos = ticks.index.values
            tick_pos[0] = 0.0001
            f.xaxis[0].ticker = tick_pos
            f.xaxis[0].major_label_overrides = ticks.to_dict()['SS']

            # Adding the second y axis to the plot.  
            f.add_layout(LinearAxis(y_range_name="disp", axis_label='Dx [m]', axis_label_text_font = 'times', axis_label_text_font_style = 'normal', major_label_text_font = 'times'), 'right')
            dx = f.line('s', 'Dx', source=source, name = 'Dx', line_width=1.5, line_color = 'black', y_range_name="disp")

            legend = Legend(items=[(u"\u03B2x", [f.renderers[0]]), (u"\u03B2y", [f.renderers[1]]), ("Dx", [f.renderers[2]])], location=(10, 165))
            f.add_layout(legend, 'right')
            legend.label_text_font = 'times'
            legend.click_policy = "hide"
            
            tooltips = [("parameter", "$name"), ("element", "@name"), ("value [mm]", "$y")]

            # define limits of x-plot based on maximum excursion of the islands
            if len(beamlets) > 1:
                y = (np.floor(min(twiss_island.X*1e3)/10) * 10, (np.floor(max(twiss_island.X*1e3)/10) + 1) * 10)
            else:
                y = (np.floor(min(data['x'])/10) * 10, (np.floor(max(data['x'])/10) + 1) * 10)

            #--------------------------------------------------------
            # add additional axis to plot elements

            twiss.drop_duplicates(subset='NAME', keep = 'last', inplace=True)

            f0 = figure(title="", width=1000, height=40, x_range=f.x_range, y_range=(-1.25, 1.25), tools="box_zoom, pan, reset, hover", active_drag = 'box_zoom', tooltips = tooltips_elements)

            f0.axis.visible = False
            f0.grid.visible = False
            f0.outline_line_color = 'white'
            f0.sizing_mode = 'scale_width'

            f0.toolbar.logo = None
            f0.toolbar_location = None

            webtools.plot_lattice_elements(f0, twiss, filename)

            #--------------------------------------------------------
            # add additional axis for orbit plot
            f1 = figure(title="", x_axis_label='s [m]', y_axis_label='x [mm]', width=1000, height=150, x_range=f.x_range, y_range=y, tools="box_zoom, reset", active_drag = 'box_zoom', tooltips = tooltips)

            f1.axis.major_label_text_font = 'times'
            f1.axis.axis_label_text_font = 'times'
            f1.axis.axis_label_text_font_style = 'normal'
            f1.outline_line_color = 'black'
            f1.sizing_mode = 'scale_width'

            if y_data == True:
                f1.line('s', 'x', source=source, name = 'x', line_width=1.5, line_color = 'black')
                f1.line('s', 'y', source=source, name = 'y', line_width=1.5, line_color = 'salmon')
                f1.yaxis.axis_label = 'x, y [mm]' 
                legend = Legend(items=[('x', [f1.renderers[0]]), ('y', [f1.renderers[1]])], location=(10, 20))
                f1.add_layout(legend, 'right')
                legend.label_text_font = 'times'
                legend.click_policy = "hide"
            else:
                f1.line('s', 'x', source=source, name = 'x', line_width=1.5, line_color = 'black')
                f1.yaxis.axis_label = 'x [mm]' 

            f1.toolbar.logo = None
            f1.toolbar_location = None

            tab.append(Panel(child = column([f0, f, f1], sizing_mode = 'scale_width'), title = beamlet))

        if MTE == True:
            tabs = Tabs(tabs=[ t for t in tab ])
            if output == 'file':
                save(tabs)
            elif output == 'inline':
                show(tabs)
        else:
            if closed_orbit == True:
                if output == 'file':
                    save(column([f0, f, f1], sizing_mode = 'scale_width'))
                elif output == 'inline':
                    show(column([f0, f, f1], sizing_mode = 'scale_width'))
            else:
                # save the results for standard configurations without orbit plot
                if output == 'file':
                    save(column([f0, f], sizing_mode = 'scale_width'))
                elif output == 'inline':
                    show(column([f0, f], sizing_mode = 'scale_width'))

    def create_survey_plots(survey, filename, output):
        
        # definition of parameters to be shown when hovering the mouse over the data points
        # tooltips = [("element", "@name"), ("x [m]", "$X"), ("z [m]", "$Z")]
        tooltips = [("element", "@name"), ("x [m]", "@X"), ("z [m]", "@Z")]

        if output == 'file':
            # output to static HTML file
            output_file(filename, mode="inline")
        elif output == 'inline':
            output_notebook()
        
        twissHeader = dict(survey.headers)

        # define the datasource
        data = pnd.DataFrame(columns = ['name', 'X', 'Y', 'Z', 'colors'])

        data['name'] = survey.NAME
        data['X'] = survey.X
        data['Y'] = survey.Y
        data['Z'] = survey.Z
        data['S'] = survey.S
        data = data[~data.name.str.contains('DRIFT')]
        data.reset_index(drop = True, inplace = True)
        
        col = []
        cm = plt.get_cmap('GnBu')
        for i in range(len(data)):
            color = cm(1.*i/len(data))[:3]
            col.append(rgb2hex(color))
        
        data['colors'] = col    
        
        # color_mapper = LinearColorMapper(palette='Viridis', low=min(data['S']), high=max(data['S']))

        source = ColumnDataSource(data)

        # calculate plot limits based on data range
        xmin = np.round(np.min(data['X']/50) - 1)*50
        xmax = np.round(np.max(data['X']/50))*50
        
        zmin = np.round(np.min(data['Z']/50))*50
        zmax = np.round(np.max(data['Z']/50) + 1)*50
        
        # create a new plot with a title and axis labels
        f = figure(title="", x_axis_label='x [m]', y_axis_label='z [m]', tools="box_zoom, pan, reset, hover", active_drag = 'box_zoom', tooltips = tooltips, match_aspect = True, width=500, height=500)

        f.axis.major_label_text_font = 'times'
        f.axis.axis_label_text_font = 'times'
        f.axis.axis_label_text_font_style = 'normal'
        f.outline_line_color = 'black'
        #f.sizing_mode = 'scale_both'
        
        center = [(np.min(data['X']) + np.max(data['X']))/2, (np.min(data['Z']) + np.max(data['Z']))/2]
        
        # define sectors or periods
        sectors = data[['name', 'S']]
        sectors.set_index(sectors['S'], inplace=True)
        
        if 'PS ' in twissHeader['TITLE']:
            sectors = sectors[sectors['name'].str.startswith('SEC') & sectors['name'].str.endswith('END')]
        elif 'PSB' in twissHeader['TITLE']:
            sectors = sectors[sectors['name'].str.startswith('P') & sectors['name'].str.endswith('RING3$END')]  

        if not len(sectors) == len(data):
            start = 0
            for i in range(len(sectors)):
                if i == len(sectors)-1:
                    end = data.index[-1]
                else:
                    end = data.index[data['S'] > sectors.index[i]][0]
                indx = np.arange(start, end+1, 1)
                f.patch([center[0]] + data['X'].values[indx].tolist(), [center[1]] + data['Z'].values[indx].tolist(), alpha=0.5, line_width=2, color = col[start])
                start = end
        else:
            pass

        f.line('X', 'Z', source=source, line_width=1.5, line_color = 'black')
        f.scatter('X', 'Z', fill_color='colors', line_color='black', source=source, size=20, alpha = 0.5)
        
        if output == 'file':
            save(f)
        elif output == 'inline':
            show(f)
            
    def create_Columndatasource(parameters, values):
        # parameters and data have to be a list

        data = pnd.DataFrame(columns = parameters)
        for i, p in enumerate(parameters):
            data[p] = values[i]

        return ColumnDataSource(data)

    def plot_lattice_elements(figure, twiss, filename):

        # Extract Twiss Header
        twissHeader = dict(twiss.headers)

        pos = twiss.S.values - twiss.L.values/2
        lengths = twiss.L.values
        # modify lengths in order to plot zero-length elements
        lengths[np.where(lengths == 0)[0]] += 0.001

        # BENDS    
        idx = np.array([idx for idx, elem in enumerate(twiss.KEYWORD.values) if 'BEND' in elem])

        idx_0 = idx[twiss.K1L.values[idx] == 0]
        # distinguish F and D half-units of the PS
        idx_1 = idx[twiss.K1L.values[idx] > 0]
        idx_2 = idx[twiss.K1L.values[idx] < 0]

        cols = ['#2ca25f', bokeh.palettes.Reds8[2], bokeh.palettes.Blues8[1]]
        for i, indx in enumerate([idx_0, idx_1, idx_2]):
            source = webtools.create_Columndatasource(['pos', 'width', 'name'], [pos[indx], lengths[indx], np.array(twiss.NAME.values)[indx]])
            figure.rect(x = 'pos', y = 0, width = 'width', height = 2, fill_color=cols[i], line_color = 'black', source = source)

        # QUADRUPOLES
        idx = np.array([idx for idx, elem in enumerate(twiss.KEYWORD.values) if 'QUADRUPOLE' in elem])
        name = np.array(twiss.NAME.values)[idx]
        
        if (twissHeader['SEQUENCE'] == 'PS'): 
            # loc = [map(int, re.findall(r'\d+', n))[-1]%2 for n in name] # extract SS information from string and check whether it is even or odd SS
            loc = [int(re.findall(r'\d+', n)[-1])%2 for n in name]
            idx_1 = idx[np.array(loc) == 1]     # even SS - focusing
            idx_2 = idx[np.array(loc) == 0]     # odd SS - defocusing        
        elif (twissHeader['SEQUENCE'] == 'PSB1') or (twissHeader['SEQUENCE'] == 'LEIR'): 
            idx_1 = idx[np.array([i for i, n in enumerate(name) if 'F' in n])]
            idx_2 = idx[np.array([i for i, n in enumerate(name) if 'D' in n])]

        cols = [bokeh.palettes.Reds8[2], bokeh.palettes.Blues8[1]]
        offset = [0.6, -0.6]
        for i, indx in enumerate([idx_1, idx_2]):
            source = webtools.create_Columndatasource(['pos', 'width', 'name'], [pos[indx], lengths[indx], np.array(twiss.NAME.values)[indx]])
            figure.rect(x = 'pos', y = 0 + offset[i], width = 'width', height = 1.2, fill_color=cols[i], line_color = 'black', source = source)

        # SEXTUPOLES
        try:
            idx = np.array([idx for idx, elem in enumerate(twiss.KEYWORD.values) if 'SEXTUPOLE' in elem])
            name = np.array(twiss.NAME.values)[idx]

            if (twissHeader['SEQUENCE'] == 'PS'):
                # loc = [map(int, re.findall(r'\d+', n))[-1]%2 for n in name] # extract SS information from string and check whether it is even or odd SS
                loc = [int(re.findall(r'\d+', n)[-1])%2 for n in name]
                idx_1 = idx[np.array(loc) == 1]     # even SS - focusing
                idx_2 = idx[np.array(loc) == 0]     # odd SS - defocusing
            elif (twissHeader['SEQUENCE'] == 'LEIR'): 
                idx_1 = idx[np.array([i for i, n in enumerate(name) if 'F' in n])]
                idx_2 = idx[np.array([i for i, n in enumerate(name) if 'D' in n])]

            offset = [0.4, -0.4]
            for i, indx in enumerate([idx_1, idx_2]):
                source = webtools.create_Columndatasource(['pos', 'width', 'name'], [pos[indx], lengths[indx], np.array(twiss.NAME.values)[indx]])
                figure.rect(x = 'pos', y = 0 + offset[i], width = 'width', height = 0.8, fill_color='#fff7bc', line_color = 'black', source = source)

        except IndexError:
            pass
            
        # OCTUPOLES
        try:
            idx = np.array([idx for idx, elem in enumerate(twiss.KEYWORD.values) if 'OCTUPOLE' in elem])
            name = np.array(twiss.NAME.values)[idx]

            if (twissHeader['SEQUENCE'] == 'PS'): 
                # loc = [map(int, re.findall(r'\d+', n))[-1]%2 for n in name] # extract SS information from string and check whether it is even or odd SS
                loc = [int(re.findall(r'\d+', n)[-1])%2 for n in name]
                idx_1 = idx[np.array(loc) == 1]     # even SS - focusing
                idx_2 = idx[np.array(loc) == 0]     # odd SS - defocusing        

                offset = [0.3, -0.3]
                for i, indx in enumerate([idx_1, idx_2]):
                    source = webtools.create_Columndatasource(['pos', 'width', 'name'], [pos[indx], lengths[indx], np.array(twiss.NAME.values)[indx]])
                    figure.rect(x = 'pos', y = 0 + offset[i], width = 'width', height = 0.6, fill_color='#756bb1', line_color = 'black', source = source)

        except IndexError:
            pass

        # KICKERS
        idx = np.array([idx for idx, elem in enumerate(twiss.KEYWORD.values) if 'KICKER' in elem])
        source = webtools.create_Columndatasource(['pos', 'width', 'name'], [pos[idx], lengths[idx], np.array(twiss.NAME.values)[idx]])
        figure.rect(x = 'pos', y = 0, width = 'width', height = 2, fill_color='#1b9e77', line_color = 'black', source = source)

        # CAVITIES
        idx = np.array([idx for idx, elem in enumerate(twiss.KEYWORD.values) if 'CAVITY' in elem])
        source = webtools.create_Columndatasource(['pos', 'width', 'name'], [pos[idx], lengths[idx], np.array(twiss.NAME.values)[idx]])
        figure.rect(x = 'pos', y =  0.7, width = 'width', height = 1, fill_color='#e7e1ef', line_color = 'black', source = source)
        figure.rect(x = 'pos', y = -0.7, width = 'width', height = 1, fill_color='#e7e1ef', line_color = 'black', source = source)

        # MONITORS and INSTRUMENTS
        idx = np.array([idx for idx, elem in enumerate(twiss.KEYWORD.values) if ('MONITOR' in elem) or ('INSTRUMENT' in elem)])
        source = webtools.create_Columndatasource(['pos', 'width', 'name'], [pos[idx], lengths[idx], np.array(twiss.NAME.values)[idx]])
        figure.rect(x = 'pos', y = 0, width = 'width', height = 2, fill_color='gray', line_color = 'black', source = source)

        # SOLENOIDS
        try:
            idx = np.array([idx for idx, elem in enumerate(twiss.KEYWORD.values) if 'SOLENOID' in elem])
            source = webtools.create_Columndatasource(['pos', 'width', 'name'], [pos[idx], lengths[idx], np.array(twiss.NAME.values)[idx]])
            figure.rect(x = 'pos', y = 0, width = 'width', height = 0.5, fill_color='#a4a4a4', line_color = 'black', source = source)
        except IndexError:
            pass

        # horizontal line at zero
        source = webtools.create_Columndatasource(['pos', 'name'], [[0, twiss.S.iloc[-1]], ['START', 'END']])
        figure.line('pos', 0., line_width=.5, line_color = 'black', source = source)  

    def plot_magnetic_cycle(infile, outfile, output):
        
        # definition of parameters to be shown when hovering the mouse over the data points
        tooltips = [("parameter", "$name"), ("element", "@name"), ("value [m]", "$y")]
        
        if output == 'file':
            # output to static HTML file
            output_file(outfile, mode="inline")
        elif output == 'inline':
            output_notebook()
        
        # load the datasource
        data = pnd.read_pickle(infile)
        
        source = ColumnDataSource(data)

        # calculate plot limits based on data range
        ymax = np.floor(np.max(data['B'])*10+1)/10
        factor = np.max(data['p'])/np.max(data['B'])

        # create a new plot with a title and axis labels
        f = figure(title="", x_axis_label='Cycle time [ms]', y_axis_label='Magnetic field [T]', width=400, height=250, x_range=Range1d(0, data['t'].iloc[-1], bounds="auto"), y_range=Range1d(0, ymax), tools="box_zoom, pan, reset", active_drag = 'box_zoom')

        f.axis.major_label_text_font = 'times'
        f.axis.axis_label_text_font = 'times'
        f.axis.axis_label_text_font_style = 'normal'
        f.outline_line_color = 'black'
        f.sizing_mode = 'scale_width'

        f.line('t', 'B', source=source, line_width=1.5, line_color = 'firebrick')

        # Setting the second y axis range name and range
        f.extra_y_ranges = {"p": Range1d(0, ymax * factor)}

        # Adding the second y axis to the plot.  
        f.add_layout(LinearAxis(y_range_name="p", axis_label='Momentum [GeV/c]', axis_label_text_font = 'times', axis_label_text_font_style = 'normal', major_label_text_font = 'times'), 'right')
        dx = f.line('t', 'p', source=source, line_width=1.5, line_color = 'darkblue', y_range_name="p")
        
        if output == 'file':
            save(f)
        elif output == 'inline':
            show(f)
