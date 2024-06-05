import numpy as np
import pandas as pd


def identify_closest_arc_bpm_to_ip(ip, side, beam, bpms):
    indices = range(1,15)
    for ii in indices:
        bpm = f'BPM.{ii}{side}{ip}.B{beam}'
        if bpm in bpms:
            return bpm


def get_left_right_pair(arc, beam, bpms):
    left_of_arc = identify_closest_arc_bpm_to_ip(int(arc[0]), 'R', beam, bpms)
    right_of_arc = identify_closest_arc_bpm_to_ip(int(arc[1]), 'L', beam, bpms)
    return [left_of_arc, right_of_arc]


def get_arc_by_arc_bpm_pairs(meas_dict, opt, plane):
    bpms = meas_dict[f'PHASE{plane}'].index
    beam = bpms[0][-1]
    bpm_pairs = {}
    bpm_pairs_with_ips = {}
    
    arcs_to_cycle = ['81', '12', '23', '34', '45', '56', '67', '78']

    for lhc_arc in arcs_to_cycle:
        bpm_pairs[lhc_arc] = get_left_right_pair(lhc_arc, beam, bpms)

    if opt.include_ips_in_arc_by_arc == 'left': 
        bpm_pairs_with_ips['81'] =  [bpm_pairs['78'][1], bpm_pairs['81'][1]]
        bpm_pairs_with_ips['12'] =  [bpm_pairs['81'][1], bpm_pairs['12'][1]]
        bpm_pairs_with_ips['23'] =  [bpm_pairs['12'][1], bpm_pairs['23'][1]]
        bpm_pairs_with_ips['34'] =  [bpm_pairs['23'][1], bpm_pairs['34'][1]]
        bpm_pairs_with_ips['45'] =  [bpm_pairs['34'][1], bpm_pairs['45'][1]]
        bpm_pairs_with_ips['56'] =  [bpm_pairs['45'][1], bpm_pairs['56'][1]]
        bpm_pairs_with_ips['67'] =  [bpm_pairs['56'][1], bpm_pairs['67'][1]]
        bpm_pairs_with_ips['78'] =  [bpm_pairs['67'][1], bpm_pairs['78'][1]]
        bpm_pairs = bpm_pairs_with_ips
    elif opt.include_ips_in_arc_by_arc == 'right': 
        bpm_pairs_with_ips['81'] =  [bpm_pairs['78'][0], bpm_pairs['81'][0]]
        bpm_pairs_with_ips['12'] =  [bpm_pairs['81'][0], bpm_pairs['12'][0]]
        bpm_pairs_with_ips['23'] =  [bpm_pairs['12'][0], bpm_pairs['23'][0]]
        bpm_pairs_with_ips['34'] =  [bpm_pairs['23'][0], bpm_pairs['34'][0]]
        bpm_pairs_with_ips['45'] =  [bpm_pairs['34'][0], bpm_pairs['45'][0]]
        bpm_pairs_with_ips['56'] =  [bpm_pairs['45'][0], bpm_pairs['56'][0]]
        bpm_pairs_with_ips['67'] =  [bpm_pairs['56'][0], bpm_pairs['67'][0]]
        bpm_pairs_with_ips['78'] =  [bpm_pairs['67'][0], bpm_pairs['78'][0]]
        bpm_pairs = bpm_pairs_with_ips

    return bpm_pairs 


def circular_sum_phase(phase_df, tune, bpm_pair, key):
    idx_0 = phase_df[key].index.get_loc(bpm_pair[0])
    idx_1 = phase_df[key].index.get_loc(bpm_pair[1])
    if idx_0 > idx_1:
        inverted_result = sum(phase_df[key][bpm_pair[1]:bpm_pair[0]])
        result = tune - inverted_result
    else:
        result = sum(phase_df[key][bpm_pair[0]:bpm_pair[1]])
    return result


def circular_sum_phase_error(phase_df, bpm_pair):
    idx_0 = phase_df['ERROR'].index.get_loc(bpm_pair[0])
    idx_1 = phase_df['ERROR'].index.get_loc(bpm_pair[1])
    if idx_0 > idx_1:
        selection = pd.concat([phase_df['ERROR'].loc[:bpm_pair[1]], phase_df['ERROR'].loc[bpm_pair[0]:]])
        result = np.sqrt(np.sum(selection**2))
    else:
        result = np.sqrt(np.sum(phase_df['ERROR'][bpm_pair[0]:bpm_pair[1]]**2))
    return result

def get_arc_phases(bpm_pairs, meas_dict, tune, plane):
    arc_meas = []
    for arc, bpm_pair in bpm_pairs.items():
        results = {}
        results['NAME'] = bpm_pair[0]
        results['NAME2'] = bpm_pair[1]
        results['WEIGHT'] = meas_dict[f'PHASE{plane}'].loc[bpm_pair[0], 'WEIGHT']
        results['VALUE'] = circular_sum_phase(meas_dict[f'PHASE{plane}'], tune, bpm_pair, 'VALUE')
        results['MODEL'] = circular_sum_phase(meas_dict[f'PHASE{plane}'], tune, bpm_pair, 'MODEL')
        results['ERROR'] = circular_sum_phase_error(meas_dict[f'PHASE{plane}'], bpm_pair)
        results['DIFF'] = results['VALUE'] - results['MODEL']
        arc_meas.append(results)
    
    meas_dict[f'PHASE{plane}'] = pd.DataFrame(arc_meas).set_index('NAME')

    return meas_dict


def reduce_to_arc_extremities(meas_dict, nominal_model, opt):
    bpm_pairs_x = get_arc_by_arc_bpm_pairs(meas_dict, opt, "X")
    bpm_pairs_y = get_arc_by_arc_bpm_pairs(meas_dict, opt, "Y")
    meas_dict = get_arc_phases(bpm_pairs_x, meas_dict, nominal_model.headers['Q1'], 'X')
    meas_dict = get_arc_phases(bpm_pairs_y, meas_dict, nominal_model.headers['Q2'], 'Y')
    return meas_dict
