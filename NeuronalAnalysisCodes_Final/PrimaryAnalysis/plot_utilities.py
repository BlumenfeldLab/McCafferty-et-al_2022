#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 20:36:18 2020

@author: pss52
"""

import pandas as pd
import pickle as pk
import warnings
import itertools
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.signal as sig
import math
import sys
sys.path.insert(0, '/mnt/Data4/GAERS_Codes/DataExtraction')
sys.path.insert(0, '/mnt/Data4/GAERS_Codes/CellCodes')
import trim_database as td
import LoadData as ld
import build_database as bd
import functools
import TimecourseAnalysis as ta
import GenericFuncsRenee as gfr
import SpikeTrainAnalysis as sta
import Cell_matching_utilities as cmu

def vrms_quartile_data(onset_period=20000,offset_period=10000,period='onset',cell_group='Cortical',cell_type=[]):
    database = ta.get_database(classtype='Seizure',label='All')
    database = ld.db_add_durs(database)
    all_seizures = td.pull_seiz_celltype(database, celltype = cell_group)
    
    # Get cell dataframe
    if cell_group == 'Cortical':
        cell_df = ta.get_database()
    elif cell_group == 'Thalamic':
        cell_df = ta.get_database(classtype='Cell',label='Thalamic',cell_type=cell_type)
    else:
        warnings.warn('Invalid cell group provided.')
        return 0
    
    vrms_list = []
    vrms_mean_list = []
    vrms_fsmean_list = []
    for idx,seizure in all_seizures.iterrows():
        
        extended_eeg = get_seiz_eeg_w_padding(seizure,onset_period=onset_period,offset_period=offset_period,period=period)
        seiz_eeg = gfr.get_seiz_eeg(seizure)
        first_sec_seiz_eeg = seiz_eeg[0:999]
        
        try:
            vrms_list.append(gfr.calc_vrms(extended_eeg, binsize=0.5, overlap=0.25, Fs=1000))
            vrms_mean_list.append(np.nanmean(gfr.calc_vrms(seiz_eeg, binsize=1, overlap=0, Fs=1000)))
            vrms_fsmean_list.append(np.nanmean(gfr.calc_vrms(first_sec_seiz_eeg, binsize=1, overlap=0, Fs=1000)))
        except:
            vrms_list.append(np.nan)
            vrms_mean_list.append(np.nan)
            vrms_fsmean_list.append(np.nan)
        if (idx%100) == 0:
            print(idx)
    return vrms_list,vrms_mean_list,vrms_fsmean_list,all_seizures,cell_df

def power_quartile_data(onset_period=20000,offset_period=10000,period='onset',cell_group='Cortical',cell_type=[]):
    database = ta.get_database(classtype='Seizure',label='All')
    database = ld.db_add_durs(database)
    
    all_seizures = td.pull_seiz_celltype(database, celltype = cell_group)
    
    # Get cell dataframe
    if cell_group == 'Cortical':
        cell_df = ta.get_database()
    elif cell_group == 'Thalamic':
        cell_df = ta.get_database(classtype='Cell',label='Thalamic',cell_type=cell_type)
    else:
        warnings.warn('Invalid cell group provided.')
        return 0
    
    spikep_list = []
    wavep_list = []
    domf_list = []
    domf_mean_list = []
    spikep_mean_list = []
    wavep_mean_list = []
    spikep_fsmean_list = []
    wavep_fsmean_list = []
    domf_fsmean_list = []
    for idx,seizure in all_seizures.iterrows():
        
        extended_eeg = get_seiz_eeg_w_padding(seizure,onset_period=onset_period,offset_period=offset_period,period=period)
        seiz_eeg = gfr.get_seiz_eeg(seizure)
        first_sec_seiz_eeg = seiz_eeg[0:999]
        
        try:
            f,t,s=gfr.calc_spect(extended_eeg, window_size = 1000, window_type = 'hamming', sampfreq = 1000, overlap = 0.5, plot=0, seiz_name = 'seizure')
            #spikep_list.append(gfr.calc_pwr(f, t, s, sampfreq=1000, lowfreq = 15, highfreq = 100))
            #wavep_list.append(gfr.calc_pwr(f, t, s, sampfreq=1000, lowfreq = 5, highfreq = 12))
            domf_list.append(gfr.calc_domfreq(f, r, s, sampfreq=1000, lowfreq=5, highfreq = 12))
            
            f,t,s=gfr.calc_spect(seiz_eeg, window_size = 1000, window_type = 'hamming', sampfreq = 1000, overlap = 0.5, plot=0, seiz_name = 'seizure')
            #spikep_mean_list.append(np.nanmean(gfr.calc_pwr(f, t, s, sampfreq=1000, lowfreq = 15, highfreq = 100)))
            #wavep_mean_list.append(np.nanmean(gfr.calc_pwr(f, t, s, sampfreq=1000, lowfreq = 5, highfreq = 12)))
            domf_mean_list.append(np.nanmean(gfr.calc_domfreq(f, r, s, sampfreq=1000, lowfreq=5, highfreq = 12)))
            '''
            f,t,s=gfr.calc_spect(first_sec_seiz_eeg, window_size = 1000, window_type = 'hamming', sampfreq = 1000, overlap = 0.5, plot=0, seiz_name = 'seizure')
            spikep_fsmean_list.append(np.nanmean(gfr.calc_pwr(f, t, s, sampfreq=1000, lowfreq = 15, highfreq = 100)))
            wavep_fsmean_list.append(np.nanmean(gfr.calc_pwr(f, t, s, sampfreq=1000, lowfreq = 5, highfreq = 12)))
            domf_fsmean_list.append(np.nanmean(gfr.calc_domfreq(f, r, s, sampfreq=1000, lowfreq=5, highfreq = 12)))
            '''
        except:
            '''
            spikep_list.append(np.nan)
            wavep_list.append(np.nan)
            spikep_mean_list.append(np.nan)
            wavep__mean_list.append(np.nan)
            spikep_fsmean_list.append(np.nan)
            wavep__fsmean_list.append(np.nan)
            '''
        if (idx%100) == 0:
            print(idx)
    return spikep_list,wavep_list,spikep_mean_list,wavep_mean_list,spikep_fsmean_list,wavep_fsmean_list,all_seizures,cell_df,domf_list,domf_mean_list,domf_fsmean_list
            
def vrms_quartile_plots(vrms_list,vrms_mean_list,vrms_fsmean_list,all_seizures,cell_df,onset_period=20000,offset_period=10000,period='onset'):       
    plot_property_tc_by_duration(vrms_list,seiz_df = all_seizures,onset_period=onset_period,offset_period=offset_period,period=period)
    plot_quartile_tc(values_list=vrms_mean_list,cell_df=cell_df,seizures_df=all_seizures,onset_period=onset_period,offset_period=offset_period,period=period)
    plot_quartile_tc(values_list=vrms_fsmean_list,cell_df=cell_df,seizures_df=all_seizures,onset_period=onset_period,offset_period=offset_period,period=period)
    plt.figure()
    plt.hist(np.array(vrms_mean_list), bins= 30, density = True, color = 'r', alpha = 0.7)
    plt.figure()
    plt.hist(np.array(vrms_fsmean_list), bins= 30, density = True, color = 'r', alpha = 0.7)
    
def plot_property_tc_by_duration(values_list,seiz_df,onset_period=20000,offset_period=10000,period='onset',fs=1000,lower_thresh=5000,upper_thresh=10000):
    
    short_values = []
    long_values = []
    for idx,seizure in seiz_df.iterrows():
        duration = seizure['end']-seizure['start']
        if duration <= lower_thresh:
            short_values.append(values_list[idx])
        elif duration >= upper_thresh:
            long_values.append(values_list[idx])
    
    short_array = np.array(short_values)
    long_array = np.array(long_values)
    xarr = np.arange(-onset_period/fs,offset_period/fs-0.5,0.5)
    gfr.plot_timecourses(arr1=short_array, arr2=long_array, xarr=xarr, 
                         arr1_label = 'Short (<5s)', arr2_label = 'Long (>10s)', 
                         win_overlap = 0, period = 'onset', smooth_size=0,
                         smooth_type = 'none', error_type = 'sem')
    
def plot_quartile_tc(values_list,cell_df,seizures_df, onset_period=20000,offset_period=10000,period='onset',fs=1000):
    
    # Sort list of values and get sizes of quartiles
    values = values_list
    values_valid = [values for values in values if str(values) != 'nan']
    sorted_values = np.sort(values_valid)
    numseiz = len(sorted_values)
    cut_num = numseiz/4
    
    # Get values belonging to each quartile
    lower_quartile = sorted_values[0:cut_num]
    upper_quartile = sorted_values[numseiz-cut_num:numseiz]
    
    # Get indeces for each quartile
    lower_idx = []
    upper_idx = []
    for value in lower_quartile:
        lower_idx.append(values_list.index(value))
    for value in upper_quartile:
        upper_idx.append(values_list.index(value))
        
    lower_df = seizures_df.iloc[lower_idx]
    upper_df = seizures_df.iloc[upper_idx] 
    
    lower_tc = gfr.get_fr_tcs(cell_df = cell_df, seiz_df = lower_df, 
               onset_period = onset_period,offset_period = offset_period, 
               period = period, within_cell = 0)
    upper_tc = gfr.get_fr_tcs(cell_df = cell_df, seiz_df = upper_df, 
               onset_period = onset_period,offset_period = offset_period, 
               period = period, within_cell = 0)
    
    xarr = np.arange(-onset_period/fs,offset_period/fs,0.025)
    gfr.plot_timecourses(arr1=lower_tc, arr2=upper_tc, xarr=xarr, 
                         arr1_label = 'Lower Quartile', arr2_label = 'Upper Quartile', 
                         win_overlap = 0.5, period = 'onset', smooth_size=50,
                         smooth_type = 'bins', error_type = 'sem')
    print('Figure 1: Binned smoothing')
    xarr = np.arange(-onset_period/fs,offset_period/fs,0.001)
    gfr.plot_timecourses(arr1=lower_tc, arr2=upper_tc, xarr=xarr, 
                         arr1_label = 'Lower Quartile', arr2_label = 'Upper Quartile', 
                         win_overlap = 0.5, period = 'onset', smooth_size=100,
                         smooth_type = 'gaussian', error_type = 'sem')
    print('Figure 2: Gaussian smoothing')
    
    '''
    lower_tc_cell = gfr.get_fr_tcs(cell_df = cell_df, seiz_df = lower_df, 
               onset_period = onset_period,offset_period = offset_period, 
               period = period, within_cell = 1)
    upper_tc_cell = gfr.get_fr_tcs(cell_df = cell_df, seiz_df = upper_df, 
               onset_period = onset_period,offset_period = offset_period, 
               period = period, within_cell = 1)
    
    xarr = np.arange(-onset_period/fs,offset_period/fs,0.025)
    gfr.plot_timecourses(arr1=lower_tc_cell, arr2=upper_tc_cell, xarr=xarr, 
                         arr1_label = 'Lower Quartile', arr2_label = 'Upper Quartile', 
                         win_overlap = 0.5, period = 'onset', smooth_size=50,
                         smooth_type = 'bins', error_type = 'sem')
    print('Figure 3: Binned smoothing. Averaged within cell')
    xarr = np.arange(-onset_period/fs,offset_period/fs,0.001)
    gfr.plot_timecourses(arr1=lower_tc_cell, arr2=upper_tc_cell, xarr=xarr, 
                         arr1_label = 'Lower Quartile', arr2_label = 'Upper Quartile', 
                         win_overlap = 0.5, period = 'onset', smooth_size=100,
                         smooth_type = 'gaussian', error_type = 'sem')
    print('Figure 4: Gaussian smoothing. Averaged within cell')
    '''
    
def get_seiz_eeg_w_padding(seizure,onset_period=20000,offset_period=10000,period='onset'):
    analysis_length = onset_period+offset_period
    
    # Get seizure values
    session = seizure['recording_id']
    seiz_num = int(seizure['number'])
    
    # Get session EEG
    sess_eeg = ld.sess_eeg_load(rec_name=session)
    
    # Get seizure times
    sess_seiz_times = ld.sess_seiz_load(rec_name=session)
    nseiz = len(sess_seiz_times)
    seiz_start = int(sess_seiz_times[seiz_num,0])
    seiz_end = int(sess_seiz_times[seiz_num,1])
    seiz_len = seiz_end-seiz_start
    
    # Get start and end indeces
    if seiz_num==0:
        first = 0
    else:
        first = int(sess_seiz_times[seiz_num-1,1])
    if seiz_num==nseiz-1:
        last = len(sess_eeg)
    else:
        last = int(sess_seiz_times[seiz_num+1,0])

    seiz_eeg = np.empty((analysis_length,))
    seiz_eeg[:] = np.NaN
    if period == 'onset':
        # Get start idx
        if first > seiz_start - onset_period:
            start_idx = first
            start_offset = onset_period-(seiz_start-first)
        else:
            start_idx = seiz_start - onset_period
            start_offset = 0
        
        # Get end idx
        if seiz_len < offset_period:
            end_idx = seiz_end
            end_offset = analysis_length-(offset_period-seiz_len)
        else:
            end_idx = seiz_start + offset_period
            end_offset = analysis_length
            
        seiz_eeg[start_offset:end_offset]=sess_eeg[start_idx:end_idx]
    else:
        # Get start idx
        if seiz_len < onset_period:
            start_idx = seiz_start
            start_offset = onset_period-seiz_len
        else:
            start_idx = seiz_end-onset_period
            start_offset = 0
            
        # Get end idx
        if last < seiz_end+offset_period:
            end_idx = last
            end_offset = analysis_length-(offset_period-(last-seiz_end))
        else:
            end_idx = seiz_end + offset_period
            end_offset = analysis_length
            
        seiz_eeg[start_offset:end_offset]=sess_eeg[start_idx:end_idx]
 
    return seiz_eeg 

def get_seiz_spikes_w_padding(seizure,cell,onset_period=20000,offset_period=10000,period='onset'):
    analysis_length = onset_period+offset_period
    
    # Get seizure values
    session = seizure['recording_id']
    seiz_num = int(seizure['number'])
    
    # Get session spikes
    spk_times,spk_log,cell_times= sta.load_cell(cell['Name'])
    
    # Get seizure times
    sess_seiz_times = ld.sess_seiz_load(rec_name=session)
    nseiz = len(sess_seiz_times)
    seiz_start = int(sess_seiz_times[seiz_num,0])
    seiz_end = int(sess_seiz_times[seiz_num,1])
    seiz_len = seiz_end-seiz_start
    
    # Get start and end indeces
    if seiz_num==0:
        first = 0
    else:
        first = int(sess_seiz_times[seiz_num-1,1])
    if seiz_num==nseiz-1:
        sess_eeg = ld.sess_eeg_load(rec_name=session)
        last = len(sess_eeg)
    else:
        last = int(sess_seiz_times[seiz_num+1,0])

    seiz_spk_log = np.empty((analysis_length,))
    seiz_spk_log[:] = np.NaN
    if period == 'onset':
        # Get start idx
        if first > seiz_start - onset_period:
            start_idx = first
            start_offset = onset_period-(seiz_start-first)
        else:
            start_idx = seiz_start - onset_period
            start_offset = 0
        
        # Get end idx
        if seiz_len < offset_period:
            end_idx = seiz_end
            end_offset = analysis_length-(offset_period-seiz_len)
        else:
            end_idx = seiz_start + offset_period
            end_offset = analysis_length
            
        diff1 = 0
        if start_idx-cell_times['start']  < 0:
            diff1 = start_idx-cell_times['start']
            start_idx = cell_times['start']
        diff2 = 0
        if end_idx-cell_times['start'] > len(spk_log):
            diff2 = end_idx-cell_times['start']-len(spk_log)+1
            end_idx = cell_times['end']
        
        seiz_spk_log[start_offset-diff1:end_offset-diff2]=spk_log[start_idx-cell_times['start']:end_idx-cell_times['start']]
        seiz_spk_times = [value for value in spk_times if value >=start_idx and value <= end_idx]
        if len(seiz_spk_times) > 0:
            seiz_spk_times = np.array(seiz_spk_times)-seiz_start
        offsets = [start_idx-seiz_start, end_idx-seiz_start]
            
    else:
        # Get start idx
        if seiz_len < onset_period:
            start_idx = seiz_start
            start_offset = onset_period-seiz_len
        else:
            start_idx = seiz_end-onset_period
            start_offset = 0
            
        # Get end idx
        if last < seiz_end+offset_period:
            end_idx = last
            end_offset = analysis_length-(offset_period-(last-seiz_end))
        else:
            end_idx = seiz_end + offset_period
            end_offset = analysis_length
            
        diff1 = 0
        if start_idx-cell_times['start']  < 0:
            diff1 = start_idx-cell_times['start']
            start_idx = cell_times['start']
        diff2 = 0
        if end_idx-cell_times['start'] > len(spk_log):
            diff2 = end_idx-cell_times['start']-len(spk_log)+1
            end_idx = cell_times['end']
        
        seiz_spk_log[start_offset-diff1:end_offset-diff2]=spk_log[start_idx-cell_times['start']:end_idx-cell_times['start']]
        seiz_spk_times = [value for value in spk_times if value >=start_idx and value <= end_idx]
        if len(seiz_spk_times) > 0:
            seiz_spk_times = np.array(seiz_spk_times)-seiz_end
        offsets = [start_idx-seiz_end, end_idx-seiz_end]
    
    return seiz_spk_log,seiz_spk_times,offsets
    
def duration_get_list_of_cells(cutoff = 5,lower_thresh=5000,upper_thresh=10000,celltype='Cortical'):   
    database = bd.load_in_dataframe()
    #update database code once newest database is verified
    database = ld.clean_liveDB(database)
    #database = ld.db_add_durs(database)
    #delete these 2 above lines once sess_times are saved
    all_seizures = td.pull_seiz_celltype(database, celltype = celltype)
    cortical_df = td.extract_dataframe(database,label=celltype)
            
    valid_cells = []      
    for idx,cell in cortical_df.iterrows():
        cell_seiz = gfr.get_seizs_in_cell(cell['Name'])
        if isinstance(cell_seiz,str):
            print(cell['Name'] + ' has no seizures...')
            continue
        
        short_list = []
        long_list = []
        for idx,seizure in cell_seiz.iterrows():
            duration = seizure['end']-seizure['start']
            if duration <= lower_thresh:
                short_list.append(seizure)
            elif duration >= upper_thresh:
                long_list.append(seizure)
        n_short = len(short_list)
        n_long = len(long_list)
        if n_short >= cutoff and n_long >=cutoff:
            valid_cells.append(cell)
            print(cell['Name'] + ' is a valid cell')
    
    cell_df = pd.DataFrame(valid_cells)
    return cell_df
        
def duration_raster_plot(cell,onset_period = 10000,offset_period = 5000, period = 'onset',
                lower_thresh=5000,upper_thresh=10000,fs=1000,bin_len=100,overlap=50,save=1,returntc=0):
    cell_seiz = gfr.get_seizs_in_cell(cell['Name'])
    
    try:
        nbins = np.int64((onset_period+offset_period)/bin_len)
    except:
        warnings.warn('Analysis period must be dividable by bin length.')
        exit(1)
    
    short_list = []
    long_list = []
    for idx,seizure in cell_seiz.iterrows():
        duration = seizure['end']-seizure['start']
        if duration <= lower_thresh:
            short_list.append(seizure)
        elif duration >= upper_thresh:
            long_list.append(seizure)
    short_df = pd.DataFrame(short_list)
    long_df = pd.DataFrame(long_list)
    
    short_spk_times = []
    short_spk_log = []
    short_offsets = []
    for idx,seizure in short_df.iterrows():
        seiz_spk_log,seiz_spk_times,offsets=get_seiz_spikes_w_padding(seizure=seizure,cell=cell,onset_period=onset_period,
                                                              offset_period=offset_period,period=period)
        short_spk_times.append(seiz_spk_times)
        short_spk_log.append(seiz_spk_log)
        short_offsets.append(offsets)
            
    long_spk_times = []
    long_spk_log = []
    long_offsets = []
    for idx,seizure in long_df.iterrows():
        seiz_spk_log,seiz_spk_times,offsets=get_seiz_spikes_w_padding(seizure=seizure,cell=cell,onset_period=onset_period,
                                                              offset_period=offset_period,period=period)
        long_spk_times.append(seiz_spk_times)
        long_spk_log.append(seiz_spk_log)
        long_offsets.append(offsets)
    
    short_array = np.array(short_spk_log)
    long_array = np.array(long_spk_log)
    
    if returntc:
        analysis_len = onset_period+offset_period
        ntcbins = (analysis_len-bin_len)/(bin_len-overlap)
        short_fr_list = []
        long_fr_list = []
        for i in range(ntcbins):
            this_bin = short_array[:,i*overlap:i*overlap+bin_len]
            this_n = np.count_nonzero(~np.isnan(this_bin))
            short_fr_list.append(fs*np.nansum(this_bin)/this_n)
            
            this_bin = long_array[:,i*overlap:i*overlap+bin_len]
            this_n = np.count_nonzero(~np.isnan(this_bin))
            long_fr_list.append(fs*np.nansum(this_bin)/this_n)
        short_fr=np.array(short_fr_list)
        long_fr=np.array(long_fr_list)
        return short_fr,long_fr  
    
    # Set up Figure
    f = plt.figure(figsize=(20,10))
    ax1 = f.add_subplot(2,2,1)
    ax2 = f.add_subplot(2,2,3,sharex=ax1)
    ax3 = f.add_subplot(2,2,2,sharex=ax1)
    ax4 = f.add_subplot(2,2,4,sharey = ax2,sharex=ax1)
    ax5 = ax2.twinx()
    ax6 = ax4.twinx()
    
    # Short Raster Plot
    lineoffsets1 = np.arange(1,len(short_array)+1,1)
    lineoffsets1 = np.concatenate((lineoffsets1,lineoffsets1))
    color_list = [[0,0,1],[0,1,0]]
    color_choices = list(itertools.chain.from_iterable(itertools.repeat(x, len(short_array)) for x in color_list))
    line_list = [0.8,1]
    line_lengths = list(itertools.chain.from_iterable(itertools.repeat(x, len(short_array)) for x in line_list))
    ax1.eventplot(short_spk_times+short_offsets,lineoffsets=lineoffsets1,colors=color_choices,linelengths=line_lengths)
    
    # Short Histogram
    counts = []
    n = []
    for i in range(nbins):
        this_bin = short_array[:,i*bin_len:(i+1)*bin_len]
        n.append(len([value for value in np.array(short_offsets) if value[0] <=(i*bin_len)-onset_period and value[1] >= (i+1)*bin_len-onset_period]))
        this_n = np.count_nonzero(~np.isnan(this_bin))
        if this_n <= 100:
            counts.append(0)
        else:
            counts.append(fs*np.nansum(this_bin)/this_n)
    x = np.arange(-onset_period,offset_period,bin_len)
    ax2.bar(x,counts,color=[0,0,1],align='edge',width=bin_len)
    ax5.plot(x,n,color=[0,1,0])
    
    # Long Raster Plot
    lineoffsets2 = np.arange(1,len(long_array)+1,1)
    lineoffsets2 = np.concatenate((lineoffsets2,lineoffsets2))
    color_list = [[1,0,0],[0,1,0]]
    color_choices = list(itertools.chain.from_iterable(itertools.repeat(x, len(long_array)) for x in color_list))
    line_list = [0.8,1]
    line_lengths = list(itertools.chain.from_iterable(itertools.repeat(x, len(long_array)) for x in line_list))
    ax3.eventplot(long_spk_times+long_offsets,lineoffsets=lineoffsets2,colors=color_choices,linelengths=line_lengths)
    
    # Long Histrogram
    counts = []
    n = []
    for i in range(nbins):
        this_bin = long_array[:,i*bin_len:(i+1)*bin_len]
        n.append(len([value for value in np.array(long_offsets) if value[0] <=(i*bin_len)-onset_period and value[1] >= (i+1)*bin_len-onset_period]))
        this_n = np.count_nonzero(~np.isnan(this_bin))
        if this_n <= 100:
            counts.append(0)
        else:
            counts.append(fs*np.nansum(this_bin)/this_n)
    x = np.arange(-onset_period,offset_period,bin_len)
    ax4.bar(x,counts,color=[1,0,0],align='edge',width=bin_len)
    ax6.plot(x,n,color=[0,1,0])
    
    # Set labels and ticks
    ax1.set(ylabel='Seizure Number',title='Short SWD')
    ax2.set(ylabel='Frequency',xlabel='Seconds to seizure onset')
    ax3.set(title='Long SWD')
    ax4.set(xlabel='Seconds to seizure onset')
    ax5.set(ylabel='Number of Seizures',ylim=[0,len(short_array)+1])
    ax6.set(ylabel='Number of Seizures',ylim=[0,len(long_array)+1])
    f.suptitle('Spikes for '+cell['Name'])
    #ylabs = ax2.get_yticks()
    #ax2.set_yticklabels(ylabs*fs)
    xlabs = ax1.get_xticks()
    ax1.set_xticklabels(xlabs/fs)
    
    # Save figure (optional)
    if save:
        figname = '/mnt/Data4/MakeFigures/Raster_Plots/'+cell['Name']+'_rasterplot.png'
        f.savefig(figname)
        plt.close(f)
        
def get_cluster_df():
    idx0 = np.where(corr_6_labels==0)[0]
    idx1 = np.where(corr_6_labels==1)[0]
    idx2 = np.where(corr_6_labels==2)[0]
    idx3 = np.where(corr_6_labels==3)[0]
    idx4 = np.where(corr_6_labels==4)[0]
    idx5 = np.where(corr_6_labels==5)[0]
    cell_names0 = []
    for x in range(len(idx0)):
        cell_names0.append(cell_names[idx0[x]])
    cell_df0 = td.pull_select_dataframe(cell_df, cell_names0)
    cell_names0 = []
    for x in range(len(idx1)):
        cell_names0.append(cell_names[idx1[x]])
    cell_df1 = td.pull_select_dataframe(cell_df, cell_names0)
    cell_names0 = []
    for x in range(len(idx2)):
        cell_names0.append(cell_names[idx2[x]])
    cell_df2 = td.pull_select_dataframe(cell_df, cell_names0)
    cell_names0 = []
    for x in range(len(idx3)):
        cell_names0.append(cell_names[idx3[x]])
    cell_df3 = td.pull_select_dataframe(cell_df, cell_names0)
    cell_names0 = []
    for x in range(len(idx4)):
        cell_names0.append(cell_names[idx4[x]])
    cell_df4 = td.pull_select_dataframe(cell_df, cell_names0)
    cell_names0 = []
    for x in range(len(idx5)):
        cell_names0.append(cell_names[idx5[x]])
    cell_df5 = td.pull_select_dataframe(cell_df, cell_names0)
    
    cell_names0 = []
    for x in range(len(group0)):
        cell_names0.append(cell_names[group0[x]])
    cell_df0 = td.pull_select_dataframe(cell_df, cell_names0)
    
    cell_names0 = []
    for x in range(len(group1)):
        cell_names0.append(cell_names[group1[x]])
    cell_df1 = td.pull_select_dataframe(cell_df, cell_names0)
    
    cell_names0 = []
    for x in range(len(group2)):
        cell_names0.append(cell_names[group2[x]])
    cell_df2 = td.pull_select_dataframe(cell_df, cell_names0)
    
    cell_names0 = []
    for x in range(len(group3)):
        cell_names0.append(cell_names[group3[x]])
    cell_df3 = td.pull_select_dataframe(cell_df, cell_names0)
    
    cell_names0 = []
    for x in range(len(group4)):
        cell_names0.append(cell_names[group4[x]])
    cell_df4 = td.pull_select_dataframe(cell_df, cell_names0)
    cell_names0 = []
    for x in range(len(group5)):
        cell_names0.append(cell_names[group5[x]])
    cell_df5 = td.pull_select_dataframe(cell_df, cell_names0)
    
    pu.group_duration_raster_plot(cell_df=cell_df0,cluster_num=0,short_df=bottomq_df,long_df=topq_df,directory=directory)
    pu.group_duration_raster_plot(cell_df=cell_df1,cluster_num=1,short_df=bottomq_df,long_df=topq_df,directory=directory)
    pu.group_duration_raster_plot(cell_df=cell_df2,cluster_num=2,short_df=bottomq_df,long_df=topq_df,directory=directory)
    pu.group_duration_raster_plot(cell_df=cell_df3,cluster_num=3,short_df=bottomq_df,long_df=topq_df,directory=directory)
    pu.group_duration_raster_plot(cell_df=cell_df4,cluster_num=4,short_df=bottomq_df,long_df=topq_df,directory=directory)
    pu.group_duration_raster_plot(cell_df=cell_df5,cluster_num=5,short_df=bottomq_df,long_df=topq_df,directory=directory)
    
    directory = '/mnt/Data4/MakeFigures/Clustering/Manual/vRMS/'
    
    bottomq_list = []
    topq_list = []
    for idx,seizure in seiz_df.iterrows():
        seiz_eeg = gfr.get_seiz_eeg(seizure)
        first_sec_seiz_eeg = seiz_eeg[0:999]
        fsvrms = np.nanmean(gfr.calc_vrms(first_sec_seiz_eeg, binsize=0.2, overlap=0.1, Fs=1000))
        if fsvrms <= 0.099415148421625937:
            bottomq_list.append(seizure)
        if fsvrms >= 0.16466279967892491:
            topq_list.append(seizure)
        if (idx%100) == 0:
            print(idx)
    bottomq_df = pd.DataFrame(bottomq_list)
    topq_df = pd.DataFrame(topq_list)
   
    
    wavep = []
    spikep = []
    for idx,seizure in seiz_df.iterrows():
        seiz_eeg = gfr.get_seiz_eeg(seizure)
         
        f,t,s=gfr.calc_spect(seiz_eeg, window_size = 1000, window_type = 'hamming', sampfreq = 1000, overlap = 0.5, plot=0, seiz_name = 'seizure')
        spikep.append(np.nanmean(gfr.calc_pwr(f, t, s, sampfreq=1000, lowfreq = 15, highfreq = 100)))
        wavep.append(np.nanmean(gfr.calc_pwr(f, t, s, sampfreq=1000, lowfreq = 5, highfreq = 12)))
        if (idx%100) == 0:
            print(idx)
        
        fsvrms = np.nanmean(gfr.calc_vrms(first_sec_seiz_eeg, binsize=0.2, overlap=0.1, Fs=1000))
        if fsvrms <= 0.099415148421625937:
            bottomq_list.append(seizure)
        if fsvrms >= 0.16466279967892491:
            topq_list.append(seizure)
        
    bottomq_df = pd.DataFrame(bottomq_list)
    topq_df = pd.DataFrame(topq_list)

def group_duration_raster_plot(cell_df,cluster_num,short_df,long_df,directory,onset_period = 10000,offset_period = 5000, period = 'onset',
                lower_thresh=5000,upper_thresh=10000,fs=1000,bin_len=500,overlap=250,save=1):
    
    try:
        nbins = np.int64((onset_period+offset_period)/bin_len)
    except:
        warnings.warn('Analysis period must be dividable by bin length.')
        exit(1)
    
    short_tc = gfr.get_fr_tcs(cell_df, seiz_df=short_df, onset_period = onset_period,offset_period = offset_period, 
               period = period, within_cell = 0)
    long_tc = gfr.get_fr_tcs(cell_df, seiz_df=long_df, onset_period = onset_period,offset_period = offset_period, 
               period = period, within_cell = 0)
    
    analysis_len = onset_period+offset_period
    ntcbins = (analysis_len-bin_len)/(bin_len-overlap)
    short_fr_list = []
    long_fr_list = []
    for i in range(ntcbins):
        this_bin = short_tc[:,i*overlap:i*overlap+bin_len]
        this_n = np.count_nonzero(~np.isnan(this_bin))
        short_fr_list.append(fs*np.nansum(this_bin)/this_n)
        
        this_bin = long_tc[:,i*overlap:i*overlap+bin_len]
        this_n = np.count_nonzero(~np.isnan(this_bin))
        long_fr_list.append(fs*np.nansum(this_bin)/this_n)
    short_fr=np.array(short_fr_list)
    long_fr=np.array(long_fr_list)
    
    # Set up Figure
    x = np.arange(-onset_period+overlap,offset_period-overlap,overlap)
    f = plt.figure(figsize=(10,10))
    ax1 = f.add_subplot(1,1,1)
    shortline,=ax1.plot(x,short_fr,color=[0,0,1])
    longline,=ax1.plot(x,long_fr,color=[1,0,0])
    ax1.legend([shortline,longline],['Bottom Quartile, n = '+str(len(short_tc)),'Top Quartile, n ='+str(len(long_tc))])
    ax1.set(ylabel='Spikes/Second',xlabel='Seconds to seizure onset')
    f.suptitle('Cluster Number '+str(cluster_num) +' vRMS Quartile, n_cells = '+str(len(cell_df)))
    xlabs = ax1.get_xticks()
    ax1.set_xticklabels(xlabs/fs)
    
    # Save figure (optional)
    if save:
        figname = directory+str(cluster_num)+'_vrms_comparison.png'
        f.savefig(figname)
        plt.close(f)
    
def duration_plot_all(celltype='Cortical'):
    cell_df = duration_get_list_of_cells(celltype=celltype)
    for idx,cell in cell_df.iterrows():
        duration_raster_plot(cell)

def all_seiz_plot_all(celltype='Thalamic',thal_type=[]):
    cell_df = ta.get_database(classtype='Cell',label=celltype,cell_type=thal_type)
    for idx,cell in cell_df.iterrows():
        all_seiz_raster_plot(cell)
    
def all_seiz_raster_plot(cell,onset_period = 10000,offset_period = 10000, period = 'onset',
                fs=1000,bin_len=100,overlap=50,save=1,plot=1,returntc=0):
    cell_seiz = gfr.get_seizs_in_cell(cell['Name'])
    if isinstance(cell_seiz, basestring): #would be a string if no seizs in cell
        return 0
    
    try:
        nbins = np.int64((onset_period+offset_period)/bin_len)
    except:
        warnings.warn('Analysis period must be dividable by bin length.')
        exit(1)
    
    spk_times = []
    spk_log = []
    offsets = []
    for idx,seizure in cell_seiz.iterrows():
        seiz_spk_log,seiz_spk_times,these_offsets=get_seiz_spikes_w_padding(seizure=seizure,cell=cell,onset_period=onset_period,
                                                              offset_period=offset_period,period=period)
        if len(seiz_spk_times) > 0:
            spk_times.append(seiz_spk_times)
            spk_log.append(seiz_spk_log)
            offsets.append(these_offsets)
    
    spk_log_array = np.array(spk_log)
    if spk_log_array.ndim < 2:
        return 0
    
    if returntc:
        analysis_len = onset_period+offset_period
        ntcbins = (analysis_len-bin_len)/(bin_len-overlap)
        fr_list = []
        for i in range(ntcbins):
            this_bin = spk_log_array[:,i*overlap:i*overlap+bin_len]
            this_n = np.count_nonzero(~np.isnan(this_bin))
            fr_list.append(fs*np.nansum(this_bin)/this_n)
        fr=np.array(fr_list)
        return fr        
            
    
    if plot:
        # Set up Figure
        f = plt.figure(figsize=(10,10))
        ax1 = f.add_subplot(2,1,1)
        ax2 = f.add_subplot(2,1,2,sharex=ax1)
        ax3 = ax2.twinx()
        
        # Raster Plot
        lineoffsets1 = np.arange(1,len(spk_log_array)+1,1)
        lineoffsets1 = np.concatenate((lineoffsets1,lineoffsets1))
        color_list = [[0,0,1],[0,1,0]]
        color_choices = list(itertools.chain.from_iterable(itertools.repeat(x, len(spk_log_array)) for x in color_list))
        line_list = [0.8,1]
        line_lengths = list(itertools.chain.from_iterable(itertools.repeat(x, len(spk_log_array)) for x in line_list))
        ax1.eventplot(spk_times+offsets,lineoffsets=lineoffsets1,colors=color_choices,linelengths=line_lengths)
        
        # Histogram
        counts = []
        n = []
        for i in range(nbins):
            this_bin = spk_log_array[:,i*bin_len:(i+1)*bin_len]
            n.append(len([value for value in np.array(offsets) if value[0] <=(i*bin_len)-onset_period and value[1] >= (i+1)*bin_len-onset_period]))
            this_n = np.count_nonzero(~np.isnan(this_bin))
            if this_n <= 100:
                counts.append(0)
            else:
                counts.append(fs*np.nansum(this_bin)/this_n)
        x = np.arange(-onset_period,offset_period,bin_len)
        ax2.bar(x,counts,color=[0,0,1],align='edge',width=bin_len)
        ax3.plot(x,n,color=[0,1,0])
        
        # Set labels and ticks
        ax1.set(ylabel='Seizure Number')
        ax2.set(ylabel='Spikes / second',xlabel='Seconds to seizure ' + period)
        ax3.set(ylabel='Number of Seizures',ylim=[0,len(spk_log_array)+1])
        f.suptitle('Spikes for '+cell['Name'])
        xlabs = ax1.get_xticks()
        ax1.set_xticklabels(xlabs/fs)
        
        # Save figure (optional)
        if save:
            figname = '/mnt/Data4/MakeFigures/All_Seiz_Rasters/'+cell['Name']+'_rasterplot.png'
            f.savefig(figname)
            plt.close(f)
    
def get_cell_spike_stats(cell,baseline_start=20000,baseline_end=2000,seiz_analysis_len=100000,fs = 1000,first_len=500,prelen=1000):
    analysis_length = baseline_start+seiz_analysis_len
    
    # Get cell values
    session = cell['recording_id']
    spk_times,spk_log,cell_times= sta.load_cell(cell['Name'])
    cell_seiz = gfr.get_seizs_in_cell(cell['Name'])
    sess_seiz_times = ld.sess_seiz_load(rec_name=session)
    
    if isinstance(cell_seiz, basestring): #would be a string if no seizs in cell
        baseline_fr = 0
        preseiz_fr = 0
        fp_fr = 0
        seiz_fr = 0
        return baseline_fr,preseiz_fr,fp_fr,seiz_fr
    
    all_spk_log=[]
    for idx,seizure in cell_seiz.iterrows():
        # Get seizure times
        seiz_num = int(seizure['number'])
        nseiz = len(sess_seiz_times)
        seiz_start = int(sess_seiz_times[seiz_num,0])
        seiz_end = int(sess_seiz_times[seiz_num,1])
        seiz_len = seiz_end-seiz_start
    
        # Get start and end indeces
        if seiz_num==0:
            first = 0
        else:
            first = int(sess_seiz_times[seiz_num-1,1])

        seiz_spk_log = np.empty((analysis_length,))
        seiz_spk_log[:] = np.NaN
        
        # Get start idx
        if first > seiz_start - baseline_start:
            start_idx = first
            start_offset = baseline_start-(seiz_start-first)
        else:
            start_idx = seiz_start - baseline_start
            start_offset = 0
        
        # Get end idx
        if seiz_len < seiz_analysis_len:
            end_idx = seiz_end
            end_offset = analysis_length-(seiz_analysis_len-seiz_len)
        else:
            end_idx = seiz_start + seiz_analysis_len
            end_offset = analysis_length
            
        diff1 = 0
        if start_idx-cell_times['start']  < 0:
            diff1 = start_idx-cell_times['start']
            start_idx = cell_times['start']
        diff2 = 0
        if end_idx-cell_times['start'] > len(spk_log):
            diff2 = end_idx-cell_times['start']-len(spk_log)
            end_idx = cell_times['end']
        
        seiz_spk_log[start_offset-diff1:end_offset+diff2]=spk_log[start_idx-cell_times['start']:end_idx-cell_times['start']]
        seiz_spk_times = [value for value in spk_times if value >=start_idx and value <= end_idx]
        if len(seiz_spk_times) > 0:
            all_spk_log.append(seiz_spk_log)
 
    # Convert list to array
    spk_log_array = np.array(all_spk_log)
    if spk_log_array.ndim < 2:
        baseline_fr = 0
        fp_fr = 0
        seiz_fr = 0
        return baseline_fr,fp_fr,seiz_fr
    
    # Get baseline firing rate
    baseline_period = spk_log_array[:,0:baseline_start-baseline_end]
    n = np.count_nonzero(~np.isnan(baseline_period))
    baseline_fr = fs*np.nansum(baseline_period)/n
    
    # Get preseizure firing rate
    preseizure_period = spk_log_array[:,baseline_start-prelen:baseline_start]
    n = np.count_nonzero(~np.isnan(preseizure_period))
    preseiz_fr = fs*np.nansum(preseizure_period)/n
    
    # Get first second firing rate
    first_part = spk_log_array[:,baseline_start:baseline_start+first_len]
    n = np.count_nonzero(~np.isnan(first_part))
    fp_fr = fs*np.nansum(first_part)/n
    
    # Get seizure firing rate
    seizure_period = spk_log_array[:,baseline_start+first_len:analysis_length]
    n = np.count_nonzero(~np.isnan(seizure_period))
    seiz_fr = fs*np.nansum(seizure_period)/n
    
    return baseline_fr,preseiz_fr,fp_fr,seiz_fr

def plot_cell_spike_stats(cell_df,title,save=1):
    seiz_diff = []
    fp_diff = []
    seiz_percentage = []
    fp_percentage = []
    for idx,cell in cell_df.iterrows():
        baseline_fr,preseiz_fr,fp_fr,seiz_fr = get_cell_spike_stats(cell)
        
        if baseline_fr == 0:
            continue
        
        # Get seizure stats
        seiz_diff.append(seiz_fr-baseline_fr)
        seiz_ratio = seiz_fr/baseline_fr
        seiz_percentage.append(100*(seiz_ratio-1))
        
        # Get first part stats
        fp_diff.append(fp_fr-baseline_fr)
        fp_ratio = fp_fr/baseline_fr
        fp_percentage.append(100*(fp_ratio-1))
    
    # Convert lists to arrays
    seiz_diff = np.array(seiz_diff)
    seiz_percentage = np.array(seiz_percentage)
    fp_diff = np.array(fp_diff)
    fp_percentage = np.array(fp_percentage)
    
    # difference plot
    f = plt.figure(figsize=(10,10))
    plt.plot(seiz_diff, fp_diff, 'o', color='blue')
    plt.title(title + ' Firing Rate Differences')
    plt.ylabel('First Part of Seizure Firing Rate - Baseline FR (Spikes/Second)')
    plt.xlabel('Seizure Firing Rat - Baseline FR (Spikes/Second)')
    plt.ylim((-20,20))
    plt.xlim((-20,20))
    
    # Save figure (optional)
    if save:
        figname = '/mnt/Data4/MakeFigures/Firing_Rate_Analysis/'+title+'_FR_Differences.png'
        f.savefig(figname)
        plt.close(f)
    
    # percentage plot
    g = plt.figure(figsize=(10,10))
    plt.plot(seiz_percentage, fp_percentage, 'o', color='blue')
    plt.title(title + ' Firing Rate Percentage')
    plt.ylabel('First Part of Seizure Firing Rate % Difference to Baseline FR')
    plt.xlabel('Seizure Firing Rat % Difference to Baseline FR')
    plt.ylim((-100,150))
    plt.xlim((-100,150))
    
    # Save figure (optional)
    if save:
        figname = '/mnt/Data4/MakeFigures/Firing_Rate_Analysis/'+title+'_FR_Percentages.png'
        g.savefig(figname)
        plt.close(g)
        
        