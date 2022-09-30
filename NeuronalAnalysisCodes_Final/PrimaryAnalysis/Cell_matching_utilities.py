#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
18/06/2019 - Renee Tung
This Python file contains functions that []

"""

import sys
sys.path.insert(0, '/mnt/Data4/GAERS_Codes/DataExtraction')
import extract_utils as eu
import class_definitions as ccd
import pickle as pk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import struct as st
from scipy.stats import norm
from scipy.cluster.vq import vq, kmeans, whiten
import scipy.signal as sig
import build_database as bd
import seaborn as sns
import locate_files as lf
import SpikeTrainAnalysis as sta
import math
import Get_waveforms as gw
from scipy import stats
from sklearn.mixture import GaussianMixture
import TimecourseAnalysis as ta
import itertools

def build_cell_seizure_database(cell_dataframe,seizure_dataframe, save=0):
    cell_seiztimes = pd.DataFrame()
    for index,cell in cell_dataframe.iterrows(): #for each cell
        cell_start = cell['start']
        cell_end = cell['end']
        for index,seiz in seizure_dataframe.iterrows():
            if seiz['recording_id'] != cell['recording_id']:
                continue
            seiz_start = seiz['start']
            seiz_end = seiz['end']
            if seiz_start - cell_start > 0 and cell_end - seiz_start > 0 and cell_end - seiz_end > 0:
                seiz_data = [[seiz['recording_id'], cell['Name'], cell['number'], seiz['Name'], seiz['number'], seiz['start'], seiz['end'], seiz['label'], cell['Rat']]]
                headers   = ['recording_id', 'cell_name', 'cell_number', 'seiz_name', 'seiz_number', 'seiz_start', 'seiz_end','seiz_label', 'Rat']
            
                new_data = pd.DataFrame(seiz_data, columns = headers)
                cell_seiztimes = pd.concat([cell_seiztimes,new_data])
    cell_seiztimes = sta.fix_df_index(cell_seiztimes)
    if save:
        filename = '/mnt/Data4/GAERS_Data/Cell_Seizure_Dataframe.pkl'
#        filename = '/mnt/Data4/GAERS_Data/Cell_Seiz_CortSpared.pkl'
        #filename = '/mnt/Data4/GAERS_Data/Cell_Seiz_CortImpaired.pkl'
#        filename = '/mnt/Data4/GAERS_Data/Cell_Seiz_ThalSpared.pkl'
        #filename = '/mnt/Data4/GAERS_Data/Cell_Seiz_ThalImpaired.pkl'
        bd.save_dataframe(cell_seiztimes,save_dir = filename)
    return cell_seiztimes

def extract_lfp_for_cell(seizure_series, data_dir = '/mnt/Data4/GAERS_Data/'):
    with open(data_dir + seizure_series.seiz_name, 'rb') as sz:
        seiz_info = pk.load(sz)
    with open(seiz_info.data_file, 'rb') as lfp:
        seiz_lfp = pk.load(lfp)
    dat_location = eu.get_orig_data_from_session(seizure_series.recording_id)
    num_channels = int(64/gw.total_spk_files(dat_location))

    if num_channels == 4:
        chan_num = int((seizure_series.cell_number-1)/2)
    elif num_channels == 8:
        chan_num = int(seizure_series.cell_number-1)
    this_seiz_lfp = seiz_lfp[seiz_info.number][chan_num,:]
    return this_seiz_lfp

def test_intrachannel_seizLFP(seiz_df, data_dir = '/mnt/Data4/GAERS_Data/', plot = 'sig'):
    for index,seizure in seiz_df.iterrows():
        dat_location = eu.get_orig_data_from_session(seizure['recording_id'])
        num_channels = int(64/gw.total_spk_files(dat_location))
        with open(data_dir + seizure['Name'], 'rb') as sz:
            seiz_info = pk.load(sz)
        with open(seiz_info.data_file, 'rb') as lfp:
            seiz_lfp = pk.load(lfp)
#        if num_channels == 4:
#            source_chan_num = int((seizure_series.cell_number-1)/2)
#        elif num_channels == 8:
#            source_chan_num = int(seizure_series.cell_number-1)

        this_seiz_lfp = seiz_lfp[seiz_info.number]
        
        if plot == 'sig':
            # Raw Signal
            plt.figure()
            for i in range(num_channels):
                if i < 4:
                    plt.plot(range(this_seiz_lfp.shape[1]), this_seiz_lfp[i,:], c='r', alpha=0.5)
                else:
                    plt.plot(range(this_seiz_lfp.shape[1]), this_seiz_lfp[i,:], c='b', alpha=0.5)
            lfp_mean = np.mean(this_seiz_lfp,axis=0)
            plt.plot(range(this_seiz_lfp.shape[1]), lfp_mean, c='k')
            
            plt.title('LFP from each Channel for {} in {}'.format(seizure['Name'], seizure['Rat']))
            plt.xlabel('Time Relative to Seizure Onset (ms)')
            plt.ylabel('microVolts')
                
            if num_channels == 4:
                chan12mean = (this_seiz_lfp[0,:] + this_seiz_lfp[1,:])/2
                chan34mean = (this_seiz_lfp[2,:] + this_seiz_lfp[3,:])/2
                plt.figure()
                plt.plot(range(this_seiz_lfp.shape[1]), this_seiz_lfp[0,:], c='r', alpha=1)
                plt.plot(range(this_seiz_lfp.shape[1]), this_seiz_lfp[1,:], c='b', alpha=1)
                plt.plot(range(this_seiz_lfp.shape[1]), chan12mean, c='m', alpha=0.5)
                plt.legend(['channel 1', 'channel 2', 'mean of channels 1 and 2'])
                plt.title('LFP for {} in {}'.format(seizure['Name'], seizure['Rat']))
                
                plt.figure()
                plt.plot(range(this_seiz_lfp.shape[1]), chan12mean, c='m', alpha=1)
                plt.plot(range(this_seiz_lfp.shape[1]), chan34mean, c='k', alpha=1)
                plt.legend(['mean of channels 1 and 2', 'mean of channels 3 and 4'])
                plt.title('LFP for {} in {}'.format(seizure['Name'], seizure['Rat']))
    
            elif num_channels == 8:
                chan1234mean = (this_seiz_lfp[0,:] + this_seiz_lfp[1,:] + this_seiz_lfp[2,:] + this_seiz_lfp[3,:])/4
                chan5678mean = (this_seiz_lfp[4,:] + this_seiz_lfp[5,:] + this_seiz_lfp[6,:] + this_seiz_lfp[7,:])/4
                plt.figure()
                plt.plot(range(this_seiz_lfp.shape[1]), this_seiz_lfp[0,:], c='r', alpha=1)
                plt.plot(range(this_seiz_lfp.shape[1]), this_seiz_lfp[4,:], c='b', alpha=1)
                plt.plot(range(this_seiz_lfp.shape[1]), (this_seiz_lfp[0,:] + this_seiz_lfp[4,:])/2, c='m', alpha=0.5)
                plt.legend(['channel 1', 'channel 5', 'mean of channels 1 and 5'])
                plt.title('LFP for {} in {}'.format(seizure['Name'], seizure['Rat']))
                
                plt.figure()
                plt.plot(range(this_seiz_lfp.shape[1]), chan1234mean, c='m', alpha=1)
                plt.plot(range(this_seiz_lfp.shape[1]), chan5678mean, c='k', alpha=1)
                plt.legend(['mean of channels 1:4', 'mean of channels 5:8'])
                plt.title('LFP for {} in {}'.format(seizure['Name'], seizure['Rat']))
#                
        # Correlations
        elif plot == 'corr':
            test_chan1 = this_seiz_lfp[0,:]
            test_chan2 = this_seiz_lfp[(num_channels/2), :]
            duration = this_seiz_lfp.shape[1]
            corr = sig.correlate(test_chan1, test_chan2)
            plt.figure()
            plt.plot(corr)
#            plt.xlim(duration-50, duration+50)
            plt.title('LFP Correlation for {} in {} in Channels 1 and {}'.format(seizure['Name'], seizure['Rat'], (num_channels/2 + 1)))
            plt.xlabel('Time (ms)')
#            ticks = range(0,(2*duration), 10000); labels = range(-duration,duration, 10000);
#            plt.xticks(ticks, labels)
            plt.ylabel('Correlation')
    

def seizure_firing(cell_name, num_to_plot=10, data_dir = '/mnt/Data4/GAERS_Data/', plot=0, event_look = 'stars'):
#    database = bd.load_in_dataframe()
#    seizures = sta.extract_dataframe(database, Type = 'Seizure')
#    cells = sta.extract_dataframe(database, Type = 'Cell', Name = cell_name)
    cell_seiztimes = bd.load_in_dataframe('/mnt/Data4/GAERS_Data/Cell_Seizure_Dataframe.pkl')
#    cell_seiztimes = build_cell_seizure_database(cells, seizures)
        
    spk_times,_,_ = sta.load_cell(cell_name) #load spike info
    spk_times = spk_times.values
    seiz_times,_ = sta.load_seizures(cell_name) #load seizure info 
    seiz_times = seiz_times.sort_values('start')
    seiz_times = sta.fix_df_index(seiz_times)
    
    lfp_list = []
    spike_list = []
    idx = 0
    for index,seizure in cell_seiztimes.iterrows():
        if idx == 0: #for the first time..
            with open(data_dir + seizure['seiz_name'], 'rb') as sz:
                seiz_info = pk.load(sz)
            with open(seiz_info.data_file, 'rb') as lfp:
                seiz_lfp = pk.load(lfp)
                idx +=1
            dat_location = eu.get_orig_data_from_session(seizure['recording_id'])
            num_channels = int(64/gw.total_spk_files(dat_location))
        else:
            with open(data_dir + seizure['seiz_name'], 'rb') as sz:
                seiz_info = pk.load(sz) 
        if num_channels == 4:
            chan_num = int((seizure['cell_number']-1)/2)
        elif num_channels == 8:
            chan_num = int(seizure['cell_number']-1)
        this_seiz_lfp = seiz_lfp[seiz_info.number][chan_num,:]
        lfp_list.append(this_seiz_lfp)
        
        seiz_start = seiz_times.iloc[int(seizure['seiz_number']),0]
        new_spk_times = spk_times - seiz_start
        seiz_spikes = new_spk_times[new_spk_times >= 0]
        seiz_spikes = seiz_spikes[seiz_spikes <= this_seiz_lfp.shape[0]]
        spike_list.append(spike_list)
        
        if plot:
            plt.figure()
            plt.plot(range(this_seiz_lfp.shape[0]),this_seiz_lfp)
            if event_look == 'lines':
                lfp_range = np.amax(this_seiz_lfp) - np.amin(this_seiz_lfp)
                plt.eventplot(seiz_spikes, orientation='horizontal', colors = 'r', linelengths = lfp_range)
            elif event_look == 'stars':
                events = np.empty([2,seiz_spikes.shape[0]])
                events[0,:] = np.ones(seiz_spikes.shape[0]) * (np.amin(this_seiz_lfp) - 100)
                events[1,:] = seiz_spikes
                plt.scatter(events[1],events[0],c='r',marker='*')
            plt.title('Firing of ' + cell_name + ' in ' + seizure['seiz_name'])
            plt.xlabel('Time Relative to Seizure Onset (ms)')
        if index == num_to_plot:
            break
#    cell_seiztimes['LFP'] = lfp_list
#    cell_seiztimes['spikes'] = spike_list
    return lfp_list, spike_list
        
def seizure_firing_log(seiz_df):
    seizure_firing_log = pd.DataFrame()
    seiz_firing = sta.seiz_fr(seiz_df,spk_logs=1)
    for index,seizure in seiz_firing.iterrows():
        for cell in range(seiz_firing.shape[1]):
            if np.any(np.isnan(seizure[cell])):
                continue
            seiz_data = [[index, seiz_firing.columns[cell], seizure[cell]]]
            headers   = ['seiz_name', 'cell_name', 'firing_log']
            new_data = pd.DataFrame(seiz_data, columns = headers)
            seizure_firing_log = pd.concat([seizure_firing_log, new_data])
    
    seizure_firing_log = sta.fix_df_index(seizure_firing_log)

    return seizure_firing_log

def build_LFP_database(seiz_df, data_dir = '/mnt/Data4/GAERS_Data/', save=1, window_size=500):
    LFP_dataframe = pd.DataFrame()
    ids = seiz_df['recording_id'].unique() #find unique recording id's
#    existing_df = bd.load_in_dataframe('/mnt/Data4/GAERS_Data/LFP_dataframe')
#    done_ids = existing_df['recording_id'].unique()
#    new_ids = [x for x in ids if x not in done_ids]
    for rec_id in ids: 
#        for rec_id in new_ids:
#        dat_location = eu.get_orig_data_from_session(rec_id)
#        num_channels = int(64/gw.total_spk_files(dat_location))
        seizs_in_rec = sta.extract_dataframe(seiz_df, recording_id = rec_id)
        with open(data_dir + seizs_in_rec.Name[0], 'rb') as sz:
            seiz_info = pk.load(sz)
        with open(seiz_info.data_file, 'rb') as lfp:
            seiz_lfp = pk.load(lfp) #this is the LFPs for all seizs in recording
        for index, seizure in seizs_in_rec.iterrows():
            if (seizure['end'] - seizure['start'] < window_size):
                continue
            this_seiz_lfp = seiz_lfp[int(seizure['number'])]
            mean_seiz_lfp = np.mean(this_seiz_lfp, axis=0)
            freq, time, spec = spec_of_signal(mean_seiz_lfp, window_size, plot=0)
            spike_power, wave_power, highgamma_power, dom_freq = calc_LFP_properties(freq,time,spec)
            LFP_data = [[seizure['Name'], seizure['recording_id'], seizure['Type'], seizure['number'], seizure['start'], seizure['end'], mean_seiz_lfp, spec, time, freq, spike_power, wave_power, highgamma_power, dom_freq]]
            headers  = ['Name', 'recording_id', 'Type', 'Number', 'start', 'end', 'LFP', 'spec', 'time', 'freq', 'spike_power', 'wave_power', 'highgamma_power', 'dom_freq']
            
            new_data = pd.DataFrame(LFP_data, columns = headers)
            LFP_dataframe = pd.concat([LFP_dataframe, new_data])
    LFP_dataframe = sta.fix_df_index(LFP_dataframe)
    if save:
        filename = '/mnt/Data4/GAERS_Data/LFP_dataframe.pkl'
        bd.save_dataframe(LFP_dataframe, save_dir = filename)
    return LFP_dataframe

def calc_LFP_properties(freq,time,spec):
    freq_binsize = 500 / (len(freq)-1)
    spike_power = np.sum(spec[int(math.ceil(15./freq_binsize)-1):(100/freq_binsize)-1,], axis=0) #spike is 15-100Hz
    wave_power = np.sum(spec[int(math.ceil(5./freq_binsize)-1):(12/freq_binsize)-1,], axis=0) #wave is 5-12Hz
    highgamma_power = np.sum(spec[(300/freq_binsize)-1:(500/freq_binsize)-1,], axis=0) #high gamma is 300-500Hz
    dom_freq = (np.argmax(spec[int(math.ceil(5./freq_binsize)-1):(12/freq_binsize)-1,], axis=0) + 1 + int(math.ceil(5./freq_binsize)-1)) *freq_binsize
    return spike_power, wave_power, highgamma_power, dom_freq

def LFP_prop_dic(LFP_database, seiz_dur = 'all'):
    spkpwr_dic = {}
    wvpwr_dic = {}
    hgmapwr_dic = {}
    domfrq_dic = {}
    for index, seizure in LFP_database.iterrows():
        spike_power, wave_power, highgamma_power, dom_freq = calc_LFP_properties(seizure['freq'], seizure['time'], seizure['spec'])
        if isinstance(seiz_dur, basestring):
            spkpwr_dic[seizure['Name']] = np.nanmean(spike_power)
            wvpwr_dic[seizure['Name']] = np.nanmean(wave_power)
            hgmapwr_dic[seizure['Name']] = np.nanmean(highgamma_power)
            domfrq_dic[seizure['Name']] = np.nanmean(dom_freq)
        else:
#            closest = min(seizure['time'][seizure['time']-1 > 0]) #this is the closest positive
            closest = np.argmin(abs(seizure['time']-1))
            print('Using {} ms of LFP'.format(seizure['time'][closest]))
            spkpwr_dic[seizure['Name']] = np.nanmean(spike_power[0:closest])
            wvpwr_dic[seizure['Name']] = np.nanmean(wave_power[0:closest])
            hgmapwr_dic[seizure['Name']] = np.nanmean(highgamma_power[0:closest])
            domfrq_dic[seizure['Name']] = np.nanmean(dom_freq[0:closest])
    return spkpwr_dic, wvpwr_dic, hgmapwr_dic, domfrq_dic

def plot_LFP_property(seiz_names, prop = 'all',dataframe='/mnt/Data4/GAERS_Data/LFP_dataframe.pkl'):
    LFP_dataframe = bd.load_in_dataframe(dataframe)
    select_LFP_dataframe = sta.pull_select_dataframe(LFP_dataframe, seiz_names)
    for index,seizure in select_LFP_dataframe.iterrows():
        ## NOTE: FOR THESE, THE X-AXIS TICKS SHOULD EACH BE THE LENGTH OF A WINDOW, CURRENTLY SET TO 500MS
        if prop == 'spike_power' or 'all':
            plt.figure()
            plt.plot(seizure['time'], seizure['spike_power'])
            plt.title('Spike Power in {}'.format(seizure['Name']))
            plt.xlabel('Time Relative to Seizure Onset (ms)')
            plt.ylabel('Spike Power')
        if prop == 'wave_power' or 'all':
            plt.figure()
            plt.plot(seizure['time'], seizure['wave_power'])
            plt.title('Wave Power in {}'.format(seizure['Name']))
            plt.xlabel('Time Relative to Seizure Onset (ms)')
            plt.ylabel('Wave Power')
        if prop == 'dom_freq' or 'all':
            plt.figure()
            plt.plot(seizure['time'], seizure['dom_freq'])
            plt.title('Dominant Frequency in {}'.format(seizure['Name']))
            plt.xlabel('Time Relative to Seizure Onset (ms)')
            plt.ylabel('Dominant Frequency')
        if prop == 'subplots':
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
            ax1.plot(seizure['time'], seizure['spike_power']); ax1.title.set_text('Spike Power in {}'.format(seizure['Name']))
            ax2.plot(seizure['time'], seizure['wave_power']); ax2.title.set_text('Wave Power in {}'.format(seizure['Name']))
            ax3.plot(seizure['time'], seizure['dom_freq']); ax3.title.set_text('Dominant Frequency in {}'.format(seizure['Name']))
            plt.xlabel('Time Relative to Seizure onset (ms)')

'''


'''

def plot_spared_impaired_props(prop='all'):
    spared_df = ta.get_database(classtype='Seizure',label='Spared')
    impaired_df = ta.get_database(classtype='Seizure',label='Impaired')
    spared_seiz = spared_df['Name']
    impaired_seiz = impaired_df['Name']
    
    LFP_dataframe = bd.load_in_dataframe(dataframe='/mnt/Data4/GAERS_Data/LFP_dataframe.pkl')
    spared_LFP_dataframe = sta.pull_select_dataframe(LFP_dataframe, spared_seiz)
    impaired_LFP_dataframe = sta.pull_select_dataframe(LFP_dataframe, impaired_seiz)
    
    #Get spared stats
    spared_lfp_list = []
    spared_vrms_list = []
    spared_spikep_list = []
    spared_wavep_list = []
    spared_highgamma_list = []
    spared_domfreq_list = []
    for index,seizure in spared_LFP_dataframe.iterrows():
        spared_lfp_list.append(seizure['LFP'])
        spared_vrms_list.append(vrms(seizure['LFP']))
        spared_spikep_list.append(seizure['spike_power'])
        spared_wavep_list.append(seizure['wave_power'])
        spared_highgamma_list.append(seizure['highgamma_power'])
        spared_domfreq_list.append(seizure['dom_freq'])
    
    spared_lfp = np.nanmean(nanpad_list(spared_lfp_list),axis=0)
    spared_vrms = np.nanmean(nanpad_list(spared_vrms_list),axis=0)
    spared_spikep = np.nanmean(nanpad_list(spared_spikep_list),axis=0)
    spared_wavep = np.nanmean(nanpad_list(spared_wavep_list),axis=0)
    spared_highgamma = np.nanmean(nanpad_list(spared_highgamma_list),axis=0)
    spared_domfreq = np.nanmean(nanpad_list(spared_domfreq_list),axis=0)
    
    # Get impairedstats
    impaired_lfp_list = []
    impaired_vrms_list = []
    impaired_spikep_list = []
    impaired_wavep_list = []
    impaired_highgamma_list = []
    impaired_domfreq_list = []
    for index,seizure in impaired_LFP_dataframe.iterrows():
        impaired_lfp_list.append(seizure['LFP'])
        impaired_vrms_list.append(vrms(seizure['LFP']))
        impaired_spikep_list.append(seizure['spike_power'])
        impaired_wavep_list.append(seizure['wave_power'])
        impaired_highgamma_list.append(seizure['highgamma_power'])
        impaired_domfreq_list.append(seizure['dom_freq'])
    
    impaired_lfp = np.nanmean(nanpad_list(impaired_lfp_list),axis=0)
    impaired_vrms = np.nanmean(nanpad_list(impaired_vrms_list),axis=0)
    impaired_spikep = np.nanmean(nanpad_list(impaired_spikep_list),axis=0)
    impaired_wavep = np.nanmean(nanpad_list(impaired_wavep_list),axis=0)
    impaired_highgamma = np.nanmean(nanpad_list(impaired_highgamma_list),axis=0)
    impaired_domfreq = np.nanmean(nanpad_list(impaired_domfreq_list),axis=0)
    
    nS = spared_LFP_dataframe.shape[0]
    nI = impaired_LFP_dataframe.shape[0]
    
    # Plot LFP
    #plt.figure()
    #x1 = np.arange(0,len(spared_lfp)*0.001,0.001)
    #plt.plot(x1, spared_lfp, c='r')
    #x2 = np.arange(0,len(impaired_lfp)*0.001,0.001)
    #plt.plot(x2, impaired_lfp, c='b')
    #plt.xlim(0, 30)
    #plt.ylim(-1000,1000)
    #plt.autoscale(axis='y')
    #plt.legend(['spared n = {}'.format(nS),'impaired n = {}'.format(nI)])
    #plt.xlabel('Time Relative to Seizure Onset')
    #plt.ylabel('uV')
    #plt.title('Seizure LFP by severity, aligned to onset')
    
    # Plot vRMS
    plt.figure()
    x1 = np.arange(0,len(spared_vrms)*0.25,0.25)
    plt.plot(x1, spared_vrms, c='r')
    x2 = np.arange(0,len(impaired_vrms)*0.25,0.25)
    plt.plot(x2, impaired_vrms, c='b')
    plt.xlim(0, 30)
    #plt.ylim(-1000,1000)
    #plt.autoscale(axis='y')
    plt.legend(['spared n = {}'.format(nS),'impaired n = {}'.format(nI)])
    plt.xlabel('Time Relative to Seizure Onset')
    plt.ylabel('uV^2')
    plt.title('Seizure vRMS by severity, aligned to onset')
    
    # Plot spike power
    plt.figure()
    x1 = np.arange(0.25,len(spared_spikep)*0.44,0.44)
    plt.plot(x1, spared_spikep, c='r')
    x2 = np.arange(0.25,len(impaired_spikep)*0.44,0.44)
    plt.plot(x2, impaired_spikep, c='b')
    plt.xlim(0, 30)
    #plt.ylim(-1000,1000)
    #plt.autoscale(axis='y')
    plt.legend(['spared n = {}'.format(nS),'impaired n = {}'.format(nI)])
    plt.xlabel('Time Relative to Seizure Onset')
    plt.ylabel('uV^2')
    plt.title('Seizure spike power by severity, aligned to onset')
    
    # Plot wave power
    plt.figure()
    x1 = np.arange(0.25,len(spared_wavep)*0.44,0.44)
    plt.plot(x1, spared_wavep, c='r')
    x2 = np.arange(0.25,len(impaired_wavep)*0.44,0.44)
    plt.plot(x2, impaired_wavep, c='b')
    plt.xlim(0, 30)
    #plt.ylim(-1000,1000)
    #plt.autoscale(axis='y')
    plt.legend(['spared n = {}'.format(nS),'impaired n = {}'.format(nI)])
    plt.xlabel('Time Relative to Seizure Onset')
    plt.ylabel('uV^2')
    plt.title('Seizure wave power by severity, aligned to onset')
    
    # Plot dominant frequency power
    plt.figure()
    x1 = np.arange(0.25,len(spared_highgamma)*0.44,0.44)
    plt.plot(x1, spared_highgamma, c='r')
    x2 = np.arange(0.25,len(impaired_highgamma)*0.44,0.44)
    plt.plot(x2, impaired_highgamma, c='b')
    plt.xlim(0, 30)
    #plt.ylim(-1000,1000)
    #plt.autoscale(axis='y')
    plt.legend(['spared n = {}'.format(nS),'impaired n = {}'.format(nI)])
    plt.xlabel('Time Relative to Seizure Onset')
    plt.ylabel('uV^2')
    plt.title('Seizure high gamma power by severity, aligned to onset')
    
    # Plot highgamma power
    plt.figure()
    x1 = np.arange(0.25,len(spared_domfreq)*0.44,0.44)
    plt.plot(x1, spared_domfreq, c='r')
    x2 = np.arange(0.25,len(impaired_domfreq)*0.44,0.44)
    plt.plot(x2, impaired_domfreq, c='b')
    plt.xlim(0, 30)
    #plt.ylim(-1000,1000)
    #plt.autoscale(axis='y')
    plt.legend(['spared n = {}'.format(nS),'impaired n = {}'.format(nI)])
    plt.xlabel('Time Relative to Seizure Onset')
    plt.ylabel('uV^2')
    plt.title('Seizure dominant frequency by severity, aligned to onset')
    
def nanpad_list(list, fillval=np.nan):
    lens = np.array([len(item) for item in list])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(list)
    return out

def vrms(signal, binsize=0.5, overlap=0.25, Fs=1000):

    # Calculate bin sizes and padd behavior vector
    binwidth = int(binsize*Fs)
    delta = int((binsize - overlap)*Fs)
    signalLength = len(signal)
    num_bins = int(np.floor(signalLength/delta))
    padding = np.empty((np.floor(binwidth/2).astype(int)))
    padding[:] = np.nan
    signal_padded = np.concatenate((padding, signal, padding))
    vrms = np.zeros(num_bins)

    # Calculate lick rate for each bin
    indeces = np.arange(0,signalLength-delta, delta,dtype=int)
    m = 0
    for n in indeces:
        this_bin = signal_padded[n:(n+binwidth-1)]
        bin_sqrd = np.square(this_bin)
        vrms[m] = np.sqrt(np.nanmean(bin_sqrd))
        m = m + 1;
    
    return vrms


def dom_freq_tc(seiz_list = 'all', max_time = 10000, plot = 0):
    LFP_dataframe = bd.load_in_dataframe(dataframe='/mnt/Data4/GAERS_Data/LFP_dataframe.pkl')
    if ~isinstance(seiz_list, basestring):
        LFP_dataframe = sta.pull_select_dataframe(LFP_dataframe, seiz_list)
    dom_freqs = list(LFP_dataframe['dom_freq'])
    dom_freqs = np.array(sta.pad_epochs_to_equal_length(dom_freqs, np.nan, align = 'onset'))
    mean_tc = np.nanmean(dom_freqs, axis=0)
    sem_tc = stats.sem(dom_freqs, nan_policy = 'omit')
    xs = np.array(range(0, 10001, 500))
    num_samps = len(xs)
    mean_tc = mean_tc[:num_samps]
    sem_tc = sem_tc[:num_samps]
    
    if plot:
        plt.figure()
        plt.plot(xs, mean_tc, color = 'r')
        plt.fill_between(xs,mean_tc - sem_tc,
                             mean_tc + sem_tc, alpha = 0.2, color = 'r')
        plt.xticks(xs)
        plt.title('Dominant Frequency during Seizure (500ms bins), n = {}'.format(len(dom_freqs)))
        plt.xlabel('Time Relative to Seizure Onset (ms)')
        plt.ylabel('Dominant Frequency (Hz)')
    return xs, mean_tc, sem_tc

def dom_freq_tc_durs():
    database = bd.load_in_dataframe()
    seiz_df = sta.extract_dataframe(database, Type = 'Seizure')
    short_seiz_list = list(sta.sep_seizs_by_dur(seiz_df, min_dur = 0, max_dur=5000)['Name'])
    med_seiz_list = list(sta.sep_seizs_by_dur(seiz_df, min_dur=5000, max_dur=10000)['Name'])
    long_seiz_list = list(sta.sep_seizs_by_dur(seiz_df, min_dur=10000, max_dur = 'max')['Name'])
    _, short_mean, short_sem = dom_freq_tc(seiz_list = short_seiz_list)
    _, med_mean, med_sem = dom_freq_tc(seiz_list = med_seiz_list)
    xs, long_mean, long_sem = dom_freq_tc(seiz_list = long_seiz_list)
    
    plt.figure()
    plt.xticks(xs)
    plt.title('Dominant Frequency during Short/Medium/Long Seizure (500ms bins)')
    plt.xlabel('Time Relative to Seizure Onset (ms)')
    plt.ylabel('Dominant Frequency (Hz)')
               
    plt.plot(xs, short_mean, color = 'r')
    plt.fill_between(xs, short_mean - short_sem, short_mean + short_sem, alpha = 0.2, color = 'r')
    plt.plot(xs, med_mean, color = 'b')
    plt.fill_between(xs, med_mean - med_sem, med_mean + med_sem, alpha = 0.2, color = 'b')
    plt.plot(xs, long_mean, color = 'g')
    plt.fill_between(xs, long_mean - long_sem, long_mean + long_sem, alpha = 0.2, color = 'g')
    plt.legend(['Short', 'Medium', 'Long'])
        
    
def spec_of_signal(signal, window_size, plot=0, seiz_name = 'seizure'):
    f,t,s = sig.spectrogram(signal, fs = 1000, window=sig.get_window('hamming', window_size))
    if plot:
        plt.imshow(np.log(s), origin='lower', aspect = 'auto')
        plt.colorbar()
        plt.title('Spectrogram of ' + seiz_name)
    return f, t, s
    #    plt.xticks(range(len(t)),np.round(t))
    #    plt.yticks(range(len(f)),np.round(f*1000))
    
    
def create_template(frequency=7, template_length = 1000):
    template = np.zeros(template_length)
    spacing = int((template_length) / frequency)
    for i in range(int(template_length / spacing)+1):
        template[i*spacing] = 1
    gaussian_template = sta.sliding_gaussian_window(template)
    return gaussian_template

'''
Creating Test Data
long_template = cmu.create_template(frequency=700, template_length = 1000)
convolved_template = cmu.convolve_signal(long_template)
cmu.spec_of_signal(convolved_template,2000)

Introducing noise
percent_noise = .50
from random import sample
frequency = 700; template_length = 100000
template = np.zeros(template_length)
spacing = int((template_length) / frequency)
for i in range(int(template_length / spacing)+1):
    template[i*spacing] = 1
noise_idx = sample(range(template_length),int(template_length*percent_noise))
for i in range(len(noise_idx)):
    template[noise_idx[i]] = 1
convolved_template = sta.sliding_gaussian_window(template)
cmu.spec_of_signal(convolved_template, 2000)

'''


def convolve_signal(signal, window_length = 50, plot=0):
    window=sig.get_window('hamming', window_length)
    convolved = np.convolve(window,signal, 'same')
    if plot:
        plt.figure()
        plt.plot(convolved)
    return convolved
    
def append_similar_length_sz(cell_name, sz_length,  conv_window_length=150, spec_window_length=2000):
    database = bd.load_in_dataframe()
    this_cell = sta.extract_dataframe(database, Name = cell_name)
    rec_id = this_cell.recording_id[0]
    seiz_df = sta.extract_dataframe(database, Type = 'Seizure', recording_id = rec_id)
    if len(seiz_df) == 0:
        print("This cell did not exist during any seizures")
        return 0
    firing_log_df = seizure_firing_log(seiz_df)
    firing_log_df = sta.extract_dataframe(firing_log_df, cell_name = cell_name)
    firing_log_df = sta.fix_df_index(firing_log_df)
    
    similar_length = []
    sum_seiz = 0
    for index,seiz in firing_log_df.iterrows():
        fr_log = seiz['firing_log']
        if np.sum(fr_log) == 0:
            continue
        seiz_length = len(fr_log)
        if seiz_length < (1.5*sz_length) and seiz_length > (0.5*sz_length):
            sum_seiz = sum_seiz + seiz_length
            similar_length.append(seiz['firing_log'])
            
    all_fr_logs = np.empty([sum_seiz])
    idx=0
    for seiz in range(len(similar_length)):
        log = similar_length[seiz]
        all_fr_logs[idx:idx+len(log)] = log
        idx += len(log)
            
    convolved = convolve_signal(all_fr_logs, conv_window_length)
    spec_of_signal(convolved, spec_window_length, plot=1)

def add_similar_length_sz(cell_name, sz_length, conv_window_length, spec_window_length):
    firing_log_df = seizure_firing_log()
    firing_log_df = sta.extract_dataframe(firing_log_df, cell_name = cell_name)
    firing_log_df = sta.fix_df_index(firing_log_df)
    
    similar_length = []
    for index,seiz in firing_log_df.iterrows():
        fr_log = seiz['firing_log']
        if np.sum(fr_log) == 0:
            continue
        seiz_length = len(fr_log)
        if seiz_length > (sz_length):
            similar_length.append(seiz['firing_log'])
            
    all_fr_logs = np.zeros([sz_length])
    for seiz in range(len(similar_length)):
        log = similar_length[seiz][:sz_length]
        all_fr_logs += log
        
    convolved = convolve_signal(all_fr_logs, conv_window_length)
    spec_of_signal(convolved, spec_window_length)


''' 

copy = firing_log_df 

copy = sta.fix_df_index(copy)

similar_length = []
sum_seiz = 0
for index,seiz in copy.iterrows():
    seiz_length = len(seiz['firing_log'])
    if seiz_length < (1.5*sz_length) and seiz_length > (0.5*sz_length):
        sum_seiz = sum_seiz + seiz_length
        similar_length.append(seiz['firing_log'])

all_fr_logs = np.empty([sum_seiz])
idx=0
for seiz in range(len(similar_length)):
    log = similar_length[seiz]
    all_fr_logs[idx:idx+len(log)] = log
    idx += len(log)
    
convolved = np.convolve(gaussian_template,all_fr_logs,'same')

ax = plt.subplot(312)

for i in range(len(fr_log)):
    if fr_log[i] == 1:
        plt.plot(i,fr_log[i],'.k')
        
ax = plt.subplot(313)
plt.plot(convolved)

'''
        




