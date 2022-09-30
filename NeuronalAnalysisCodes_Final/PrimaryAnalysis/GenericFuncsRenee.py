#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 10:48:51 2020
Generic Functions
@author: rjt37
"""

import pandas as pd
import pickle as pk
import warnings
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.signal as sig
import math
import sys
sys.path.insert(0, '/mnt/Data4/GAERS_Codes/DataExtraction')
sys.path.insert(0, '/mnt/Data4/GAERS_Codes/SeizureCodes')
import seizure_classification_v2 as sc
import trim_database as td
import LoadData as ld
import build_database as bd
import get_behav_preds as gbp
import functools

def split_percentile(array, percentile = 25):
    #func splits the input array into the lower and upper percentiles
    array = np.sort(array)
    num = len(array)
    lowcut = np.percentile(array, percentile, interpolation = 'lower')
    cutsize = sum(array <= lowcut)
    lower = np.split(array, [cutsize-1])[0]
    upper = np.split(array, [num-cutsize])[1]
    return lower, upper
    
def split_dict_percentile(dic, percentile = 25):
    #same as above func, but takes and returns dict, with key as the name
    values = dic.values()
    lower, upper = split_percentile(values, percentile)
    low_dict = {}
    high_dict = {}
    for key, value in dic.items():
        if value in lower:
            low_dict[key] = value
        elif value in upper:
            high_dict[key] = value
    return low_dict, high_dict

def split_dict_value(dic, lowcut, highcut):
    #splits a dictionary into two:
    #one with values below lowcut and oneswith values above highcut
    low_dict = {}; high_dict = {};
    for key, value in dic.items():
        if value <= lowcut:
            low_dict[key] = value
        elif value > highcut:
            high_dict[key] = value
    return low_dict, high_dict

def plot_histquartiles(arr1, arr2, arr1_label = 'array1', arr2_label = 'array2'):
    #arr1 and arr2 are 2 arrays containing data divided by some predetermined property
    #func plots histogram of these data
    num1 = len(arr1)
    num2 = len(arr2)
    plt.figure()
    plt.hist(arr1, bins= int(num1*0.8), density = True, color = 'r', alpha = 0.7)
    plt.hist(arr2, bins = int(num2*0.8), density = True, color = 'b', alpha = 0.7)
    plt.legend(['{} n = {}'.format(arr1_label,num1), '{} n = {}'.format(arr2_label, num2)])
    plt.title('Normalized Density Histograms for {} and {}'.format(arr1_label, arr2_label))
    plt.ylabel('Probability')
    
def sliding_gaussian_window(spk_log,time_window=100):
    normVector = np.zeros(time_window)#create a normal distribution vector
    for i in range(0,time_window):
        normVector[i] = stats.norm.pdf(i-time_window/2,0,time_window/4)
    gaussian_tc = np.convolve(spk_log,normVector,'same')  #convolve spike function with normal distribution
    return gaussian_tc
    
def add_sev_todatabase():
    database = bd.load_in_dataframe(); rec_ids = set(list(database['recording_id']))
    spared, impaired = gbp.load_preds(); spared = spared[['Session','Seizure']]; impaired = impaired[['Session','Seizure']]
    for rec_id in rec_ids:
        sess_db = td.extract_dataframe(database, recording_id = rec_id)
        spared_sess = td.extract_dataframe(spared,Session=rec_id)
        impaired_sess = td.extract_dataframe(impaired,Session=rec_id)
        if len(spared_sess) > 0:
            temp = td.pull_select_number_dataframe(sess_db,list(spared_sess['Seizure']))
            for i,row in temp.iterrows():
                sz = database[database['Name']==row['Name']]
                database.loc[sz.index,'label'] = 'Spared'
        if len(impaired_sess) > 0:
            temp = td.pull_select_number_dataframe(sess_db,list(impaired_sess['Seizure']))
            for i,row in temp.iterrows():
                sz = database[database['Name']==row['Name']]
                database.loc[sz.index,'label'] = 'Impaired'
    return database
        

def plot_single_timecourse(x, tc, cell_label, cell_type, smooth_size, win_overlap, plot_n=False):
    mean, _, sem, n = calc_overlapwindows(tc*1000, binsize=smooth_size, overlap = win_overlap)
    
    plt.figure()
    plt.plot(x, mean, c='r')
    plt.fill_between(x, mean - sem, mean + sem, alpha = 0.2, color = 'r')
    plt.legend(['{} Cell-Seizures, n = {}'.format(cell_label,len(tc))])
    plt.title('Timecourse for {} {} Cell Seizures with SEM, {}ms bins'.format(cell_label, cell_type, smooth_size))
    plt.xlabel('Time(ms)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.axvline(x=0,color = 'red', alpha=0.5)
    if plot_n:
        plt.figure()
        plt.title('n in each bin')
        numtimes = len(x)
        plt.plot(x[:numtimes-1], n, c='r')
        plt.plot(x[:numtimes-1], n, c='b')
        plt.xlabel('Time(ms)')
        plt.ylabel('Number of Data Points Included')
    
    return mean, sem

def plot_timecourses(arr1, arr2, xarr, arr1_label = 'array1', arr2_label = 'array2', win_overlap = 0.5,
                     period = 'onset', smooth_size='no', smooth_type = 'bins', error_type = 'sem', subset_label = ' ',plot_n=0):
    #arr1 and arr2 are 2D same-col-length arrays, divided by some predetermined property
    #xarr is the range of x-values that both arrays span
    #func plots both timecourses on the same graph, along with SEM
    
    if smooth_type == 'gaussian':
        arr1_mean = np.nanmean(arr1, axis=0)*1000
        arr2_mean = np.nanmean(arr2, axis=0)*1000
        if error_type == 'sem':
            arr1_err = stats.sem(arr1, axis=0, nan_policy = 'omit')
            arr2_err = stats.sem(arr2, axis=0, nan_policy = 'omit')
        elif error_type == 'std':
            arr1_err = np.nanstd(arr1, axis=0) #stdev
            arr2_err = np.nanstd(arr2, axis=0) #stdev
        arr1_mean = sliding_gaussian_window(arr1_mean, smooth_size)
        arr2_mean = sliding_gaussian_window(arr2_mean, smooth_size)
        
    elif smooth_type == 'bins':
        if error_type == 'sem':
            arr1_mean, _, arr1_err, arr1_n = calc_overlapwindows(arr1*1000, binsize=smooth_size, overlap = win_overlap)
            arr2_mean, _, arr2_err, arr2_n = calc_overlapwindows(arr2*1000, binsize=smooth_size, overlap = win_overlap)
        elif error_type == 'std':
            arr1_mean, arr1_err, _, arr1_n = calc_overlapwindows(arr1*1000, binsize=smooth_size, overlap = win_overlap)
            arr2_mean, arr2_err, _, arr2_n = calc_overlapwindows(arr2*1000, binsize=smooth_size, overlap = win_overlap)
    
    elif smooth_type == 'none':
        arr1_mean = np.nanmean(arr1,axis=0)
        arr1_err = stats.sem(arr1,axis=0)
        arr1_n = np.full((arr1.shape[1]-1,),arr1.shape[0])
        arr2_mean = np.nanmean(arr2,axis=0)
        arr2_err = stats.sem(arr2,axis=0)
        arr2_n = np.full((arr2.shape[1]-1,),arr2.shape[0])

    if len(arr1_mean) != len(arr2_mean):
        padded = np.array(pad_epochs_to_equal_length([arr1_mean, arr2_mean], np.nan, period))
        arr1_mean = padded[0,]
        arr2_mean = padded[1,]

    plt.figure()
    plt.plot(xarr, arr1_mean, c='r')
    plt.fill_between(xarr, arr1_mean - arr1_err, arr1_mean + arr1_err, alpha = 0.2, color = 'r')
    plt.plot(xarr, arr2_mean, c='b')
    plt.fill_between(xarr, arr2_mean - arr2_err, arr2_mean + arr2_err, alpha = 0.2, color = 'b')    
    plt.legend(['{} n = {}'.format(arr1_label,len(arr1)), '{} n = {}'.format(arr2_label, len(arr2))])
    plt.title('{}Timecourses for {} and {} with {} and {}ms bins'.format(subset_label, arr1_label, arr2_label, error_type, smooth_size))
    plt.xlabel('Time(ms)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.axvline(x=0,color = 'red', alpha=0.5)
    
    if plot_n:
        plt.figure()
        plt.title('n in each bin')
        numtimes = len(xarr)
        plt.plot(xarr[:numtimes-1], arr1_n, c='r')
        plt.plot(xarr[:numtimes-1], arr2_n, c='b')
        plt.xlabel('Time(ms)')
        plt.ylabel('Number of Data Points Included')

def calc_overlapwindows(array, binsize, overlap = 0.5):
    #0 overlap is no overlap; 1 overlap is complete overlap. 
    stride = int(binsize * (1-overlap))
    array_len = array.shape[1]
    array_mean = []; array_std = []; array_sem = []; array_n = []
    for i in range(0,array_len,stride):
        if i+binsize <= array_len:
            array_1d = np.nanmean(array[:,i:i+binsize], axis=1) #get mean for each row first
            array_mean.append(np.nanmean(array_1d))
            array_std.append(np.nanstd(array_1d))
            array_n.append(sum(~np.isnan(array_1d)))
            if np.nansum(array_1d) == 0:
                array_sem.append(np.nan)
            else:
                array_sem.append(stats.sem(array_1d, nan_policy = 'omit'))
    array_mean.append(np.nan); array_std.append(np.nan); array_sem.append(np.nan)
    return np.array(array_mean), np.array(array_std), np.array(array_sem), np.array(array_n)

def calc_convmean(array, smooth_size = 1000):
    #don't really use this, but it works... can convolve each row of a 2D array
    array_convolve = np.array(map(functools.partial(sliding_gaussian_window, 
                                                    time_window = smooth_size), array))
    array_mean = np.nanmean(array_convolve, axis=0)
    array_sem = stats.sem(array_convolve, axis=0, nan_policy = 'omit')
    return array_mean, array_sem

def get_sess_seizeeg(rec_name):
    sess_eeg = ld.sess_eeg_load(rec_name)
    seiztms = ld.sess_seiz_load(rec_name)
    num_seiz = len(seiztms)
    seiz_eeg = []
    for i in range(num_seiz):
        start = int(seiztms[i,0])
        end = int(seiztms[i,1])
        seiz_eeg.append(sess_eeg[start:end])
    return seiz_eeg

def get_seiz_eeg(seiz):
    #database = bd.load_in_dataframe()
    #seiz = td.extract_dataframe(database, Name = seiz_name)
    seiz_name= str(seiz['Name'])
    rec_name = str(seiz['recording_id'])
    sess_eeg = ld.sess_eeg_load(rec_name)
    seiztms = ld.single_seiz_load(seiz_name)
    start = int(seiztms[0]); end = int(seiztms[1])
    return sess_eeg[start:end]

def calc_spect(signal, window_size = 1000, window_type = 'hamming', sampfreq = 1000, overlap = 0.5, plot=0, seiz_name = 'seizure'):
    #window_size is in ms
    #see sig.get_window for window types
    #overlap should be expressed in proportion of overlap
    f,t,s = sig.spectrogram(signal, fs = sampfreq, 
                            window=sig.get_window(window_type, window_size), 
                            noverlap = window_size*overlap, return_onesided = True, 
                            scaling = 'spectrum')
    if plot:
        plot_spect(s,t,name=seiz_name)
    return f, t, s

def plot_spect(s, t, name = 'seizure',vmin='auto',vmax='auto'):
    plt.figure(figsize=(16,12))
    if isinstance(vmin,basestring):
        plt.imshow(s, origin='lower', aspect = 'auto')
    else:
        plt.imshow(s, origin='lower', aspect = 'auto', vmin=vmin, vmax=vmax)
    #plt.imshow(10*np.log(s), origin='lower', aspect = 'auto',vmin=vmin,vmax=vmax)#,cmap=plt.get_cmap('Spectral'))
#    plt.imshow(10*np.log(s), origin='lower', aspect = 'auto',vmin=-30,vmax=50,cmap=plt.get_cmap('Spectral'))
    cbar = plt.colorbar(); #cbar.ax.set_ylabel('Power Proportion Relative to Baseline', rotation=270);
    cbar.ax.get_yaxis().set_ticks_position('left')
#    plt.title('Spectrogram of ' + name)
    plt.xlabel('time (s)')
    plt.ylabel('frequency')
    xs = range(0,s.shape[1], 20)
    plt.xticks(xs, (np.array(list(t)[0::20]) - 0.5).astype(int))
    plt.show()

def calc_pwr(freq, time, spec, sampfreq=1000, lowfreq = 15, highfreq = 100, stat='mean',plot=0):
    #spike power 15 and 100
    #wave power 5 and 12
    #high gamma power 300 and 500
    freq_binsize = (sampfreq/2) / (len(freq)-1)
    freq_range = highfreq - lowfreq
    pwr = spec[int(math.ceil(lowfreq/freq_binsize)-1):(highfreq/freq_binsize)-1,]
    if stat=='mean':
        pwr_stat = np.sum(pwr, axis=0)  / freq_range
    elif stat == 'median':
        pwr_stat = np.nanmedian(pwr,axis=0)
    elif stat == 'sum':
        pwr_stat = np.sum(pwr, axis=0)
    return pwr_stat

def plot_spect_bands(freq, time, spec, sampfreq=1000, lowfreq = 15, highfreq = 100, plot=1):
    # similar to above function, but need to return SEM and don't want to disturb other code
    freq_binsize = (sampfreq/2) / (len(freq)-1)
    freq_range = highfreq - lowfreq
    pwr = spec[int(math.ceil(lowfreq/freq_binsize)-1):(highfreq/freq_binsize)-1,]
    pwr_mean = np.sum(pwr, axis=0)  / freq_range
    pwr_sem = stats.sem(pwr, nan_policy = 'omit')
    
    if plot:
        plt.figure(figsize=(16,10))
        plt.plot(time, pwr_mean, c='r')
        plt.fill_between(time, pwr_mean - pwr_sem, pwr_mean + pwr_sem, alpha = 0.2, color = 'r')
        plt.title('Normalized power in band from {} to {} Hz'.format(lowfreq,highfreq))
        plt.ylabel('Normalized power'); plt.xlabel('time')
        plt.axvline(0, color='k', linestyle='dashed')
        plt.show()
            
    return pwr_mean, pwr_sem
    
def calc_domfreq(freq, time, spec, sampfreq=1000, lowfreq=5, highfreq = 12):
    freqsperbin = freq[1]-freq[0] # number of frequencies in each bin equals the difference in frequencies between two consecutive bins
    lowfreqbin = np.where(freq == lowfreq)[0][0] # index of the bin containing the lower bound frequency
    highfreqbin = np.where(freq == highfreq)[0][0] # index of the bin containing the higher bound frequency    
    maxbins = np.argmax(spec[lowfreqbin:highfreqbin,], axis=0) # the locations of the highest-value bins within the desired band
    maxfreqbins = ((maxbins + lowfreqbin) * freqsperbin).astype(int)
    dom_freq = freq[maxfreqbins]
    return dom_freq

def calc_vrms(signal, binsize=0.5, overlap=0.25, Fs=1000):
#copied from cell_matching_utilities
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

def create_periodlog(times_array, sesslen):
    #useful for periods such as seizure or sleep where input has start and end times
    num_events = len(times_array)
    log = np.zeros(sesslen)
    for i in range(num_events):
        start = int(times_array[i,0])
        end = int(times_array[i,1])
        log[start:end] = 1
    return log

def create_eventlog(spktms,sesslen):
    #useful for marking events, such as spikes (cell spike, SWD spike, etc.)
    log = np.zeros(sesslen)
    if any(spktms > sesslen):
        spktms = spktms[spktms<sesslen]
    log[spktms] = 1
    return log

def pad_epochs_to_equal_length(data_list, pad_val, align = 'onset'):
    #Function by Jacob Prince. Adds NaN to create list of equal lengths (equal to longest)
    max_len = max(len(l) for l in data_list)
    
    data_list_padded = []
    for i in range(len(data_list)):
        this_epoch = data_list[i]
        this_len = len(this_epoch)
        temp_data = np.empty((max_len)); temp_data[:] = np.nan
        if align == 'onset':
            temp_data[:this_len] = this_epoch
            temp_data[this_len:] = pad_val
        elif align == 'offset':
            temp_data[-this_len:] = this_epoch
            temp_data[:-this_len] = pad_val
        data_list_padded.append(temp_data)
    return data_list_padded

def get_seiztimes_in_cell(cell_name):
    #given a cell, return the seizure times that fall within that cell's existence
    database = bd.load_in_dataframe()
    cell = td.extract_dataframe(database, Name = cell_name)
    rec_id = cell['recording_id'][0]
    
    seizs = td.extract_dataframe(database, Type = 'Seizure', recording_id = rec_id)
    if len(seizs) == 0:
        print('this recording has no seizures')
        return 'no'
    seizs = seizs[seizs[:,0] > cell['start']] #keep the seizures after first spike
    seizs = seizs[seizs[:,1] < cell['end']] #keep the seizures before the last spike
    
    return seizs
    
def get_seizs_in_cell(cell_name):
    #given a cell, return the seizures (in database) that exist in that cell 
    database = bd.load_in_dataframe()
    cell = td.extract_dataframe(database, Name = cell_name)
    rec_id = cell['recording_id'][0]
    seizs = td.extract_dataframe(database,  Type = 'Seizure', recording_id = rec_id)
    if len(seizs) == 0:
        print('this recording {} has no seizures'.format(rec_id))
        return 'no'
    
    seizs_in_cell = pd.DataFrame()
    for idx, seiz in seizs.iterrows():
        if np.array(seiz['start'] > cell['start']) and np.array(seiz['end'] < cell['end']):
            seizs_in_cell = seizs_in_cell.append(seiz)
    if len(seizs_in_cell) == 0:
        return 'no'
    return seizs_in_cell

def select_seizs_in_cell(cell_name, seizures):
    #given a cell, return the seizures (in database) that exist in that cell and in the list seizures
    database = bd.load_in_dataframe()
    cell = td.extract_dataframe(database, Name = cell_name)
    rec_id = cell['recording_id'][0]
    sess_seizures = seizures[seizures['Session']==rec_id]
    seizs = td.extract_dataframe(database,  Type = 'Seizure', recording_id = rec_id)
    seizs.sort_values(by=['start'], inplace = True)
    if len(seizs) == 0:
        print('this recording {} has no seizures'.format(rec_id))
        return 'no'
    
    seizs_in_cell = pd.DataFrame()
    counter = 0
    for idx, seiz in seizs.iterrows():
        if np.array(seiz['start'] > cell['start']) and np.array(seiz['end'] < cell['end']) and counter in sess_seizures['Seizure'].to_list():
            seizs_in_cell = seizs_in_cell.append(seiz)
        counter += 1
    if len(seizs_in_cell) == 0:
        return 'no'
    return seizs_in_cell

def get_sess_sztms(rec_id):
    #this gets the seiztimes in a session
    database = bd.load_in_dataframe()
    seizs = td.extract_dataframe(database, Type = 'Seizure', recording_id = rec_id)
    seiztms = np.zeros([len(seizs),2])
    seiztms[:,0] = np.array(seizs['start'])
    seiztms[:,1] = np.array(seizs['end']) 
    seiztms = seiztms[seiztms[:,0].argsort()]
    return seiztms

def get_sess_surroundsztms(rec_id, sesslen, surround_period, period):
    #period can be before or after
    seiztms = get_sess_sztms(rec_id)
    surroundseiztms = np.zeros_like(seiztms)
    surround_arr = np.zeros_like(seiztms)
    surround_arr[:,0] = np.repeat(surround_period,len(surroundseiztms))
    startidx = 1; endidx = len(seiztms)+1
    seiztms = np.vstack([[0,0], seiztms, [sesslen,sesslen]])
    if period == 'before':
        surroundseiztms[:,1] = seiztms[startidx:endidx,0] - 1 #seizure start times - 1ms
        surround_arr[:,1] = seiztms[startidx:endidx,0] - seiztms[0:endidx-1,1] # this seizure start - previous seizure end
        surround_time = np.amin(surround_arr, axis=1)
        surroundseiztms[:,0] = surroundseiztms[:,1] - surround_time + 1
    elif period == 'after':
        surroundseiztms[:,0] = seiztms[startidx:endidx,1] + 1 #seizure end times + 1ms
        surround_arr[:,1] = seiztms[2:endidx+1,0] - seiztms[startidx:endidx,1] #next seizure start - this seizure end
        surround_time = np.amin(surround_arr, axis=1)
        surroundseiztms[:,1] = surroundseiztms[:,0] + surround_time - 1
    return surroundseiztms

def calc_meanfr_fromtcs(tc_array):
    #given an array of timecourses, spit out mean fr and sem
    mean_frs = np.nanmean(tc_array, axis=1)
    mean_fr = np.nanmean(mean_frs)
    sem_fr = stats.sem(mean_frs, nan_policy = 'omit')

    return mean_fr, sem_fr

def get_fr_tcs(cell_df, seiz_df, onset_period = 5000,offset_period = 5000, 
               period = 'onset', within_cell = 0):
    database = bd.load_in_dataframe()
    all_seizures = td.extract_dataframe(database, Type = 'Seizure')
    analysis_len = onset_period+offset_period
    tc_list = []
    try:
        ids = cell_df['recording_id'].unique() #find unique recording id's for multiple cells
    except:
         ids = cell_df['recording_id'] #Catch single cell case

    for rec_id in ids:
#        break
        seizs = td.extract_dataframe(seiz_df,recording_id = rec_id,Type = 'Seizure') #seizures of interest
        if len(seizs) == 0:
            continue
        seiztms = np.zeros([len(seizs),2])
        seiztms[:,0] = np.array(seizs['start'])
        seiztms[:,1] = np.array(seizs['end'])   
        
        
        sess = td.extract_dataframe(database,recording_id = rec_id)
        #following line is temporary to exclude sessions with no lengths..
        if math.isnan(sess['Duration'][0]):
            continue
        sesslen = int(sess['Duration'][0]) #all "Duration" rows in sess should be the same
        all_sess_seizs = td.extract_dataframe(all_seizures, recording_id = rec_id) #all seizures in session
        all_seizstarts = np.array(all_sess_seizs['start'])
        all_seizends = np.array(all_sess_seizs['end'])
        num_seiz = len(seiztms)
        cells_in_rec = td.extract_dataframe(cell_df,recording_id = rec_id)
        slptms = ld.sess_slp_load(rec_id)
        if len(slptms) !=0:
            first_slp = slptms[0,0]; last_slp = slptms[int(len(slptms)-1),1]
            slpstarts = slptms[:,0]
            slpends = slptms[:,1] 
        else:
            first_slp=float('inf'); last_slp=0
        
        for i,cell in cells_in_rec.iterrows(): #iterate through cells in recording
#            break
            cell_tcs = []
            spktms = np.array(ld.single_cell_load(cell['Name']))
            spklog = create_eventlog(spktms, sesslen)
            first_spk = cell['start']
            last_spk = cell['end']
            
            #for trimming the seiz-of-interest times to only ones within the cell
            too_early = sum(seiztms[:,0] - first_spk < 0)
            too_late = sum(seiztms[:,1] - last_spk > 0)
            cell_seiztms = np.sort(seiztms, axis=0) #these will be seizures of interest during cell time
            if too_late != 0:
                cell_seiztms = cell_seiztms[:num_seiz-too_late,]
            if too_early !=0:
                cell_seiztms = cell_seiztms[too_early:,]
            num_cellseiz = len(cell_seiztms)

            for i in range(num_cellseiz): #iterate through seizures in recording
                seiz_dur =  cell_seiztms[i,1] - cell_seiztms[i,0]
                if period =='onset':
                    ref = int(cell_seiztms[i,0]) #ref is seiz start
                    if ref < first_spk or ref > last_spk: #if seiz is outside cell time
                        continue
                    #for seizure-related checks
                    if ref == min(all_seizstarts[all_seizstarts-first_spk > 0]): #if this is the first seizure in the cell
                        prev_sz_end = first_spk
                    else:
                        prev_sz_end = max(all_seizends[ref - all_seizends > 0])
                    #for sleep-related checks (skip seizures during sleep, find ending of last sleep episode in case interference with presz period)
                    if ref < first_slp: #if this is before any sleep happens
                        prev_slp_end = first_spk
                    else:
                        closest_prevslp = np.max(slptms[ref-slpstarts > 0,:], axis=0)
                        if ref < closest_prevslp[1] + 2000: #if sz occurs during sleep period or within 2s
                            continue
                        else:
                            prev_slp_end = closest_prevslp[1]
                elif period == 'offset':
                    ref = int(cell_seiztms[i,1]) #ref is seiz end
                    if ref < first_spk or ref > last_spk:
                        continue
                    #for seizure-related checks
                    if ref == max(all_seizends[all_seizends-last_spk < 0]): #if this is the last seizure
                        next_sz_start = last_spk
                    else:
                        next_sz_start = min(all_seizstarts[all_seizstarts - ref > 0])
                    #for sleep_related checks (skip seizures going back to sleep)
                    if ref > last_slp: #if seizure end is after all sleep
                        next_slp_start = last_spk
                    else:
                        closest_nextslp = np.min(slptms[slpends-ref > 0,:], axis=0)
                        if ref > closest_nextslp[0] - 2000: #if seizure occurs during sleep period or within 2s
                            continue
                        else:
                            next_slp_start = closest_nextslp[0]
                
                tc_start = max(ref - onset_period, first_spk) #if starts before cell starts, set to first spike
                tc_end   = min(ref + offset_period, last_spk) #make sure ends if the cell time ends
                tc_template = np.full(analysis_len,np.nan) #empty nan array

                if period == 'onset':
                    tc_end = int(min(ref + seiz_dur,tc_end)) #end of seizure, or end of period of interest
                    tc_start = int(max(tc_start, prev_sz_end, prev_slp_end)) #end of previous seizure, or full period of interest, or end of previous sleep
                elif period == 'offset':
                    tc_start = int(max(ref - seiz_dur,tc_start)) #start of seizure, or start of period of interest within seizure
                    tc_end = int(min(tc_end, next_sz_start, next_slp_start)) #end of postictal period of interest, or start of next seizure, or start of next sleep
                start_idx = onset_period - (ref - tc_start) #if -onset_period is index 0, where does this firing tc start?
                tc = np.insert(tc_template, start_idx, np.array([spklog[tc_start:tc_end]]).ravel()) #insert the spike train in the middle of the nan template
                tc = tc[:len(tc_template)] #cut off the extra nan's at the end
                cell_tcs.append(tc)
            if within_cell:
                cell_tcs = np.array(cell_tcs)
                cell_mean = np.nanmean(cell_tcs, axis=0)
                tc_list.append(cell_mean)
            else:
                tc_list.extend(cell_tcs)
    tc_array = np.array(tc_list)
    
    return tc_array

def seizures_from_slp(rec_id, seiz_df = 'nope'):
    slptms = ld.sess_slp_load(rec_id)
    if len(slptms) == 0: #if there's no sleep in this session
        return []
    slpstarts = slptms[:,0]
    if isinstance(seiz_df,basestring):
        database = bd.load_in_dataframe()
        seiz_df = td.extract_dataframe(database, recording_id = rec_id, Type = 'Seizure')
    slpseizs = []
    for idx,seiz in seiz_df.iterrows():
        if seiz['start'] < slpstarts[0]: #if sz occurs before any sleep
            continue
#        break
        closest_prevslp = np.max(slptms[seiz['start']-slpstarts > 0,:], axis=0)
        if seiz['start'] < closest_prevslp[1]: #if sz occurs during sleep period
            slpseizs.append(seiz['Name'])      
    return slpseizs

def get_slponly_fr_tcs(cell_df, seiz_df, onset_period = 5000,offset_period = 5000, 
               period = 'onset', within_cell = 0):
    database = bd.load_in_dataframe()
    all_seizures = td.extract_dataframe(database, Type = 'Seizure')
    analysis_len = onset_period+offset_period
    tc_list = []
    try:
        ids = cell_df['recording_id'].unique() #find unique recording id's for multiple cells
    except:
         ids = cell_df['recording_id'] #Catch single cell case

    for rec_id in ids:
#        break
        seizs = td.extract_dataframe(seiz_df,recording_id = rec_id,Type = 'Seizure') #seizures of interest
        if len(seizs) == 0:
            continue
        seiztms = np.zeros([len(seizs),2])
        seiztms[:,0] = np.array(seizs['start'])
        seiztms[:,1] = np.array(seizs['end'])   
        
        
        sess = td.extract_dataframe(database,recording_id = rec_id)
        #following line is temporary to exclude sessions with no lengths..
        if math.isnan(sess['Duration'][0]):
            continue
        sesslen = int(sess['Duration'][0]) #all "Duration" rows in sess should be the same
        all_sess_seizs = td.extract_dataframe(all_seizures, recording_id = rec_id) #all seizures in session
        all_seizstarts = np.array(all_sess_seizs['start'])
        all_seizends = np.array(all_sess_seizs['end'])
        num_seiz = len(seiztms)
        cells_in_rec = td.extract_dataframe(cell_df,recording_id = rec_id)
        
        slptms = ld.sess_slp_load(rec_id)
        if len(slptms) !=0:
            first_slp = slptms[0,0]; last_slp = slptms[int(len(slptms)-1),1]
            slpstarts = slptms[:,0]
            slpends = slptms[:,1] 
        else:
            continue
        
        for i,cell in cells_in_rec.iterrows(): #iterate through cells in recording
#            break
            cell_tcs = []
            spktms = np.array(ld.single_cell_load(cell['Name']))
            spklog = create_eventlog(spktms, sesslen)
            first_spk = cell['start']
            last_spk = cell['end']
            
            #for trimming the seiz-of-interest times to only ones within the cell
            too_early = sum(seiztms[:,0] - first_spk < 0)
            too_late = sum(seiztms[:,1] - last_spk > 0)
            cell_seiztms = np.sort(seiztms, axis=0) #these will be seizures of interest during cell time
            if too_late != 0:
                cell_seiztms = cell_seiztms[:num_seiz-too_late,]
            if too_early !=0:
                cell_seiztms = cell_seiztms[too_early:,]
            num_cellseiz = len(cell_seiztms)

            for i in range(num_cellseiz): #iterate through seizures in recording
                seiz_dur =  cell_seiztms[i,1] - cell_seiztms[i,0]
                if period =='onset':
                    ref = int(cell_seiztms[i,0]) #ref is seiz start
                    if ref < first_spk or ref > last_spk: #if seiz is outside cell time
                        continue
                    #for seizure-related checks
                    if ref == min(all_seizstarts[all_seizstarts-first_spk > 0]): #if this is the first seizure in the cell
                        prev_sz_end = first_spk
                    else:
                        prev_sz_end = max(all_seizends[ref - all_seizends > 0])
                    #for sleep-related checks (skip seizures outside sleep, find ending of last sleep episode in case interference with presz period)
                    if ref < first_slp: #if this is before any sleep happens
                        continue
                    else:
                        closest_prevslp = np.max(slptms[ref-slpstarts > 0,:], axis=0)
                        if ref > closest_prevslp[1] + 2000: #if sz not during sleep period or within 2s
                            continue
                        else:
                            slp_start = closest_prevslp[0]
                elif period == 'offset':
                    ref = int(cell_seiztms[i,1]) #ref is seiz end
                    if ref < first_spk or ref > last_spk:
                        continue
                    #for seizure-related checks
                    if ref == max(all_seizends[all_seizends-last_spk < 0]): #if this is the last seizure
                        next_sz_start = last_spk
                    else:
                        next_sz_start = min(all_seizstarts[all_seizstarts - ref > 0])
                    #for sleep_related checks (skip seizures going to awake)
                    if ref > last_slp: #if seizure end is after all sleep
                        continue
                    else:
                        closest_nextslp = np.min(slptms[slpends-ref > 0,:], axis=0)
                        if ref < closest_nextslp[0] - 2000: #if seizure occurs outside sleep period or within 2s of it
                            continue
                        else:
                            slp_end = closest_nextslp[1]
                
                tc_start = max(ref - onset_period, first_spk) #if starts before cell starts, set to first spike
                tc_end   = min(ref + offset_period, last_spk) #make sure ends if the cell time ends
                tc_template = np.full(analysis_len,np.nan) #empty nan array

                if period == 'onset':
                    tc_end = int(min(ref + seiz_dur,tc_end)) #end of seizure, or end of period of interest
                    tc_start = int(max(tc_start, prev_sz_end, slp_start)) #end of previous seizure, or full period of interest, or end of previous sleep
                elif period == 'offset':
                    tc_start = int(max(ref - seiz_dur,tc_start)) #start of seizure, or start of period of interest within seizure
                    tc_end = int(min(tc_end, next_sz_start, slp_end)) #end of postictal period of interest, or start of next seizure, or start of next sleep
                start_idx = onset_period - (ref - tc_start) #if -onset_period is index 0, where does this firing tc start?
                tc = np.insert(tc_template, start_idx, np.array([spklog[tc_start:tc_end]]).ravel()) #insert the spike train in the middle of the nan template
                tc = tc[:len(tc_template)] #cut off the extra nan's at the end
                cell_tcs.append(tc)
            if within_cell:
                cell_tcs = np.array(cell_tcs)
                cell_mean = np.nanmean(cell_tcs, axis=0)
                tc_list.append(cell_mean)
            else:
                tc_list.extend(cell_tcs)
    tc_array = np.array(tc_list)
    
    return tc_array

def plot_3_timecourses(arr1, arr2, arr3, xarr, arr1_label = 'array1', arr2_label = 'array2', arr3_label = 'array3', win_overlap = 0.5,
                     period = 'onset', smooth_size='no',  subset_label = ' ',plot_n=0):
    #arr1 and arr2 are 2D same-col-length arrays, divided by some predetermined property
    #xarr is the range of x-values that both arrays span
    #func plots both timecourses on the same graph, along with SEM
        
    arr1_mean, _, arr1_err, arr1_n = calc_overlapwindows(arr1*1000, binsize=smooth_size, overlap = win_overlap)
    arr2_mean, _, arr2_err, arr2_n = calc_overlapwindows(arr2*1000, binsize=smooth_size, overlap = win_overlap)
    arr3_mean, _, arr3_err, arr3_n = calc_overlapwindows(arr3*1000, binsize=smooth_size, overlap = win_overlap)
        
    
    padded = np.array(pad_epochs_to_equal_length([arr1_mean, arr2_mean, arr3_mean], np.nan, period))
    arr1_mean = padded[0,]
    arr2_mean = padded[1,]
    arr3_mean = padded[2,]

    plt.figure()
    plt.plot(xarr, arr1_mean, c='r')
    plt.fill_between(xarr, arr1_mean - arr1_err, arr1_mean + arr1_err, alpha = 0.2, color = 'r')
    plt.plot(xarr, arr2_mean, c='b')
    plt.fill_between(xarr, arr2_mean - arr2_err, arr2_mean + arr2_err, alpha = 0.2, color = 'b')    
    plt.plot(xarr, arr3_mean, c='g')
    plt.fill_between(xarr, arr3_mean - arr3_err, arr3_mean + arr3_err, alpha = 0.2, color = 'g')    
    
    plt.legend(['{} n = {}'.format(arr1_label,len(arr1)), '{} n = {}'.format(arr2_label, len(arr2)), '{} n = {}'.format(arr3_label, len(arr3))])
    plt.title('{}Timecourses for {} and {} and {} with {}ms bins'.format(subset_label, arr1_label, arr2_label, arr3_label, smooth_size))
    plt.xlabel('Time(ms)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.axvline(x=0,color = 'red', alpha=0.5)
    
    if plot_n:
        plt.figure()
        plt.title('n in each bin')
        numtimes = len(xarr)
        plt.plot(xarr[:numtimes-1], arr1_n, c='r')
        plt.plot(xarr[:numtimes-1], arr2_n, c='b')
        plt.plot(xarr[:numtimes-1], arr3_n, c='g')
        plt.xlabel('Time(ms)')
        plt.ylabel('Number of Data Points Included')
