# -*- coding: utf-8 -*-
"""
15/06/2019 - Peter Vincent & Renee Tung
This Python file contains functions that extract spike waveforms for a given 
cell

"""
import sys
import os
import glob
import re
sys.path.insert(0, '/mnt/Data4/GAERS_Codes/DataExtraction')
sys.path.insert(0, '/mnt/Data4/GAERS_Codes/CellCodes')
import extract_utils as eu
import class_definitions as ccd
import pickle as pk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import struct as st
from scipy.stats import norm
import scipy.io as scio
from scipy.cluster.vq import vq, kmeans, whiten
import build_database as bd
import seaborn as sns
import locate_files as lf
import SpikeTrainAnalysis as sta
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.signal import argrelextrema
from scipy.linalg import svd
import scipy.stats as stats
import trim_database as td

def get_thalamic_cell_classification(cell, classification_dict='/mnt/Data4/GAERS_Data/ThalamicCellClassifications.pkl'):
    # Function for looking up the classification (excitatory, inhibitory) of a thalamic cell
    
    with open(classification_dict,'r') as p:
        matching_dict = pickle.load(p)
    cell_name = cell['Name']
    cell_handle = cell_name.split('_')[1]
    label = '0'
    for key in matching_dict:
        key_parts = key.split('_')
        key_handle = key_parts[0] + key_parts[1]
        if cell_handle == key_handle:
            label = matching_dict[key]
    
    if label == '1':
        return 'Excitatory'
    elif label == '2':
        return 'Inhibitory'
    elif label == '0':
        return 'None'

#%% To plot half-width and total duration for each waveform, colour coded by recording location

def plot_spk_width(database):
    # This function loads mean spike waveform data for each cell. It then calculates
    #the width at half maximum and total duration of the spike, and plots those
    #on the same figure as every other cell, colour-coded according to whether they
    #are cortical or thalamic.
    
    widthfig = plt.figure()
    plt.xlabel('Half-Width (ms)')
    plt.ylabel('Duration (ms)')
    
    data_loc = '/mnt/Data4/AnnotateSWD/'
    
    cell_db = td.extract_dataframe(database, Type = 'Cell')
    
    for idx, cell_series in c_cell_db.iterrows(): # go through each cell (each row of cell df)
       
        if cell_series.label == 'Cortical':
            # for cortical cells, the name of the waveform file is easy to find because
            # there is a unique name for each cell and waveforms were generated based on those
            fullname = cell_series.Name
            partname = fullname.split('.')
            rootname = partname[0] # find the unique name of this cell
            mean_wv = np.load(data_loc + cell_series.recording_id + '/' + rootname + '_wvfrm.npy')
            fS = 30 # cortical data sampled at 30 kHz
        else:
            # unfortunately for thalamic cells, mean waveforms were generated in MATLAB based
            #on the rat_changrp_clusnum unique naming system, which takes more effort to find
            # - in fact we need to load the cell object to do so 
        
            cell_pkl = '/mnt/Data4/GAERS_Data/' + cell_series.Name # the cell pickle is here
            with open(cell_pkl) as co: # we need to open the cell pickle to find cluster-specific name
                cell_data = pk.load(co)
        
            spk_info = cell_data.cell_data
            chan_grp = '_' + str(spk_info.Chan_Num.iloc[0]) # channel group is saved as Chan Num
            clu_num = '_' + str(spk_info.Cluster_Num.iloc[0]) # cluster number is saved as Cluster Num
    
            wvfrm_name = cell_series.recording_id + chan_grp + clu_num + '_wvfrm.mat' # with the addition of _wvfrm.mat
            wav_file = scio.loadmat('/mnt/Data4/AnnotateSWD/' + cell_series.recording_id + '/' + wvfrm_name) # load that matlab file
            mean_wv = wav_file['meanwvfrm'][0] # mean waveform is stored in variable meanwvfrm 
            fS = 20 # thalamic data sampled at 20 kHz

        wv_props = find_waveform_props(mean_wv, plot = 1) # find various props incl amplitude & half-width 
        
        if cell_series.label == 'Thalamic':
            halfwidth = wv_props[2]/fS
            duration = float(wv_props[4][0])/fS
            plt.plot(halfwidth,duration,'bo')
        else:
            halfwidth = wv_props[2]/fS
            duration = wv_props[4]/fS
            plt.plot(halfwidth,duration,'gd')


#%% To calculate mean representative waveforms for each cell and save them in a suitable format

def mean_rep_wvrm(database):
    # This function extracts spike waveform data from spk files for each cell.
    # It then sorts that into the appropriate number of channels, averages all 
    #spikes on each channel, and finds the channel with the largest amplitude
    #waveform. Finally, it saves the mean waveform from this channel.
    c_cell_db = td.extract_dataframe(database, Type ='Cell', label ='Cortical') # Take all cell series from database
    # only cortical cells require this extraction (thalamic already have means in mat files)
    
    
    for idx, cell_series in c_cell_db.iterrows(): # go through each cortical cell (each row of cell df)
        [waveforms, cell_data] = extract_cell_data(cell_series) # extract waveform matrix (channels x samples x spikes)
        max_wvfrm = source_channel_waveforms(waveforms) # extract channel dimension of largest mean amplitude
        rep_wvfrm = np.mean(max_wvfrm, axis = 0) # get the average waveform from this max channel
        
        fullname = cell_series.Name
        partname = fullname.split('.')
        rootname = partname[0]
        
        np.save('/mnt/Data4/AnnotateSWD/' + cell_series.recording_id + '/' + rootname + '_wvfrm.npy',rep_wvfrm)
        
        if not np.isnan(np.sum(rep_wvfrm)):
            wvfrm_props = find_waveform_props(rep_wvfrm) # find various props incl amplitude & half-width 
            np.save('/mnt/Data4/AnnotateSWD/' + cell_series.recording_id + '/wv_props.npy',wvfrm_props)
        else:
            print('No waveform for cell', cell_series.Name, ' animal', cell_series.Rat)

def extract_cell_data(cell_series,num_samples=30,
                      object_locations='/mnt/Data4/GAERS_Data/'):
    # This function extracts the spike waveform data, taking in as reference
    # a series from the database.  It returns a 3 dimensional array.
    # The first dimension spans the channels, the second spans the samples,
    # the third spans the individual spikes
    bytes_per_number = 2
    recording_id = cell_series.recording_id
    channel = int(cell_series.number) # find the group of channels on which the cell was found
    dat_location = eu.get_orig_data_from_session(recording_id)[0] # find location of raw data (spk file)
    extension = '.spk.' + str(channel) # the file will end in .spk.[group]
    spk_path = lf.locate_clusters(dat_location,{extension}) # this function searches for a specified extension (in this case .spk)
    spk_file_path = spk_path[extension][0] # the exact path to the file itself

    num_channels = int(64/total_spk_files(dat_location))  # the number of chans/grp = total chans (64) divided by no. of groups
    num_samples = 30 # cortical sessions (apart from Sil01) had 30 samples/spike


    if cell_series.Rat == 'Sil01':
        num_samples=32 #for the first animal only there were 32 samples per spk waveform
        
    # once we have the exact file path for both cortical & thalamic, and their differing numbers of channels & samples, we can proceeed unified    
        
    obj_name = object_locations + cell_series.Name # construct the location of the cell object
    with open(obj_name) as co:
        cell_data = pk.load(co) # load the cell object
    spk_info = cell_data.cell_data # this extracts df of info for each spike
    num_spikes = spk_info.shape[0]  # number of rows in the df is no of spikes
    waveforms  = np.empty([num_channels,num_samples,num_spikes]) # create empty 3d array for channels x samples x spikes
    with open(spk_file_path,'rb') as sd:
        for spk in range(num_spikes): # go through each spike in that cluster/cell
            spk_num_series = spk_info.iloc[spk] # get its info
            spk_num = spk_num_series.Spk_Num # get its number within all spikes of that cell
            file_offset = (spk_num-1)*num_samples*num_channels*bytes_per_number
            # think this ^ is to do with where the binary data in the target file will be
            if file_offset > os.path.getsize(spk_file_path): # if the seek location is beyond the end of the target file
                print('Attempting to load non-existent waveform from cell', cell_series.Name, ' animal', cell_series.Rat) # then there is no waveform there (usually because)
                waveforms[:] = np.nan # set the waveform to empty
            else:
            
                sd.seek(file_offset,0) # sets the position for loading according to the offset
                for sample in range(num_samples):
                    for chan in range(num_channels):
                        value = st.unpack('h',sd.read(2))
                        value = int(value[0])
                        waveforms[chan,sample,spk] = value
                    
    return [waveforms,cell_data]

def total_spk_files(dat_location):
    spk_extension = '.spk'
    spk_paths = lf.locate_clusters(dat_location,{spk_extension})
    spk_paths = spk_paths.get(spk_extension)
    spk_nums = np.empty([len(spk_paths)])
    for spk in range(len(spk_paths)):
        spk_name = spk_paths[spk]
        spk_nums[spk] = int(spk_name.split('.')[-1])
    return np.amax(spk_nums)

def average_chan_spk(waveforms,plot=0):
    num_channels = waveforms.shape[0]
    num_samples  = waveforms.shape[1]
    avg_chan     = np.average(waveforms,2)
    if plot:
        plt.figure()
        for chan in range(num_channels):
            plt.plot(time_axis(num_samples),avg_chan[chan,:])
            waveform_plot_labels()
    return avg_chan

def time_axis(num_samples):
    plot_range = np.arange(num_samples)
    time_range = sample_to_time(plot_range)
    return time_range

def waveform_plot_labels():
    plt.xlabel('Time (ms)')
    plt.ylabel(u'\u03bcV')

def sem_chan_spk(waveforms, plot=0, std=0):
    #Plots mean of an array with SEM shaded
    num_channels = waveforms.shape[0]
    num_samples  = waveforms.shape[1]
    avg_chan = np.average(waveforms,2)
    sem_chan = stats.sem(waveforms,axis=2)
    std_chan = np.std(waveforms,axis=2)
    err_chan = sem_chan
    if std:
            err_chan = std_chan
    if plot:
        plt.figure()
        for chan in range(num_channels):
            plt.plot(time_axis(num_samples),avg_chan[chan,:], 'r')
            if std:
                plt.fill_between(time_axis(num_samples),avg_chan[chan,:] - std_chan[chan,:],
                                 avg_chan[chan,:] + std_chan[chan,:], alpha = 0.4, color = 'r')
            else:
                plt.fill_between(range(num_samples),avg_chan[chan,:] - sem_chan[chan,:],
                         avg_chan[chan,:] + sem_chan[chan,:], alpha = 0.2, color = 'r')
    waveform_plot_labels()
    return avg_chan, err_chan

def std_sourcechan_plot(cell_name, ylims=(-1250,600)):
    database = bd.load_in_dataframe()
    cell_series = sta.extract_dataframe(database, Name=cell_name).iloc[0]
    waveforms = extract_cell_data(cell_series)[0]
    source_chan = cell_source_chan(waveforms)
    num_samples = waveforms.shape[1]
    avg_chan = np.average(waveforms,2)
    std_chan = np.std(waveforms,axis=2)
    plt.figure()
    plt.title(cell_name)
    plt.ylim(ylims)
    plt.plot(time_axis(num_samples),avg_chan[source_chan,:],'r')
    plt.fill_between(time_axis(num_samples),avg_chan[source_chan,:] - std_chan[source_chan,:],
                     avg_chan[source_chan,:] + std_chan[source_chan,:], alpha=0.2, color = 'r')
    waveform_plot_labels()
    ax = plt.gca()
    plt.text(0.99,0.99,'Number of Spikes: {}'.format(waveforms.shape[2],horizontalalignment='right',verticalalignment='top',transform = ax.transAxes))
    
    return avg_chan[source_chan,:], std_chan[source_chan,:]

def show_all_spks(waveforms,max_col=[1,0,1]):
    num_channels = waveforms.shape[0]
    num_samples  = waveforms.shape[1]
    num_spks     = waveforms.shape[2]
    fig = plt.figure(figsize=(5,num_channels*2))
    grid = plt.GridSpec(num_channels,1,hspace=0.2,wspace=0.2)
    for sub_plot in range(num_channels):
        cur_channel  = fig.add_subplot(grid[sub_plot,0])
        for spk in range(num_spks):
            scalar = (float(spk)/num_spks)
            col_vec =tuple([x * scalar for x in max_col])
            cur_channel.plot(time_axis(num_samples),
                             waveforms[sub_plot,:,spk],color=col_vec,
                             alpha=0.1)
        waveform_plot_labels()
            
def seizure_waveform(plot=0, plotNum = 5, **kwargs):
    #This function will get the mean spike waveform in seizure and outside of seizure
    database = bd.load_in_dataframe()
    database = database[database['Type'] == 'Cell']
    database = sta.extract_dataframe(database,**kwargs) #limit to cells of interest
    for index,cell in database.iterrows(): #for each cell
        spk_times,_,cell_times = sta.load_cell(cell['Name']) #load spike info
        _,seiz_logs = sta.load_seizures(cell['Name']) #load seizure info (seiz_times is in recording time, seiz_logs is in spike-existence time)
        waveforms = extract_cell_data(cell_series = cell)[0] #waveform data for this cell at each spike
        num_spikes = waveforms.shape[2]
        seiz_log = seiz_logs['seiz']

        if num_spikes != len(spk_times):
            raise Exception('waveform and spike data do not match')
        
        spk_times = spk_times - cell_times['start'] #spike times in cell time
        spk_szlog = np.empty([num_spikes])
        
        for spike in range(num_spikes): #for each spike
            this_spike_time = spk_times.iloc[spike]
            if seiz_log[this_spike_time] == 1: #if this spike is in seizure
                spk_szlog[spike] = 1
            else:
                spk_szlog[spike] = 0

        seizure_waveforms = waveforms[:,:,(spk_szlog == 1)]
        nonseizure_waveforms = waveforms[:,:,(spk_szlog == 0)]

        textstr = 'Cell ID: ' + cell['Name']
        seizure_avgchan = average_chan_spk(seizure_waveforms)
        nonseizure_avgchan = average_chan_spk(nonseizure_waveforms)
        
        if plot:
            ax = plt.subplot(111)
            sem_chan_spk(seizure_waveforms)
            plt.title('Average Waveform for Spikes during Seizure')
            plt.text(0.05, 0.95, textstr, verticalalignment = 'top', horizontalalignment = 'left', transform=ax.transAxes)
            
            ax = plt.subplot(112)
            sem_chan_spk(nonseizure_waveforms)
            plt.title('Average Waveform for Spikes during Non-Seizure')
            plt.text(0.05, 0.95, textstr, verticalalignment = 'top', horizontalalignment = 'left', transform=ax.transAxes)
        
        if index == plotNum:
            return seizure_waveforms, nonseizure_waveforms, seizure_avgchan, nonseizure_avgchan
    
    return seizure_waveforms, nonseizure_waveforms, seizure_avgchan, nonseizure_avgchan

def plot_seizure_waveforms(cell_name):
    [seizure_waveforms, nonseizure_waveforms, seizure_avgchan, nonseizure_avgchan] = seizure_waveform(plot=0, plotNum = 5, Name=cell_name)
    source_chan = cell_source_chan(seizure_waveforms)
    num_samples = seizure_waveforms.shape[1]
    
    plt.figure()
    p1 = plt.plot(time_axis(num_samples),seizure_avgchan[source_chan,:],'r',label='Seizure')
    p2 = plt.plot(time_axis(num_samples),nonseizure_avgchan[source_chan,:],'b',label='Non-seizure')
    plt.legend(loc='lower right')
    waveform_plot_labels()
    plt.title(cell_name)

def cell_source_chan(waveforms):
    #determines the source channel by finding which has the maximum displacement
    #of signal^2
    if waveforms.ndim == 2:
        avgchan = waveforms
    else:
        avgchan = average_chan_spk(waveforms)
    displacement = np.empty([avgchan.shape[0]])
    for n_chan in range(avgchan.shape[0]):
        if avgchan.ndim == 2:
            displacement[n_chan] = np.amax((avgchan[n_chan]**2))
        else: #pretty sure this won't happen, but just in case
            displacement[n_chan] = np.amax((avgchan[n_chan,:])**2)
    source_chan = np.argmax(displacement)
    return source_chan

def source_channel_waveforms(waveforms):
    #pulls out the source channel waveform only for the inputted waveform(s)
    if waveforms.ndim == 2:
        num_spikes = 1
    else:
        num_spikes = waveforms.shape[2]
    num_samples = waveforms.shape[1]
    source_waveforms = np.empty([num_spikes,num_samples])
    for n_spike in range(num_spikes): #for each spike
        if waveforms.ndim == 2:
            this_source = cell_source_chan(waveforms)
            source_waveforms[n_spike] = waveforms[this_source,:]
            source_waveforms = np.ravel(source_waveforms)
        else:
            this_source = cell_source_chan(waveforms[:,:,n_spike])
            source_waveforms[n_spike,:] = waveforms[this_source,:,n_spike]
    return source_waveforms

def plot_cell_spikes(cell_name, plotNum = 100):
    from random import sample
    database = bd.load_in_dataframe()
    database = database[database['Name'] == cell_name]
    for index,cell in database.iterrows(): #for each cell
        waveforms = extract_cell_data(cell_series = cell)[0] #waveform data for this cell at each spike
    num_spikes = waveforms.shape[2]
    num_samples = waveforms.shape[1]
    if num_spikes < plotNum:
        num_spikes = plotNum
    source_waveforms = source_channel_waveforms(waveforms)
    plot_these = sample(range(num_spikes),plotNum)
    plt.figure()
    plotted_waveforms = np.empty([plotNum,num_samples])
    for n_spike in range(len(plot_these)):
        plt.plot(time_axis(num_samples),source_waveforms[plot_these[n_spike]])
        plotted_waveforms[n_spike,:] = source_waveforms[plot_these[n_spike]]
    plt.title('Waveforms of ' + str(plotNum) + ' Spikes in ' + cell_name)
    plt.ylim(-1250,600)
    waveform_plot_labels()
    return plotted_waveforms

def save_waveform_to_dict(dict2= 'waveform_analysis', data_dir = '/mnt/Data4/GAERS_Data/', no_overwrite=1):
    database = bd.load_in_dataframe()
    database = database[database['Type'] == 'Cell']
    for index,cell in database.iterrows():
        cell_name = cell['Name']
        print('Saving waveform for ' + cell_name)
        cell_file = data_dir + cell_name
        if no_overwrite:
            with open(cell_file,'r') as p:
                cell_pk = pk.load(p)
            if cell_pk.properties.get(dict2)!=None:
                continue
        cell_waveforms = extract_cell_data(cell_series = cell)[0] #waveform data for this cell at each spike
        source_chan = cell_source_chan(waveforms = cell_waveforms)
        avg_waveforms, sem_waveforms = sem_chan_spk(cell_waveforms)
        sta.save_cell(cell_name, dict2, data_dir, source_idx = source_chan,
                      waveforms = avg_waveforms, waveforms_sem = sem_waveforms)

def sample_to_time(sample_time):
    ratio = 1.0/30.0 #30 samples in 1 ms
    ms_time = sample_time * ratio
    return ms_time

def open_saved_waveforms(cell_name):
    data_dir = '/mnt/Data4/GAERS_Data/'
    cell_file = data_dir + cell_name
    with open(cell_file,'r') as p:
        cell_info = pk.load(p)
        cell_waveform = cell_info.properties['waveform_analysis']
    return cell_waveform

def get_avg_cell_waveforms(save_new = 0):
    data_dir = '/mnt/Data4/GAERS_Data/'
    database = bd.load_in_dataframe()
    database = database[database['Type'] == 'Cell']
    waveform_panda = pd.DataFrame()
    min_num_samples = 1000 # setting min no of spikes that must be present per cell?
    for index,cell in database.iterrows():
#        break
        cell_name = cell['Name']
        cell_file = data_dir + cell_name
        with open(cell_file,'r') as p:
            cell_info = pk.load(p)
            cell_waveform = cell_info.properties['waveform_analysis']
        source_chan = cell_waveform['waveforms'][cell_waveform['source_idx'],:]
        cell_data = [[cell_name, cell['recording_id'], source_chan]]
        headers   = ['Name', 'recording_id', 'source_waveform']
        num_samples = source_chan.shape[0]
        if num_samples < min_num_samples:
            min_num_samples = num_samples
        new_data = pd.DataFrame(cell_data, columns = headers)
        waveform_panda = pd.concat([waveform_panda,new_data])
    waveform_panda = sta.fix_df_index(waveform_panda)
    
    odd = min_num_samples%2 #0 if even, 1 if odd
    
    for index,cell in waveform_panda.iterrows():
        if cell['source_waveform'].shape[0] != min_num_samples:
            source = cell['source_waveform']
            source_shape = source.shape[0]
            if source_shape%2 != odd:
                source = source[1:source_shape]
            while source.shape[0] != min_num_samples:
                source = source[1:(source.shape[0]-1)]
            waveform_panda.at[index,'source_waveform'] = source
    
    if save_new:
        filename = '/mnt/Data4/GAERS_Data/cell_waveform_dataframe.pkl'
        bd.save_dataframe(waveform_panda,save_dir = filename)
    return waveform_panda

'''
Check for if all waveforms are same length
for index,cell in waveform_panda.iterrows():
    if index == 0:
        min_size = cell['source_waveform'].shape[0]
        print(str(min_size))
    if cell['source_waveform'].shape[0] != min_size:
        print('incorrect array size')
        print(cell['Name'])
'''

def manual_check_waveforms(waveform_panda = 'all'):
    if np.any(waveform_panda == 'all'):
        waveform_panda = get_avg_cell_waveforms()
    for index,cell in waveform_panda.iterrows():
        if index > 100 and index < 105:
            plt.figure(10)
#            plt.plot(cell['source_waveform'])
            if index == 104:
                plt.plot(time_axis(cell['source_waveform'].shape[0]),cell['source_waveform'],c='r')
            else:
                plt.plot(time_axis(cell['source_waveform'].shape[0]),cell['source_waveform'],c='b')
            plt.title(cell['Name'])
            waveform_plot_labels()
            plt.show(10)
#        raw_input('Press key: ')
#        plt.close()

'''
Manually checking the waveform for each cell. Initially set idx=0
idx +=1
cell = waveform_panda.iloc[idx]
plt.figure(10)
plt.plot(cell['source_waveform'])
plt.title(cell['Name'])
plt.ylim(-1250,600)
plt.show(10)
print(cell['Name']) 

gw.std_sourcechan_plot(cell['Name'])
plt.title(cell['Name'])
gw.plot_cell_spikes(cell['Name'])
plt.title(cell['Name'])
plt.show()
'''

def plot_all_cell_waveforms(waveform_panda = 'all'):
    if isinstance(waveform_panda,basestring):
        waveform_panda = bd.load_in_dataframe('/mnt/Data4/GAERS_Data/cell_waveform_dataframe.pkl')
    for index,cell in waveform_panda.iterrows():
        plt.figure(12)
        num_samples = cell['source_waveform'].shape[0]
        plt.plot(time_axis(num_samples),cell['source_waveform'])
    plt.title('Average Waveforms from Source Channels for All Cells')
    plt.ylim(-1250,600)
    waveform_plot_labels()

def plot_all_chans(cell_name):
    #Checking all of the channels for the specified cell
    plt.figure()
    cell_waveforms = open_saved_waveforms(cell_name)['waveforms']
    for n_chan in range(cell_waveforms.shape[0]):
        plt.plot(time_axis(cell_waveforms.shape[1]),cell_waveforms[n_chan,:])
    plt.title(cell_name + ' channel waveforms')
    plt.ylim(-1250,600)
    waveform_plot_labels()
    plt.show()
    
def find_waveform_props(channel_waveform, plot=0):
    #this function finds properties for a given waveform
    num_samples = channel_waveform.shape[0]
    peak = np.nan; next_extrema = np.nan; peak_idx = np.nan; next_extrema_idx = np.nan;
    peak_idx = np.argmax(channel_waveform**2)
    peak = channel_waveform[peak_idx]
    
    peaks = argrelextrema(channel_waveform, np.greater)[0]
    troughs = argrelextrema(channel_waveform, np.less)[0]
    extrema = np.concatenate((peaks,troughs), axis=None)
    extrema = np.sort(extrema)
    
    peak_in_extrema = np.where(extrema == peak_idx)[0]
    if peak_in_extrema+1 >= extrema.shape[0]:
        next_extrema_idx = peak_idx + np.argmin(abs(np.diff(channel_waveform[peak_idx:num_samples])))
    else:
        next_extrema_idx = extrema[peak_in_extrema+1]
    next_extrema = channel_waveform[next_extrema_idx]
    if peak_in_extrema-1 < 0:
        prev_extrema_idx = peak_idx - np.argmin(abs(np.diff(channel_waveform[0:peak_idx])))
    else:
        prev_extrema_idx = extrema[peak_in_extrema-1]
    prev_extrema = channel_waveform[prev_extrema_idx]
    
    symmetry = (prev_extrema - next_extrema) / (abs(prev_extrema) + abs(next_extrema))
    duration = next_extrema_idx - prev_extrema_idx
    
    peak_latency = next_extrema_idx - peak_idx
    amplitude = next_extrema - peak
    half_peak = peak/2
    quarter_peak = peak*.25
    
    left_half_intersect = find_half_peak_intersection(channel_waveform, peak_idx, half_peak, move_left = 1)
    right_half_intersect = find_half_peak_intersection(channel_waveform, peak_idx, half_peak, move_left = 0)
    half_width = right_half_intersect[0] - left_half_intersect[0]
    
    quarter_left_half_intersect = find_half_peak_intersection(channel_waveform, peak_idx, quarter_peak, move_left = 1)
    quarter_right_half_intersect = find_half_peak_intersection(channel_waveform, peak_idx, quarter_peak, move_left = 0)
    quarter_width = quarter_right_half_intersect[0] - quarter_left_half_intersect[0]
    
    if plot:
        plt.figure()
        plt.plot(time_axis(num_samples),channel_waveform)
        plt.scatter(sample_to_time(peak_idx),peak)
        plt.scatter(sample_to_time(next_extrema_idx),next_extrema)
        plt.scatter(sample_to_time(prev_extrema_idx),prev_extrema)
        plt.scatter(sample_to_time(left_half_intersect[0]), left_half_intersect[1])
        plt.scatter(sample_to_time(right_half_intersect[0]), right_half_intersect[1])
        plt.scatter(sample_to_time(quarter_left_half_intersect[0]), quarter_left_half_intersect[1])
        plt.scatter(sample_to_time(quarter_right_half_intersect[0]),quarter_right_half_intersect[1])
        waveform_plot_labels()
        ax = plt.gca()
#        plt.text(0.57,-450,'Amplitude: {} microvolts \nPeak Latency: {} ms \nHalf-width: {} ms \nSymmetry: {} \nDuration: {} ms \nQuarter-width: {}'
#                 .format(amplitude[0], sample_to_time(peak_latency)[0], sample_to_time(half_width), symmetry[0], sample_to_time(duration)[0], sample_to_time(quarter_width),
#                 horizontalalignment='right',verticalalignment='top',transform = ax.transAxes))
        # the above is commented because some variables sometimes emerge as floats and sometimes as single-dimension arrays
    return np.array([amplitude, peak_latency, half_width, symmetry, duration, quarter_width])

'''
def find_fancy_waveform_props(channel_waveform, plot=0):
    from scipy.interpolate import UnivariateSpline
    num_samples = channel_waveform.shape[0]
    x_range = range(num_samples)
    waveform_spline = UnivariateSpline(range(num_samples),channel_waveform)
    deriv_1 = waveform_spline.derivative(n=1)
    deriv_2 = waveform_spline.derivative(n=2)
    
    if plot:
        plt.figure()
        plt.plot(channel_waveform)
        plt.plot(x_range,waveform_spline(x_range))
        plt.plot(x_range, deriv_1(x_range))
        plt.plot(x_range, deriv_2(x_range))
'''

def find_half_peak_intersection(channel_waveform, peak_idx, half_peak, move_left = 0):
    move = np.nan; end = np.nan
    if move_left == 1:
        move = -1
        end = 0
    else:
        move = 1
        end = channel_waveform.shape[0]
    channel_wave_orig = channel_waveform
    channel_waveform = abs(channel_waveform)
    half_peak_orig = half_peak
    half_peak = abs(half_peak)
    surround_idx = np.empty([2])
    for n_sample in range(peak_idx,end,move):
        point = channel_waveform[n_sample]
        if point < half_peak:
            surround_idx[0] = n_sample
            surround_idx[1] = n_sample - move
            break
    surround_idx = np.sort(surround_idx)
    
    point_pair_line = np.array([[int(surround_idx[0]), channel_wave_orig[int(surround_idx[0])]],
                                   [int(surround_idx[1]),channel_wave_orig[int(surround_idx[1])]]])
    half_peak_line = np.array([[int(surround_idx[0]),half_peak_orig],[int(surround_idx[1]), half_peak_orig]])
    x,y = line_intersection(line1 = point_pair_line, line2 = half_peak_line)
    return np.array([x,y])

def line_intersection(line1, line2):
    #function from stackoverflow, modified
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       print ('do not intersect')
       return np.nan, np.nan

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

def plot_features(prop1_name='half_width', prop2_name='symmetry',waveform_panda='all'):
    prop_list = ['amplitude', 'peak_latency', 'half_width', 'symmetry', 'duration', 'quarter_width', 'preseiz_firing', 'seiz_firing']
    prop1_idx = prop_list.index(prop1_name)
    prop2_idx = prop_list.index(prop2_name)
    prop_labels = ['amplitude (microvolts)', 'peak_latency (ms)', 'half-width (ms)', 'symmetry', 'duration (ms)', 'quarter-width (ms)', 'pre-seizure firing rate (spikes/s)', 'seizure firing rate (spike/s)']
    if isinstance(waveform_panda, basestring):
        waveform_panda = bd.load_in_dataframe(dataframe = '/mnt/Data4/GAERS_Data/cell_waveform_dataframe.pkl')
    prop1 = np.empty([waveform_panda.shape[0]])
    prop2 = np.empty([waveform_panda.shape[0]])
    plt.figure()
    plt.xlabel(prop_labels[prop1_idx])
    plt.ylabel(prop_labels[prop2_idx])
#    skinny = []
    for index,cell in waveform_panda.iterrows():
        source_chan = cell['source_waveform']
        props = np.ones([len(prop_list)])
        props[0:6] = find_waveform_props(source_chan)
        if prop1_name == 'preseiz_firing' or prop2_name == 'preseiz_firing' or prop1_name == 'seiz_firing' or prop2_name == 'seiz_firing':
            data_dir = '/mnt/Data4/GAERS_Data/'
            cell_file = data_dir + cell['Name']
            with open(cell_file,'r') as p:
                cell_info = pk.load(p)
                cell_fr = cell_info.properties['spike_rate_analysis']
            props[6] = cell_fr['preseiz_fr']
            props[7] = cell_fr['seiz_fr']
        if prop1_name == 'peak_latency' or prop1_name == 'half_width' or prop1_name == 'duration' or prop1_name == 'quarter_width':
            prop1[index] = sample_to_time(props[prop1_idx])
        else:
            prop1[index] = props[prop1_idx]
#        if prop1[index] > 0.3:
#            print(cell['Name'])
#            skinny.append(cell['Name'])
        if prop2_name == 'peak_latency' or prop2_name == 'half_width' or prop2_name == 'duration' or prop2_name == 'quarter_width':
            prop2[index] = sample_to_time(props[prop2_idx])
        else:
            prop2[index] = props[prop2_idx]
        
        plt.scatter(prop1[index],prop2[index])
#        if sample_to_time(props[2]) > 0.1:
#            plt.scatter(prop1[index],prop2[index],c='r')
#        else:
#            plt.scatter(prop1[index],prop2[index],c='b')
#    plt.scatter(prop1, prop2)
    plt.title(prop1_name + ' vs ' + prop2_name)
#    return skinny
    
def plot_all_features(waveform_panda = 'all'):
    prop_list = ['amplitude', 'peak_latency', 'half_width', 'symmetry', 'duration', 'quarter_width', 'preseiz_firing', 'seiz_firing']
    num_props = len(prop_list)
    if isinstance(waveform_panda, basestring):
        waveform_panda = bd.load_in_dataframe(dataframe = '/mnt/Data4/GAERS_Data/cell_waveform_dataframe.pkl')
    for n_prop in range(num_props-1):
        for m_prop in range(n_prop+1,num_props):
            plot_features(prop1_name = prop_list[n_prop], prop2_name = prop_list[m_prop],waveform_panda=waveform_panda)

def waveform_panda_to_matrix():
    waveform_panda = bd.load_in_dataframe(dataframe = '/mnt/Data4/GAERS_Data/cell_waveform_dataframe.pkl')
    num_samples = waveform_panda.iloc[0][1].shape[0]
    waveform_matrix = np.empty([waveform_panda.shape[0],num_samples])
    for index,cell in waveform_panda.iterrows():
        waveform_matrix[index,:] = cell['source_waveform']
    return waveform_matrix    

def waveform_props_in_panda(waveform_panda='all', num_props=8):
    waveform_prop_pd = pd.DataFrame()
    if isinstance(waveform_panda, basestring):
        waveform_panda = bd.load_in_dataframe(dataframe = '/mnt/Data4/GAERS_Data/cell_waveform_dataframe.pkl')
    for index,cell in waveform_panda.iterrows():
        prop_array = np.empty([num_props])
        data_dir = '/mnt/Data4/GAERS_Data/'
        cell_file = data_dir + cell['Name']
        source_chan = cell['source_waveform']
        prop_array[0:6] = find_waveform_props(source_chan)
        with open(cell_file,'r') as p:
            cell_info = pk.load(p)
            cell_fr = cell_info.properties['spike_rate_analysis']
        prop_array[6] = cell_fr['preseiz_fr']
        prop_array[7] = cell_fr['seiz_fr']
        
        seiz_data = [[cell['Name'], prop_array[0], sample_to_time(prop_array[1]),sample_to_time(prop_array[2]),prop_array[3],sample_to_time(prop_array[4]),prop_array[5],prop_array[6]]]
        headers   = ['cell_name', 'amplitude', 'peak_latency', 'half_width', 'symmetry', 'duration', 'quarter_width', 'preseiz_firing', 'seiz_firing']
        
        new_data = pd.DataFrame(seiz_data, columns = headers)
        waveform_prop_pd = pd.concat([waveform_prop_pd,new_data])
    return waveform_prop_pd

'''
def find_narrow_waveform_cellnames(cutoff):
    waveform_prop_pd = waveform_props_in_panda()
    narrow_list = []
    for index,cell in waveform_prop_pd.iterrows():
        if cell['half_width'] < 0.1:
            narrow_list.append(cell['cell_name'])
    return narrow_list

'''

def waveform_props_to_matrix(waveform_panda='all', num_props=8):
    if isinstance(waveform_panda, basestring):
        waveform_panda = bd.load_in_dataframe(dataframe = '/mnt/Data4/GAERS_Data/cell_waveform_dataframe.pkl')
    num_cells = waveform_panda.shape[0]
    prop_matrix = np.empty([num_cells,num_props])
    for index,cell in waveform_panda.iterrows():
        data_dir = '/mnt/Data4/GAERS_Data/'
        cell_file = data_dir + cell['Name']
        source_chan = cell['source_waveform']
        prop_matrix[index,0:6] = find_waveform_props(source_chan)
        prop_matrix[index,1:3] = sample_to_time(prop_matrix[index,1:3])
        prop_matrix[index,4:5] = sample_to_time(prop_matrix[index,4:5])
        with open(cell_file,'r') as p:
            cell_info = pk.load(p)
            cell_fr = cell_info.properties['spike_rate_analysis']
        prop_matrix[index,5] = cell_fr['preseiz_fr']
        prop_matrix[index,6] = cell_fr['seiz_fr']
    return prop_matrix

def plot_prop_histograms():
    prop_matrix = waveform_props_to_matrix()
    prop_list = ['amplitude', 'peak_latency', 'half_width', 'symmetry', 'duration', 'quarter_width', 'preseiz_firing', 'seiz_firing']
    for prop in range(prop_matrix.shape[1]):
        plt.figure()
        plt.title(prop_list[prop] + ' Histogram for All Cells')
        plt.hist(prop_matrix[np.isfinite(prop_matrix[:,prop]),prop],20)
        plt.xlabel(prop_list[prop])
        plt.ylabel('Number of Cells')

def cluster_waveform_props(prop_matrix, num_centroids=3):
#    prop_list = ['amplitude', 'peak_latency', 'half_width', 'symmetry']
    centroids,_ = kmeans(prop_matrix,num_centroids)
    index,_ = vq(prop_matrix,centroids)
#    from mpl_toolkits.mplot3d import Axes3D #For 3D plotting
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d') #3D subplot
#    ax.scatter(prop_matrix[:,0],prop_matrix[:,1],prop_matrix[:,2]) #Scatter plot of data
#    ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c='r')#Scatter plot of centroids in red
#    ax.set_xlabel()

#%% correlations, k-means clustering, and silhouette values

def corr_vector(waveforms, remove_duplicate = 0, plot=0):
    #calculates the correlation matrix and converts it to a vector
    #if select remove_duplicate, returns a vector without the lower triangle of the matrix
    corr_mat = np.corrcoef(waveforms) #correlation matrix
    corr_vec = corr_mat.reshape(-1,1) #reshape to 1 row vector
    if remove_duplicate:
        upper_mat = np.triu(corr_mat)
        upper_vec = upper_mat.reshape(-1,1)
        corr_vec = upper_vec[upper_vec != 0].reshape(-1,1)
    if plot:
        plt.figure()
        plt.hist(corr_vec,200)
    return corr_vec

def cluster_waveforms(corr_vec, num_clusters = 3):
    #function that clusters
    num_waveforms = int((corr_vec.shape[0])**(1/2))
    centroids,_ = kmeans(corr_vec,num_clusters) #finds the centroids
    index,_ = vq(corr_vec,centroids) #observation indices and corresponding centroid
    index_mat = index.reshape(num_waveforms,num_waveforms)
    
    return centroids, index_mat

#def calc_best_num_clusters(waveforms, source_only = 1, num_to_test = range(2,10), times_to_calc = 10, plot=0):
#    #silhouette analysis
#    silhouette_avgs = []
#    for n_cluster in num_to_test:
#        corr_vec = corr_vector(waveforms, remove_duplicate=1)
#        clusterer = KMeans(n_clusters = n_cluster,random_state=times_to_calc)
#        cluster_labels = clusterer.fit_predict(corr_vec)
#        silhouette_avgs.append(silhouette_samples(corr_vec,cluster_labels))
#   
##        for i in range(times_to_calc):
##            clusterer = KMeans(n_clusters = n_cluster,random_state=times_to_calc)
##            cluster_labels = centroids.fit_predict(corr_vec)
##            silhouette_avgs[times_to_calc+1,n_cluster] = silhouette_score(corr_vec,cluster_labels)
##    if plot:
##        plt.figure()
##        for i in times_to_calc:
##            plt.scatter(silhouette_avgs[0],silhouette_avgs[i+1])
#    return silhouette_avgs
        

#def get_spike_corr(index_mat,centroid_idx):
#    points = [] #list of points
#    for x in range(index_mat.shape[0]):
#        for y in range(x,index_mat.shape[0]):
#            if index_mat[x,y] == centroid_idx:
#                points.append([x,y])
#    return np.array(points)

#def plot_clusters(waveforms, source_only=1, plot_corr_hist=1):
#
#    #plot_high_corr: 
#    highcorr_idx = np.argmax(centroids)
#    highcorr_pts = get_spike_corr(index_mat,highcorr_idx)
#    plt.figure(5)
#    for point in range(highcorr_pts.shape[0]):
#        plt.plot([0,1], [highcorr_pts[point,0],highcorr_pts[point,1]])
#    
#    #plot_low_corr:
#    lowcorr_idx = np.argmin(centroids)
#    lowcorr_pts = gw.get_spike_corr(index_mat,lowcorr_idx)
#    plt.figure(5)
#    plt.scatter(lowcorr_pts[:,0], lowcorr_pts[:,1], s=0.5)

#%%
'''
Code for finding source channels of individual channels

def source_channel(waveform, middle_value = 0, median_chan = 0):
    #outputs the channel # that is determined to be the source channel for the given waveform
    num_chans = waveform.shape[0]
    displacement = np.empty([num_chans])
    if middle_value:
        for chan in range(num_chans):
            displacement[chan] = waveform[chan,14]
        source_chan = np.argmax(displacement)
        return source_chan
    elif median_chan:    
        peak = np.empty([num_chans]); trough = np.empty([num_chans]); 
        peak_idx = np.empty([num_chans]); trough_idx = np.empty([num_chans]); 
        max_distance = np.empty([num_chans])
        for chan in range(num_chans):
            peak[chan],peak_idx[chan],trough[chan],trough_idx[chan],max_distance[chan] = find_peak_trough(waveform[chan,:])
        peak_loc = int(np.median(peak_idx))
        trough_loc = int(np.median(trough_idx))
        for chan in range(num_chans):
            displacement[chan] = waveform[chan,peak_loc] - waveform[chan,trough_loc]
    else: 
        for chan in range(num_chans):
            displacement[chan] = np.amax((waveform[chan,:])**2)
    source_chan = np.argmax(displacement)
    return source_chan #, peak_loc

def source_channel_waveforms(waveforms):
    #pulls out the source channel waveform only for the inputted waveform(s)
    if waveforms.ndim == 2:
        num_spikes = 1
    else:
        num_spikes = waveforms.shape[2]
    num_samples = waveforms.shape[1]
    new_waveforms = np.empty([num_spikes,num_samples])
    # where_is_peak = np.empty([num_spikes])
    for n_spike in range(num_spikes): #for each spike
        if waveforms.ndim == 2:
            this_source = source_channel(waveforms)
            new_waveforms[n_spike] = waveforms[this_source,:]
            new_waveforms = np.ravel(new_waveforms)
        else:
            this_source = source_channel(waveforms[:,:,n_spike])
            #this_source, where_is_peak[n_spike] = source_channel(waveforms[:,:,n_spike])
            new_waveforms[n_spike,:] = waveforms[this_source,:,n_spike]
    return new_waveforms#, where_is_peak


#plt.figure()
#plt.hist(where_is_peak)
#plt.title('histogram of peak timing')
#plt.ylabel('spikes')


def plot_sourcechan_num(waveforms):
    #for a given cell, plot which channel is the source across all of the spikes
    num_spikes = waveforms.shape[2]
    sourcechan_array = np.empty([num_spikes])
    for n_spike in range(num_spikes): #for each spike
        sourcechan_array[n_spike] = source_channel(waveforms[:,:,n_spike])
    plt.figure()
    plt.scatter(range(num_spikes),sourcechan_array)
    plt.title('Source Channel for Each Spike')
    plt.xlabel('Spike'); plt.ylabel('Source Channel')
    plt.figure()
    plt.hist(sourcechan_array)
    plt.title('Number of Spikes per Source Channel')
    plt.xlabel('Source Channel'); plt.ylabel('Number of Spikes')
    return sourcechan_array

def plot_all_channel_waveforms(waveform):
    num_channels = waveform.shape[0]
    num_samples = waveform.shape[1]
    plt.figure()
    ax = plt.subplot(111)
    for n_channel in range(num_channels):
        plt.plot(range(num_samples),waveform[n_channel,:])
    plt.legend(range(num_channels))
    source_chan = source_channel(waveform)
    source_txt =  'Source channel: ' + str(source_chan)
    plt.text(0.05, 0.95, source_txt, verticalalignment = 'top', horizontalalignment = 'left', transform=ax.transAxes)


#for n_waveform in range(waveforms.shape[2]):
#    gw.plot_all_channel_waveforms(waveforms[:,:,n_waveform])
#    if n_waveform == 5:
#        break
#


def plot_all_peaks_troughs(waveform):
    for n_channel in range(waveform.shape[0]):
        find_peak_trough(channel_waveform = waveform[n_channel,:], plot=1)
        plt.title('channel number: ' + str(n_channel))

#plt.figure()
#for i in range(1925):
#    plt.plot(range(30),source_waveforms[i,:])
#plt.title('source channels for all spikes')
#plt.figure()
#plt.plot(range(30),np.average(source_waveforms,0))
#plt.title('mean waveform from source channels')

'''
'''
#%% Singular Value Decomposition (SVD)

def SVD_stuff(source_waveforms):
    U, s,test2  = svd(np.matrix.transpose(source_waveforms))
    reconstruct = np.matrix(U[:, :2]) * np.diag(D[:2]) * np.matrix(VT[:2,:])


'''

    
