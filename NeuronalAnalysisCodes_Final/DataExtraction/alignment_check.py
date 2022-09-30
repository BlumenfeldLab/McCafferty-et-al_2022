#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:47:22 2020

@author: rjt37
"""
import sys
sys.path.insert(0,'/mnt/Data4/GAERS_Codes/SeizureCodes')
sys.path.insert(0,'/mnt/Data4/GAERS_Codes/DataExtraction')
sys.path.insert(0,'/mnt/Data4/GAERS_Codes/CellCodes')
import build_database as bd
import class_definitions as ccd
import SpikeTrainAnalysis as sta
import pandas as pd
import pickle as pk
import warnings
import numpy as np
from scipy import stats
import Cell_matching_utilities as cmu
import matplotlib.pyplot as plt
import struct as st
import numpy as np
import locate_files as lf


def extract_session_lfp(rec_id, datType='oct', seiz_times_id='sztms.txt', plot = 0):
    # This function gets the LFP for the specified recording ID
    
    # Get the paths of the files; mainly do this to check number of channels and their names
    session = '/mnt/Data4/AnnotateSWD/' + rec_id
    targets = lf.locate_bool(session, datType,seiz_times_id=seiz_times_id)
    
    if targets[0] ==0 or targets[1] == 0:
        raise Exception(' Data is missing from ' + session + '''.   Either 
                        seiztimes does not exist, or ''' + datType + ''' does 
                        not exist''')
    data_loc       = targets[3]
    num_chans    = len(data_loc)
    chan_num = 0
    
    for cur_chan in data_loc:
        with open(cur_chan,'rb') as b:
            file_content = b.read()
            values = st.unpack('h' * (len(file_content)/2), file_content)
            values = np.array(values)
        if (chan_num == 0):
            seiz_data = np.zeros([num_chans, len(values)])
        seiz_data[chan_num,] = values
        chan_num += 1
    
    if plot:
        plt.figure()
        for chan_num in range(num_chans):
            if (chan_num == 0):
                ax = plt.subplot(num_chans,1, chan_num+1)
                plt.setp(ax.get_xticklabels(), visible=False)
            ax = plt.subplot(num_chans,1, chan_num+1)
            plt.plot(seiz_data[chan_num,:])
        ax = plt.subplot(num_chans, 1, 1)
        plt.title('EEG in {} channels for recording {}'.format(num_chans, rec_id))
        plt.xlabel('Time (ms)')
        
    return seiz_data

def plot_labeled_times(rec_id):
    seiz_data = extract_session_lfp(rec_id)
    database = bd.load_in_dataframe()
    relevant = sta.extract_dataframe(database, recording_id = rec_id)
    cells = sta.extract_dataframe(relevant, Type = 'Cell')
    num_cells = len(cells)
    seizures = sta.extract_dataframe(relevant, Type = 'Seizure')
    
    #first plot the session LFP
    plt.figure()
    first_chan = seiz_data[0,:]
    plt.plot(first_chan) #just plot the first channel
    plt.title('EEG in Channel 1 for Recording {}'.format(rec_id))
    plt.xlabel('Time (ms)')
    plt.ylabel('microVolts')
    #now shade in the seizure times
    for index,seiz in seizures.iterrows(): #plot seizures
        plt.axvspan(seiz['start'],seiz['end'],facecolor='r', alpha=0.5)
    #now plot cell firing times
    high = max(first_chan)
    low = min(first_chan)
    linelength = (high - low) / 50
    colors = ['C{}'.format(i) for i in range(num_cells)]
    spike_row_start = int(low - linelength*0.5)
    spike_row_step = int(linelength * 1.2)
    spike_row_end = int(spike_row_start - ((num_cells+1)*spike_row_step))
    lineoffsets = range(spike_row_end, spike_row_start, spike_row_step)
    for idx, cell in cells.iterrows():
        spk_times,_,_= sta.load_cell(cell['Name']) #spk_times should be in recording time..
        plt.eventplot(spk_times, orientation='horizontal', colors = colors[idx], lineoffsets = lineoffsets[idx], linelengths = linelength)
            
        
    






