#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:09:00 2020

@author: Cian McCafferty

This file is for loading (processed) data of the three types that are used in
the GAERS seizure severity project: seizure times, continuous voltage (LFP/EEG),
and spike times.
Functions can either load individual files (for a seizure or cell), or all files
corresponding to a particular recording session

Prerequisites: 
    1) Processed data: files must be stored as .pkl objects, in the case of cells
    and seizures, and .mat in the case of EEG.
    2) File name info: these functions must be directed to a particular file within
    the default (or specified) data directory. To do so, pass the file name as 
    the first argument (this can be obtained from the corresponding 'Name' field
    in the reference database).
    File name should look like "SWD_YYYMMDDTTTTTTXXXX.pkl"

"""

# 1: Import required libraries & modules - first generic & then project-specific
from os.path import expanduser
home = expanduser('~')

import sys # Allows the definition of the python search path
sys.path.insert(0,home+'/mnt/Data4/GAERS_Codes/SeizureCodes')
sys.path.insert(0,home+'/mnt/Data4/GAERS_Codes/DataExtraction')
sys.path.insert(0,home+'/mnt/Data4/GAERS_Codes/CellCodes')

import glob
import os # Functions facilitating directory navigation & searching
import h5py # for opening v7.3 mat files

import pickle as pk # tools for compressing & uncompressing files
import pandas as pd # a whole library of data analysis functions
import numpy as np # a whole library of high-level math definitions & structures
import scipy.io as scio # a specific scipy subset to load mat files

import build_database as bd # Contains tools to manipulate dataframe incl loading
import trim_database as td # Further tools to manipulate dataframe after creation
import SpikeTrainAnalysis as sta

import fnmatch as fm


#%% To load the spike times of a specified cell:

def single_cell_load(cell_name,data_dir = home+'/mnt/Data4/GAERS_Data/'):
    cell_file = data_dir + cell_name # construct the full path to target file
    with open(cell_file,'r') as p: # use 'with' to open file in read mode
        cell = pk.load(p) # extract data from pickle
    cell_dataframe = pd.DataFrame(cell.cell_data) # create a dataframe containing all cell info
    spk_times = cell_dataframe['Spk_Time'] # extract just the spike times from df
    return spk_times; # pass spike times as output argument

#%% To load the start and end times of a specified seizure:
    
def single_seiz_load(sz_name,data_dir = home+'/mnt/Data4/GAERS_Data/'):
    seiz_file = data_dir + sz_name
    with open(seiz_file,'r') as s:
        seiz = pk.load(s)
            
    seiz_times = pd.Series([seiz.start,seiz.end])
    return seiz_times

#%% To load the spike times of all cells in a specified session:
    
def sess_cell_load(rec_name): # rec_name should be the recording_id as listed in our reference file (database) 
    
    database = bd.load_in_dataframe() # get reference file 
           
    rec_cells = td.extract_dataframe(database,recording_id = rec_name, Type = 'Cell')
    # select metadata of those cells in the target session/recording
    cell_ids = rec_cells['Name'] # pulls out the names/IDs of each cell
    
    spk_times_sess = [] # creates an empty list
    for cell_id in cell_ids: # iterates through each cell in this session
        # spk_times = single_cell_load(cell_id) # uses the appropriate function to load spike traim
        spk_times, spk_log, cell_times = sta.load_cell(cell_id)
        spk_times_sess.append(spk_times) # appends the spike times to the list for this session  
    return spk_times_sess


#%% To load the spike times of cells of a given type in a specified session:
    
def cell_type_load(rec_name, cell_type = 'Cortical'): # rec_name should be the recording_id as listed in our reference file (database) 
    
    database = bd.load_in_dataframe() # get reference file 
           
    rec_cells = td.extract_dataframe(database,recording_id = rec_name, Type = 'Cell', label = cell_type)
    # select metadata of those cells in the target session/recording
    cell_ids = rec_cells['Name'] # pulls out the names/IDs of each cell
    
    spk_times_sess = [] # creates an empty list
    for cell_id in cell_ids: # iterates through each cell in this session
        spk_times = single_cell_load(cell_id) # uses the appropriate function to load spike traim
        spk_times_sess.append(spk_times.values.tolist()) # appends the spike times to the list for this session  
    return spk_times_sess

#%% To load the start and end times of all seizures in a specified session:
    
def sess_seiz_load(rec_name): # rec_name should be the recording_id as listed in our reference file (database) 
    
    database = bd.load_in_dataframe() # get reference file 
    
    rec_seizs = td.extract_dataframe(database,recording_id = rec_name, Type ='Seizure')
    # select metadata of those seizures in the target session/recording
    rec_seizs.sort_values(by=['start'], inplace = True)
    # and reorganize to start with first seizure
    
    seiz_ids = rec_seizs['Name'] # pulls out the names/IDs of each seizure object
    
    seiz_times_sess_series = [] # create empty series
    for seiz_id in seiz_ids:
        try:
            seiz_times = single_seiz_load(seiz_id) #  load individual seizure
            seiz_times_sess_series.append(seiz_times.values.tolist()) #  append it to series
        except EOFError:
            print('Seizure Empty')
    seiz_times_sess =np.array(seiz_times_sess_series) # when done, convert series to array
    return seiz_times_sess

#%% To load the start and end times of SPARED and IMPAIRED seizures in a specified session:

def seiz_sev_load(rec_name): # rec_name should be the recording_id as listed in our reference file (database) 
    
    database = bd.load_in_dataframe() # get reference file 
    
    rec_seizs = td.extract_dataframe(database,recording_id = rec_name, Type ='Seizure')
    # select metadata of those seizures in the target session/recording
    rec_seizs.sort_values(by=['start'], inplace = True)
    # and reorganize to start with first seizure
    
    if not rec_seizs.empty:
        impd = td.extract_dataframe(rec_seizs, label = 'Impaired') # Take impaired seiz objs from seiz database 
        sprd = td.extract_dataframe(rec_seizs, label = 'Spared') # Take spared seiz objs from seiz database
    
        impd_ids = impd['Name'] # pulls out the names/IDs of each IMPAIRED seizure object
        
        impd_times_series = [] # create empty series
        for impd_id in impd_ids:
            impd_times = single_seiz_load(impd_id) #  load individual impaired seizure
            impd_times_series.append(impd_times.values.tolist()) #  append it to series
        
        impd_times_sess =np.array(impd_times_series) # when done, convert series to array
    
        sprd_ids = sprd['Name'] # pulls out the names/IDs of each SPARED seizure object
        
        sprd_times_series = [] # create empty series
        for sprd_id in sprd_ids:
            sprd_times = single_seiz_load(sprd_id) #  load individual spared seizure
            sprd_times_series.append(sprd_times.values.tolist()) #  append it to series
        
        sprd_times_sess =np.array(sprd_times_series) # when done, convert series to array
    else:
        impd_times_sess = [];sprd_times_sess=[]
    
    return [impd_times_sess, sprd_times_sess];    

#%% To load the EEG voltage values from a specified session:
'''
def sess_eeg_load(rec_name, data_dir = "/mnt/Data4/AnnotateSWD/"): # rec_name should be the recording_id as listed in our reference file (database) 
    
    sess_folder = data_dir + rec_name # The target recording session folder
    os.chdir(sess_folder) # Move to that folder
    if len(rec_name) < 7:# If it was a Crunelli session
        for file in glob.glob("*eeg.mat"):
            eeg_struct = scio.loadmat(file, squeeze_me=True, struct_as_record = False)
            try:
                eeg_sess = eeg_struct[rec_name + "_eeg"].values #the values are one layer down in the dictionary
            except:
                try:
                    eeg_sess = eeg_struct[rec_name + "_lfp"].values
                except:
                    return 0
    else:
        for file in glob.glob("*cleanEEG.mat"):
            eeg_struct = scio.loadmat(file, squeeze_me=True, struct_as_record = False) # load that mat file
            eeg_sess = eeg_struct["cleanEEG"] # the values are under the cleanEEG key
    return eeg_sess
'''
def sess_eeg_load(rec_name, data_dir = home+"/mnt/Data4/AnnotateSWD/"): # rec_name should be the recording_id as listed in our reference file (database) 
    
    sess_folder = data_dir + rec_name # The target recording session folder
    os.chdir(sess_folder) # Move to that folder

    with h5py.File(glob.glob("*newCleanEEG.mat")[0], 'r') as f:
        for k, v in f.items():
            eeg_sess = np.array(v)

    eeg_sess = np.ravel(eeg_sess)
    # convert eeg into uV
    if len(rec_name) < 10:
        eeg_sess *= 1000
    else:
        eeg_sess *= 0.195
    return np.ravel(eeg_sess)

#%% To load the sleep times from a specified session:

def sess_slp_load(rec_name, data_dir = home+'/mnt/Data4/AnnotateSWD/'): # rec_name should be the recording_id as listed in our reference file (database) 
    
    sess_folder = data_dir + rec_name # The target recording session folder
    
    os.chdir(sess_folder) # Move to that folder
    
    
    for file in glob.glob("*slptms.txt"): # Check for a sleep times file
        if not file:
            print("No sleep labelled")
        else:
            slp_times_full = np.loadtxt(file) # Load that file
            if np.size(slp_times_full) == 0: # If there was no sleep in the session
                slp_times_sess = [] # Set it to empty
            elif slp_times_full.ndim == 1: # If there were was only one time
                slp_times_sess = np.reshape(1000*slp_times_full[[1,2]],(-1,2))
            elif np.size(slp_times_full,1) > 2: # Sleep times from Blum sessions have index & duration & are in s
                slp_times_sess = 1000*slp_times_full[:,[1,2]] # Take only the start and end times of sleep periods
            else:
                slp_times_sess = slp_times_full   # Sleep times from Crunelli have only start/end in ms  
    return slp_times_sess

#%% To load the SWC peak times from a specified session:

def sess_SWCpk_load(rec_name, data_dir = '/mnt/Data4/AnnotateSWD/'): # rec_name should be the recording_id as listed in our reference file (database) 
    
    sess_folder = data_dir + rec_name # The target recording session folder
    
    os.chdir(sess_folder) # Move to that folder
       
    for file in glob.glob("SWCpksNew.txt"): # Check for a peak times file
        if not file:
            print("No sleep labelled")
        else:
            SWC_times_full = np.loadtxt(file) # Load that file
            SWC_times_ms = SWC_times_full*1000 # Convert s to ms
            # exclude sleep times
            slptms = sess_slp_load(rec_name)
            inds_to_rm = np.array([])
            for slptm in slptms:
                start = slptm[0]
                end = slptm[1]
                inds_to_rm = np.concatenate((inds_to_rm,np.where((SWC_times_ms>start)&(SWC_times_ms<end))[0]))
            SWC_times_ms = np.delete(SWC_times_ms,inds_to_rm)
    return SWC_times_ms

#%% To remove sessions with no seizures and/or no (verified) cells from working database
        
def clean_liveDB(to_clean):

    cleaned = to_clean # create new database for output
    sess_ids = to_clean['recording_id'].unique() # find all unique session/recording IDs
    
    for idx, sess_id in enumerate(sess_ids): # Iterate through unique recording IDs
        database_sess = to_clean.loc[to_clean['recording_id'] == sess_id] # Create mini-database for this session
        if not 'Seizure' in database_sess.values or not 'Cell' in database_sess.values:
            # if it has either no seizure or no cell objects
            cleaned = cleaned[cleaned.recording_id != sess_id] # remove that session from the database
    
    return cleaned

#%% To add session durations to database
 
def db_add_durs(database):
           
    sess_ids = database['recording_id'].unique() # find all unique session/recording IDs
               
    for idx, sess_id in enumerate(sess_ids): # Iterate through unique recording IDs
    #%% Load continuous data (EEG) as this is the only full-session data type
        sess_eeg = sess_eeg_load(sess_id) # Load EEG from this session
        sess_dur = np.size(sess_eeg,0) # The duration of the session in ms is the number of elements
        
        sess_locs = database.index[database['recording_id'] == sess_id].tolist()
        # Find those rows of the database corresponding to the sessions in question
        database.loc[sess_locs, 'Duration'] = sess_dur
        # Add a duration value to those rows
    
    return database

