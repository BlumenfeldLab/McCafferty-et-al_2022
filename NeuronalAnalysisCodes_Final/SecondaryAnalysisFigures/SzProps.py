#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 08:54:33 2021

This function is for getting seizure-by-seizure properties (firing rate and rhythmicity)
It also gets matching pre-seizure properties

@author: cpm49
"""


from os.path import expanduser
home = expanduser('~')

import sys # Allows the definition of the python search path
sys.path.insert(0,home+'/mnt/Data4/GAERS_Codes/SeizureCodes')
sys.path.insert(0,home+'/mnt/Data4/GAERS_Codes/DataExtraction')
sys.path.insert(0,home+'/mnt/Data4/GAERS_Codes/CellCodes')
sys.path.insert(0,home+'/mnt/Data4/GAERS_Codes/AnalysisCodes')
import get_behav_preds as gbp
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import build_database as bd
import SpikeTrainAnalysis as sta
import scipy.stats as ss
import scipy.signal as sig
import LoadData as ld # Tools to load each of our data types
import trim_database as td # Tools to edit our master dataframe
import os
import pickle as pk


import trim_database as td # Further tools to manipulate dataframe after creation


window = 20000 # we want 5 seconds of pre-seizure data, starting 10 seconds before the seizure
winoff = 5000

data_dir = home+'/mnt/Data4/GAERS_Data/'

database = bd.load_in_dataframe(dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl') # get reference file

database = bd.clean_liveDB(database) # remove sessions that don't have both cells and seizures


# First load group attributions of each location

with open(home+'/mnt/Data4/CtxCellClasses.pkl','rb') as f1:
    cortical_assignments = pk.load(f1)
    
cortical_assignments = cortical_assignments.rename(columns={'Cells':'Cell'})
            
with open(home+'/mnt/Data4/ThalCellClasses.pkl','rb') as f2:
    thalamic_assignments = pk.load(f2)
    
thalamic_assignments = thalamic_assignments.rename(columns={'Cells':'Cell'})

assignments = cortical_assignments.append(thalamic_assignments)


# Get all cell information
cell_df = database.loc[(database.Type == 'Cell')]
cortcell_df = database.loc[(database.Type == 'Cell') & (database.label == 'Cortical')]
thalcell_df = database.loc[(database.Type == 'Cell') & (database.label == 'Thalamic')]


cell_sess = cell_df['recording_id'].unique() # find all unique session/recording IDs

SzProps = pd.DataFrame([], columns = ['Label','Animal','Cell','Firing Rate','Rhythmicity','State','T Rhythm','T Firing'])


for sess in cell_sess:

    celsess_db = td.extract_dataframe(database,recording_id = sess, Type = 'Cell') # take out database rows for this session
    dur = int(celsess_db.iloc[0]['Duration']) # any of these will have the appropriate session duration
    rat = celsess_db.iloc[0]['Rat']
    label = celsess_db.iloc[0]['label']

    # First load the required data types: spike times and seizure times
    sess_spks = ld.sess_cell_load(sess)
    
    sess_szs = (ld.sess_seiz_load(sess)).astype(int)
    
    sess_slp = ld.sess_slp_load(sess) #  load sleep on/off times as np array
    if type(sess_slp) == list:
        sess_slp = np.array([[0,0],[0,0]])
    sess_slp = sess_slp.astype(int)
    
    sess_cells = cell_df.loc[cell_df.recording_id== sess]
    
    location = sess_cells.label.iloc[0]
    
    sess_dur = int(sess_cells.Duration.iloc[0])
    rat = sess_cells.Rat.iloc[0]
    
    namecount = 0
    
# Then iterate through each cell
    for cell in sess_spks:
        cell = [x for x in cell if x <= sess_dur] # any spikes at times after we have EEG should be excluded
        cell = np.array(cell)
        
        cell_szs = np.copy(sess_szs) # get a version of seizure and sleep times just for this cell
        cell_slp = np.copy(sess_slp)

        cell_szs = cell_szs[np.invert(np.any(cell_szs>sess_dur,axis=1)),:]    # get rid of any events that happen after our last spike  
        cell_slp = cell_slp[np.invert(np.any(cell_slp>sess_dur,axis=1)),:]
        
        cellname = sess_cells.iloc[namecount]['Name']
                
        location = sess_cells.iloc[namecount]['label']
        namecount += 1
        
        spk_log = np.zeros(sess_dur) # create a logical to align spikes with states
        spk_log[cell] = 1 # for every spike set the corresponding logical element to 1
  
        ISIs = np.diff(cell)
        ISIs = np.insert(ISIs,0,cell[0])
        missingend = ISIs>30000 # find times when there was no spiking for 30s or more, presuming cell was absent at these times 
        if sum(missingend)>0: #  if there were any such times
            missingstrt = np.roll(missingend,-1) # then find the location of spikes just before those
            
            missingspk0 = cell[missingstrt] # find the spike before the gap       
            missingspk1 = cell[missingend] #  and the spike after the gap
                    
            if missingspk0[0]>missingspk1[0]: #  if the first start is after the first end
                missingspk0=np.insert(missingspk0,0,0) #  then the real first start should be at zero
                missingspk0 = np.delete(missingspk0,-1) #  and get rid of the last 'start'
            missingspks = np.vstack([missingspk0,missingspk1]).transpose() #  put the starts and ends together
            for gap in missingspks: #  and iterate through them
                spk_log[gap[0]:gap[1]] = np.nan # to set each interval between a start and an end to nan

        onpad = np.empty(window) # create a padding array to make sure first seizure is at least one window from session start    
        onpad[:] = np.nan # set it to nan
        spk_log = np.insert(spk_log,0,onpad) # add it to the start of the spike logical
        spk_log = np.append(spk_log,onpad) # and to the end to make sure there's space there too       
        cell_szs += window # and add its length to the start and end times of seizures
        cell_slp += window

        sz_log = np.zeros(len(spk_log)).astype(int) # create a logical for all times in seizure
        for idx, seizure in enumerate(cell_szs): # for each seizure (row of seiz)
            sz_log[cell_szs[idx,0]:cell_szs[idx,1]] = 1 # set elements from start to end = 1

        slp_log = np.zeros(len(spk_log)).astype(int)
        for idx, sleep in enumerate(cell_slp): # for each sleep bout (row of slp)
            slp_log[cell_slp[idx,0]:cell_slp[idx,1]] = 1 # set elements from start to end = 1
        slp_log[sz_log.astype(bool)] = 0  #  any time designated seizure can't be sleep
        
        bl_log = np.ones(len(spk_log)).astype(int) # also a logical for all non-seizure, non-sleep, nonpre/post seizure times
        bl_log[sz_log.astype(bool)] = 0
        bl_log[slp_log.astype(bool)] = 0
        
        bl_spks = np.copy(spk_log)
        bl_spks[~bl_log.astype(bool)] = np.nan
        
        SzFR = [] # empty variable
        SzRHY = []
        BlFR = []
        BlRHY = []

        for seiz in cell_szs:
            if np.sum(np.isnan(bl_spks[seiz[0]-window:seiz[0]-winoff])) < window/2:
                blfr = np.nansum(bl_spks[(seiz[0]-window):(seiz[0]-winoff)])/(window-winoff)*1000
                blrhy = 1/(np.nanstd(np.diff(np.where(bl_spks[seiz[0]-window:seiz[0]-winoff]==1)))/np.nanmean(np.diff(np.where(bl_spks[seiz[0]-window:seiz[0]-winoff]==1))))
                BlFR.append(blfr)
                BlRHY.append(blrhy)
                
                szfr = np.nansum(spk_log[seiz[0]:seiz[1]])/(seiz[1]-seiz[0])*1000 # mean firing rate (Hz) in the seizure is total number of spikes divided by duration in secons
                szrhy = 1/(np.nanstd(np.diff(np.where(spk_log[seiz[0]:seiz[1]]==1)))/np.nanmean(np.diff(np.where(spk_log[seiz[0]:seiz[1]]==1))))
                # above is mean rhythmicity for the whole seizure: std of inter-spike intervals divided by mean value of inter-spike intervals
                SzFR.append(szfr)
                SzRHY.append(szrhy)
        
        BlRHY = np.array(BlRHY)
        BlFR = np.array(BlFR)
        SzRHY = np.array(SzRHY)
        SzFR = np.array(SzFR)
        
        BlRHY[np.isinf(BlRHY)] = np.nan
        SzRHY[np.isinf(SzRHY)] = np.nan
        
        wf,pf = ss.ttest_rel(BlFR, SzFR,nan_policy='omit')
        wr,pr = ss.ttest_rel(BlRHY, SzRHY,nan_policy='omit')

        SzProps = SzProps.append({'Label':label,'Animal':rat,'Cell':cellname,'Firing Rate':np.nanmean(SzFR),'Rhythmicity':np.nanmean(SzRHY),'State':'Seizure','T Rhythm': pr,'T Firing':pf},ignore_index=True)
        SzProps = SzProps.append({'Label':label,'Animal':rat,'Cell':cellname,'Firing Rate':np.nanmean(BlFR),'Rhythmicity':np.nanmean(BlRHY),'State':'Wake','T Rhythm':pr, 'T Firing':pf},ignore_index=True)
      
        
SzProps.to_pickle(home+'/mnt/Data4/MakeFigures/TestForOD/SzProps.pkl')
           

SzProps1 = pd.read_pickle(home+'/mnt/Data4/MakeFigures/TestForOD/SzProps.pkl')        
 


with open(home+'/mnt/Data4/CtxCellClasses.pkl','rb') as f1:
    cortical_assignments = pk.load(f1)
    
cortical_assignments = cortical_assignments.rename(columns={'Cells':'Cell'})
            
with open(home+'/mnt/Data4/ThalCellClasses.pkl','rb') as f2:
    thalamic_assignments = pk.load(f2)
    
thalamic_assignments = thalamic_assignments.rename(columns={'Cells':'Cell'})

assignments = cortical_assignments.append(thalamic_assignments)

SzProps = SzProps.merge(assignments,left_on = 'Cell', right_on = 'Cell')

SzProps.drop(SzProps[SzProps['Classes'] == 'Miscellaneous'].index, inplace= True)


locations = ['Cortical','Thalamic']
patterns = SzProps.Classes.unique()
           
##Dot plots with connecting lines for wake and seizure only
jitter = 0.05

for location in locations:
    loc_SzProps = SzProps.loc[SzProps.Label == location]
    fig, ax = plt.subplots()
    
    offset = 0

    for pattern in patterns:   
        pat_SS = loc_SzProps.loc[loc_SzProps.Classes == pattern]
        
        offset+=1
        
        based = pat_SS.loc[pat_SS['State'] == 'Wake']
        col1 = based['Rhythmicity']
        x_jitter1 = pd.Series(np.random.normal(loc=0, scale=jitter, size=col1.shape))+offset
        ax.plot(x_jitter1, col1, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='r')
        
        offset+=1
        
        seized = pat_SS.loc[pat_SS['State'] == 'Seizure']
        col2 = seized['Rhythmicity']
        x_jitter2 = pd.Series(np.random.normal(loc=0, scale=jitter, size=col2.shape))+offset
        ax.plot(x_jitter2, col2, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='b')
        
        for idx,pair in enumerate(col1): # for each pair of values from a neuron
            ax.plot([x_jitter1[idx],x_jitter2[idx]],[col1.iloc[idx],col2.iloc[idx]], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
            
        col1=col1.reset_index(drop=True)
        col2=col2.reset_index(drop=True)
        
        
        basemore = np.nansum((col1 > col2) & (pat_SS.iloc[::2,:]['T Rhythm'].reset_index(drop=True) < 0.05)).astype('float') # all pairs that have a significantly higher wake than seiz rhythmicity
        seizmore = np.nansum((col1 < col2) & (pat_SS.iloc[::2,:]['T Rhythm'].reset_index(drop=True) < 0.05)).astype('float') # all pairs that have a significantly higher seiz than wake rhythmicity
        nochange = np.nansum(pat_SS.iloc[::2,:]['T Rhythm'].reset_index(drop=True) >= 0.05).astype('float')
        count=float(col1.shape[0])
        print(location + ' ' + pattern + ' increases = ' + "{:.2f}".format(seizmore) + ' (' + "{:.2f}".format(seizmore/count*100) + '%), decreases = ' + "{:.2f}".format(basemore) + ' (' + "{:.2f}".format(basemore/count*100) + '%), no change = ' + "{:.2f}".format(nochange) + ' (' + "{:.2f}".format(nochange/count*100) + '%)') 
              
    ax.set_xticks(np.arange(1,offset+1)[0::2]+0.5)
    ax.set_xticklabels(patterns)
    fig.suptitle(location + ' Rhythmicity')

            