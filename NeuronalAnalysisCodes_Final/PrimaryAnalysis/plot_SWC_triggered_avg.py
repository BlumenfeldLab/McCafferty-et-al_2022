#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:59:58 2020

@author: jl3599

Adapted from Phase_analysis.py
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
import scipy.stats as stats
import LoadData as ld # Tools to load each of our data types
import trim_database as td # Tools to edit our master dataframe
import os
import pickle as pk

#%% Analysis: firing distribution of cells around SWC peaks
#   Go through each session, load all SWC peaks, exclude those outside seizure boundaries,
#   load all cells, for each SWC peak during which a cell was present take all spike data
#   Output will be by cell - for each cell we'll have a single vector of mean spikes around
#   SWC peak

def plot_SWC_avg(period='start'):
    '''
    Plots the firing distribution of cells around SWC peaks.
    
    Args:
        period = 'seiz', 'start' (first 1s of seiz vs rest), 
                 'baseline' (baseline vs 1s preictal)
    '''
    pSWCth = pd.DataFrame([], columns = ['Cell','Type','Cluster','nSeiz','Location','Class','Animal','AllSeizures','LongSeizures','ShortSeizures','LargeSeizures','SmallSeizures'])
    
    window = 400 # 200 ms either side of SWC peak
    database = bd.load_in_dataframe() # get the reference dataframe
    database = bd.clean_liveDB(database) # remove sessions that don't have both cells and seizures
    sess_ids = database['recording_id'].unique() # find all unique session/recording IDs
    counter = 0
    
    with open(home+'/mnt/Data4/CtxCellClasses.pkl','rb') as f1:
        cortical_assignments = pk.load(f1)
            
    with open(home+'/mnt/Data4/ThalCellClasses.pkl','rb') as f2:
        thalamic_assignments = pk.load(f2)
            
    cortical_cells = cortical_assignments['Cells'].to_list()
    thalamic_cells = thalamic_assignments['Cells'].to_list()
    
    cortical_classes = cortical_assignments['Classes'].to_list()
    thalamic_classes = thalamic_assignments['Classes'].to_list()
    
    frequencies_cortical = {'Onset Peak': 141, 
                            'Sustained Increase': 140,
                            'Sustained Decrease': 137,
                            'No Change': 140, # estimate
                            'Miscellaneous': 140} # estimate
    
    frequencies_thalamic = {'Onset Peak': 144, 
                            'Sustained Increase': 152,
                            'Sustained Decrease': 150,
                            'No Change': 150,
                            'Miscellaneous': 150} # estimate
            
    for sess in sess_ids:# iterate thru recording sessions
        print(str(counter) + '/' + str(len(sess_ids)))
        celsess_db = td.extract_dataframe(database,recording_id = sess, Type = 'Cell') # take out database rows for this session
        
        dur = int(celsess_db.iloc[0]['Duration']) # any of these will have the appropriate session duration
        rat = celsess_db.iloc[0]['Rat']
        label = celsess_db.iloc[0]['label']
    
        seiz = ld.sess_seiz_load(sess) # load seizure on & off times as np array
        
        pad = np.empty(window/2)  
        pad[:] = np.nan
        sz_log = np.zeros(dur+pad.size) # create a logical to align spikes with states
        sz_log_start = np.zeros(dur+pad.size)
        sz_log_rest = np.zeros(dur+pad.size)
        sz_log_baseline = np.ones(dur+pad.size)
        sz_log_pre = np.zeros(dur+pad.size)
        
        for idx, seizure in enumerate(seiz): # for each seizure (row of seiz)
            sz_log[int(seiz[idx,0]):int(seiz[idx,1])] = 1 # set elements from start to end = 1 for seizure
            sz_log_baseline[max(0,int(seiz[idx,0])-2000):min(dur,int(seiz[idx,1])+2000)] = 0 # set elements from start-2000 to end+2000 = 0 for baseline
            
            # set elements = 1 for start and rest
            if (seiz[idx,1] - seiz[idx,0] <= 1000):
                sz_log_start[int(seiz[idx,0]):int(seiz[idx,1])] = 1
            else:
                sz_log_start[int(seiz[idx,0]):int(seiz[idx,0])+1000] = 1
                sz_log_rest[int(seiz[idx,0])+1000:int(seiz[idx,1])] = 1
            
            # set elements = 1 for preictal
            if idx > 0 :
                if int(seiz[idx,0]) - int(seiz[idx-1,1]) >= 1000:
                    sz_log_pre[max(0,int(seiz[idx,0])-1000):int(seiz[idx,0])] = 1
                else:
                    sz_log_pre[int(seiz[idx-1,1]):int(seiz[idx,0])] = 1
            else:
                sz_log_pre[max(0,int(seiz[idx,0])-1000):int(seiz[idx,0])]
    
        
        pks = ld.sess_SWCpk_load(sess) # load SWC peak times as np array
        pks = [x for x in pks if x > window/2]        
        pks = np.array(pks).astype(int)
        
        # take the peaks within the appropriate time period
        if period == 'start':
            seiz_starts = seiz[:,0]
            seiz_ends = seiz[:,1]
            to_remove = []
            not_to_remove = []
            for pk_ind in range(pks.shape[0]):
                this_pk = pks[pk_ind]
                try:
                    start_ind = np.where(seiz_starts<=this_pk)[0][-1]
                    start_time = seiz_starts[start_ind]
                    if this_pk > start_time + 1000:
                        to_remove.append(pk_ind)
                    elif this_pk <= seiz_ends[start_ind]:
                        not_to_remove.append(pk_ind)
                except IndexError:
                    print(sess)
                    print('pk = ' + str(this_pk))
                    print('start times in [' + str(seiz_starts[0]) + ',' + str(seiz_starts[-1]) + ']')
            to_remove = np.array(to_remove)
            pks_start = np.delete(pks, to_remove)
            pks_rest = np.delete(pks, not_to_remove)
        
        spks = ld.sess_cell_load(sess) # load spike times as list of one np array per cell in session
            
        for idx, cell in enumerate(spks): # go through each cell
            
            name = celsess_db.iloc[idx]['Name'] # get the cell's unique name

            
            cell = np.array(cell) # convert to numpy array for ease of use
            
            cell = cell[cell<=dur] # remove any spikes that occur after the end of other data
            
            _,_,cell_times= sta.load_cell(name) 
            seiz_valid = seiz[np.where(seiz[:,0]>=cell_times['start'])]
            seiz_valid = seiz_valid[np.where(seiz_valid[:,1]<=cell_times['end'])]
            nseiz = seiz_valid.shape[0]
            
            spk_log = np.zeros(dur)  # create a logical for spike presence
            spk_log[cell] = 1 # put a one for every spike
            spk_log[:cell[0]] = np.nan # set to nan before cell's first spike
            spk_log[cell[-1]:] = np.nan # and after cell disappears 
            spk_log = np.append(spk_log,pad)                
            spk_log_start = np.copy(spk_log)
            spk_log_rest = np.copy(spk_log)
            spk_log_baseline = np.copy(spk_log)
            spk_log_pre = np.copy(spk_log)
            spk_log[~sz_log.astype(bool)] = np.nan  # we're only interested here in spikes during seizure
            
            
            spk_log_start[~sz_log_start.astype(bool)] = np.nan
            spk_log_rest[~sz_log_rest.astype(bool)] = np.nan
            spk_log_baseline[~sz_log_baseline.astype(bool)] = np.nan
            spk_log_pre[~sz_log_pre.astype(bool)] = np.nan
            
            if name in cortical_cells:
                this_class = cortical_classes[cortical_cells.index(name)]
                this_frequency = frequencies_cortical[this_class]
            elif name in thalamic_cells:
                this_class = thalamic_classes[thalamic_cells.index(name)]
                this_frequency = frequencies_thalamic[this_class]
            else:
                print(name + ' is not in classified.')
                
            if period == 'baseline':
                # generate fake peaks for baseline and preictal
                fake_pks = np.arange(spk_log.shape[0]/this_frequency)*this_frequency
                
                seiz_starts = seiz[:,0]
                seiz_ends = seiz[:,1]
                baseline_inds = []
                pre_inds = []
                for pk_ind, this_pk in enumerate(fake_pks):
                    try:
                        start_ind = np.where(seiz_starts<=this_pk)[0][-1]
                        if start_ind < len(seiz_starts):
                            start_time = seiz_starts[start_ind+1]
                            if this_pk >= start_time - 1000:
                                pre_inds.append(pk_ind)
                        
                            if this_pk < start_time - 2000 and this_pk > seiz_ends[start_ind] + 2000:
                                baseline_inds.append(pk_ind)
                        else:
                            start_ind = np.where(seiz_starts<=this_pk)[0][-1]
                            if this_pk > seiz_ends[start_ind] + 2000:
                                baseline_inds.append(pk_ind)
                    except IndexError:
                        if this_pk >= seiz_starts[0] - 1000:
                            pre_inds.append(pk_ind)
                        if this_pk < seiz_starts[0] - 2000:
                            baseline_inds.append(pk_ind)
                pks_pre = fake_pks[pre_inds]
                pks_baseline = fake_pks[baseline_inds]

            xvals = np.arange(-200,200, step =1)
            
            if period == 'start':
                # plot start in black
                SWCpk_spk_start = np.zeros((pks.shape[0],window))
                for i in range(pks.shape[0]): # go through each SWC
                    peak = pks[i]
                    this_start = spk_log_start[max(peak-window/2,0):min(peak+window/2,spk_log_start.shape[0])]
                    if peak-(window/2) < 0 or peak+(window/2) > spk_log_start.shape[0]:
                        this_start = np.pad(this_start, 
                                            ((max(window/2-peak,0),max(peak+window/2-spk_log_start.shape[0],0))), 
                                            mode='constant',
                                            constant_values=(np.nan,))
                    SWCpk_spk_start[i,:] = this_start
                mean_start = np.nanmean(SWCpk_spk_start, axis=0)
                sem_start = stats.sem(SWCpk_spk_start,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_start,color='green',label='First Second of Seizure')
                plt.fill_between(xvals, mean_start-sem_start, mean_start+sem_start, alpha=0.2, color='green')
                
                # plot rest in red
                SWCpk_spk_rest = np.zeros((pks.shape[0],window))
                for i in range(pks.shape[0]): # go through each SWC
                    peak = pks[i]
                    this_rest = spk_log_rest[max(peak-window/2,0):min(peak+window/2,spk_log_rest.shape[0])]
                    if peak-(window/2) < 0 or peak+(window/2) > spk_log_rest.shape[0]:
                        this_rest = np.pad(this_rest, 
                                           ((max(window/2-peak,0),max(peak+window/2-spk_log_rest.shape[0],0))), 
                                           mode='constant',
                                           constant_values=(np.nan,))
                    SWCpk_spk_rest[i,:] = this_rest
                mean_rest = np.nanmean(SWCpk_spk_rest, axis=0)
                sem_rest = stats.sem(SWCpk_spk_rest,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_rest,color='red', label = 'Rest of Seizure')
                plt.fill_between(xvals, mean_rest-sem_rest, mean_rest+sem_rest, alpha=0.2, color='red') 
                
                fake_pks = np.arange(spk_log.shape[0]/this_frequency)*this_frequency
                SWCpk_spk_baseline = np.zeros((fake_pks.shape[0],window))
                for i in range(fake_pks.shape[0]): # go through each SWC
                    peak = fake_pks[i]
                    this_baseline = spk_log_baseline[max(peak-window/2,0):min(peak+window/2,spk_log_baseline.shape[0])]
                    if peak-(window/2) < 0 or peak+(window/2) > spk_log_baseline.shape[0]:
                        this_baseline = np.pad(this_baseline, 
                                           ((max(window/2-peak,0),max(peak+window/2-spk_log_baseline.shape[0],0))), 
                                           mode='constant',
                                           constant_values=(np.nan,))
                    SWCpk_spk_baseline[i,:] = this_baseline
                mean_baseline = np.nanmean(SWCpk_spk_baseline, axis=0)
                sem_baseline = stats.sem(SWCpk_spk_baseline,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_baseline,color='gray', label = 'Baseline of Seizure', alpha=0.5)
                plt.fill_between(xvals, mean_baseline-sem_baseline, mean_baseline+sem_baseline, alpha=0.1, color='gray') 
               
            elif period == 'seiz':
                SWCpk_spk_seiz = np.zeros((pks.shape[0],window))
                for i in range(pks.shape[0]): # go through each SWC
                    peak = pks[i]
                    SWCpk_spk_seiz[i,:] = spk_log[peak-(window/2):peak+(window/2)]
                mean_seiz = np.nanmean(SWCpk_spk_seiz, axis=0)
                sem_seiz = stats.sem(SWCpk_spk_seiz,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_seiz,color='red',label='Seizure')
                plt.fill_between(xvals, mean_seiz-sem_seiz, mean_seiz+sem_seiz, alpha=0.2, color='red')
            elif period == 'baseline':
                # plot start in black
                SWCpk_spk_pre = np.zeros((pks_pre.shape[0],window))
                for i, peak in enumerate(pks_pre): # go through each SWC
                    this_pre = spk_log_pre[max(peak-window/2,0):min(peak+window/2,spk_log_pre.shape[0])]
                    if peak-(window/2) < 0 or peak+(window/2) > spk_log_pre.shape[0]:
                        this_pre = np.pad(this_pre, 
                                            ((max(window/2-peak,0),max(peak+window/2-spk_log_pre.shape[0],0))), 
                                            mode='constant',
                                            constant_values=(np.nan,))
                    SWCpk_spk_pre[i,:] = this_pre
                mean_pre = np.nanmean(SWCpk_spk_pre, axis=0)
                sem_pre = stats.sem(SWCpk_spk_pre,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_pre,color='black',label='1s Preictal')
                plt.fill_between(xvals, mean_pre-sem_pre, mean_pre+sem_pre, alpha=0.2, color='black')
                
                # plot rest in red
                SWCpk_spk_baseline = np.zeros((pks_baseline.shape[0],window))
                for i, peak in enumerate(pks_baseline): # go through each SWC
                    this_baseline = spk_log_baseline[max(peak-window/2,0):min(peak+window/2,spk_log_baseline.shape[0])]
                    if peak-(window/2) < 0 or peak+(window/2) > spk_log_rest.shape[0]:
                        this_baseline = np.pad(this_baseline, 
                                           ((max(window/2-peak,0),max(peak+window/2-spk_log_baseline.shape[0],0))), 
                                           mode='constant',
                                           constant_values=(np.nan,))
                    SWCpk_spk_baseline[i,:] = this_baseline
                mean_baseline = np.nanmean(SWCpk_spk_baseline, axis=0)
                sem_baseline = stats.sem(SWCpk_spk_baseline,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_baseline,color='red', label = 'Baseline')
                plt.fill_between(xvals, mean_baseline-sem_baseline, mean_baseline+sem_baseline, alpha=0.2, color='red') 
                
                
            plt.xticks(np.arange(-200,201, step=50))           
            plt.xlabel('Time from SWC Peak (ms)')
            plt.ylabel('Mean Spikes (per second)')
            plt.legend(loc='upper right')
            plt.suptitle(name + '(nseiz = ' + str(nseiz) + ')')
            
            save_path = '/mnt/Data4/MakeFigures/SWC_Triggered_Average/' + period
            char = 'None'
            
            if name in cortical_cells:
                this_class = cortical_classes[cortical_cells.index(name)]
                save_path += '/Cortical/' + this_class
                if period == 'start':
                    pSWCth = pSWCth.append({'Cell':name,'Type':'Cortical','Cluster':this_class,'nSeiz':nseiz,'Location':label,'Class':char,'Animal':rat,'Start':mean_start,'Rest':mean_rest,'Baseline':mean_baseline},ignore_index=True)
                elif period == 'seiz':
                    pSWCth = pSWCth.append({'Cell':name,'Type':'Cortical','Cluster':this_class,'nSeiz':nseiz,'Location':label,'Class':char,'Animal':rat,'AllSeizures':mean_seiz},ignore_index=True)
                elif period == 'baseline':
                    pSWCth = pSWCth.append({'Cell':name,'Type':'Cortical','Cluster':this_class,'nSeiz':nseiz,'Location':label,'Class':char,'Animal':rat,'Pre':mean_pre,'Baseline':mean_baseline},ignore_index=True)
            elif name in thalamic_cells:
                this_class = thalamic_classes[thalamic_cells.index(name)]
                save_path += '/Thalamic/' + this_class
                if period == 'start':
                    pSWCth = pSWCth.append({'Cell':name,'Type':'Thalamic','Cluster':this_class,'nSeiz':nseiz,'Location':label,'Class':char,'Animal':rat,'Start':mean_start,'Rest':mean_rest,'Baseline':mean_baseline},ignore_index=True)
                elif period == 'seiz':
                    pSWCth = pSWCth.append({'Cell':name,'Type':'Thalamic','Cluster':this_class,'nSeiz':nseiz,'Location':label,'Class':char,'Animal':rat,'AllSeizures':mean_seiz},ignore_index=True)
                elif period == 'baseline':
                    pSWCth = pSWCth.append({'Cell':name,'Type':'Thalamic','Cluster':this_class,'nSeiz':nseiz,'Location':label,'Class':char,'Animal':rat,'Pre':mean_pre,'Baseline':mean_baseline},ignore_index=True)
                
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                
            save_path += '/' + name + '.png'
            plt.savefig(save_path)
            plt.close('all')
            
        counter += 1
    
    classes = cortical_assignments['Classes'].iloc[:].to_list()
    classes = list(set(classes))
    
    for cell_type in ['Cortical', 'Thalamic']:
        for cell_class in classes:
            temp = pSWCth.loc[pSWCth['Type']==cell_type,:]
            temp = temp.loc[temp['Cluster']==cell_class,:]
            temp.to_pickle('/mnt/Data4/MakeFigures/SWC_Triggered_Average/' + period + '/' + cell_type + '/' + cell_class + '/Data.pkl')
    
    
#%%
def plot_SWC_avg_by_small_and_large(small, large):
    '''
    Plots the firing distribution of cells around SWC peaks of the first second 
    of seizure and the rest of seizure for small and large seizures for each 
    cell.
    
    Args:
        small: a df containing the small seizures
        large: a df containing the large seizures
    '''
    window = 400 # 200 ms either side of SWC peak
    database = bd.load_in_dataframe() # get the reference dataframe
    database = bd.clean_liveDB(database) # remove sessions that don't have both cells and seizures
    sess_ids = database['recording_id'].unique() # find all unique session/recording IDs
    
    with open('/mnt/Data4/CtxCellClasses.pkl','rb') as f1:
        cortical_assignments = pk.load(f1)
            
    with open('/mnt/Data4/ThalCellClasses.pkl','rb') as f2:
        thalamic_assignments = pk.load(f2)
            
    cortical_cells = cortical_assignments['Cells'].to_list()
    thalamic_cells = thalamic_assignments['Cells'].to_list()
    
    cortical_classes = cortical_assignments['Classes'].to_list()
    thalamic_classes = thalamic_assignments['Classes'].to_list()
    
    frequencies_cortical = {'Onset Peak': 141, 
                            'Sustained Increase': 140,
                            'Sustained Decrease': 137,
                            'No Change': 140, # estimate
                            'Miscellaneous': 140} # estimate
    
    frequencies_thalamic = {'Onset Peak': 144, 
                            'Sustained Increase': 152,
                            'Sustained Decrease': 150,
                            'No Change': 150,
                            'Miscellaneous': 150} # estimate
            
    for severity in ['small','large']:
        pSWCth = pd.DataFrame([], columns = ['Cell','Type','Cluster','nSeiz','Start','Rest'])
        counter = 0
        for sess in sess_ids:# iterate thru recording sessions
            print(str(counter) + '/' + str(len(sess_ids)))
            celsess_db = td.extract_dataframe(database,recording_id = sess, Type = 'Cell') # take out database rows for this session
            
            dur = int(celsess_db.iloc[0]['Duration']) # any of these will have the appropriate session duration
        
            seiz = ld.sess_seiz_load(sess) # load seizure on & off times as np array
            seiz_all = seiz
            
            # extract small or large seizures
            if severity == 'small':
                seiz_inds = small[small['Session']==sess]['Seizure'].to_numpy().astype(int)
                seiz = seiz[seiz_inds]
            elif severity == 'large':
                seiz_inds = large[large['Session']==sess]['Seizure'].to_numpy().astype(int)
                seiz = seiz[seiz_inds]
                
            if seiz.shape[0] < 2:
                continue
            
            sz_log = np.zeros(dur) # create a logical to align spikes with states
            sz_log_start = np.zeros(dur)
            sz_log_rest = np.zeros(dur)
            sz_log_bl = np.ones(dur)
            for idx, seizure in enumerate(seiz): # for each seizure (row of seiz)
                sz_log[int(seiz[idx,0]):int(seiz[idx,1])] = 1 # set elements from start to end = 1
                if (seiz[idx,1] - seiz[idx,0] <= 1000):
                    sz_log_start[int(seiz[idx,0]):int(seiz[idx,1])] = 1
                else:
                    sz_log_start[int(seiz[idx,0]):int(seiz[idx,0])+1000] = 1
                    sz_log_rest[int(seiz[idx,0])+1000:int(seiz[idx,1])] = 1
            for idx, seizure in enumerate(seiz_all):
                sz_log_bl[max(0,int(seiz_all[idx,0]-2000)):min(dur,int(seiz_all[idx,1]+2000))] = 0
        
            
            pks = ld.sess_SWCpk_load(sess) # load SWC peak times as np array
            pks = pks.astype(int)
            
# =============================================================================
#             # take the peaks within the appropriate time period
#             seiz_starts = seiz[:,0]
#             seiz_ends = seiz[:,1]
#             to_remove_start = []
#             to_remove_rest = []
#             to_remove_bl = []
#             for pk_ind in range(pks.shape[0]):
#                 this_pk = pks[pk_ind]
#                 try:
#                     start_ind = np.where(seiz_starts<=this_pk)[0][-1]
#                     start_time = seiz_starts[start_ind]
#                     if this_pk >= seiz_ends[start_ind]:
#                         continue
#                     if this_pk > start_time + 1000:
#                         to_remove_start.append(pk_ind)
#                     else:
#                         to_remove_rest.append(pk_ind)
#                 except IndexError:
#                     continue
#             to_remove_start = np.array(to_remove_start)
#             pks_start = np.delete(pks, to_remove_start)
#             pks_rest = np.delete(pks, to_remove_rest)
# =============================================================================
            pks_bl = pks
            pks_start = pks
            pks_rest = pks

            spks = ld.sess_cell_load(sess) # load spike times as list of one np array per cell in session
                
            for idx, cell in enumerate(spks): # go through each cell
                
                name = celsess_db.iloc[idx]['Name'] # get the cell's unique name
                cell = np.array(cell) # convert to numpy array for ease of use
                cell = cell[cell<=dur] # remove any spikes that occur after the end of other data
                
                _,_,cell_times= sta.load_cell(name) 
                seiz_valid = seiz[np.where(seiz[:,0]>=cell_times['start'])]
                seiz_valid = seiz_valid[np.where(seiz_valid[:,1]<=cell_times['end'])]
                nseiz = seiz_valid.shape[0]
                
                spk_log = np.zeros(dur)  # create a logical for spike presence
                spk_log[cell] = 1 # put a one for every spike
                spk_log[:cell[0]] = np.nan # set to nan before cell's first spike
                spk_log[cell[-1]:] = np.nan # and after cell disappears 
                spk_log_start = np.copy(spk_log)
                spk_log_rest = np.copy(spk_log)
                spk_log_bl = np.copy(spk_log)
                spk_log[~sz_log.astype(bool)] = np.nan  # we're only interested here in spikes during seizure
                spk_log_start[~sz_log_start.astype(bool)] = np.nan
                spk_log_rest[~sz_log_rest.astype(bool)] = np.nan
                spk_log_bl[~sz_log_bl.astype(bool)] = np.nan
    
                xvals = np.arange(-200,200, step =1)
                
                if name in cortical_cells:
                    this_class = cortical_classes[cortical_cells.index(name)]
                    this_frequency = frequencies_cortical[this_class]
                elif name in thalamic_cells:
                    this_class = thalamic_classes[thalamic_cells.index(name)]
                    this_frequency = frequencies_thalamic[this_class]
                else:
                    print(name + ' is not in classified.')
                
                # plot start in green
                SWCpk_spk_start = np.zeros((pks_start.shape[0],window))
                for i in range(pks_start.shape[0]): # go through each SWC
                    peak = pks_start[i]
                    this_start = spk_log_start[max(peak-window/2,0):min(peak+window/2,spk_log_start.shape[0])]
                    if peak-(window/2) < 0 or peak+(window/2) > spk_log_start.shape[0]:
                        this_start = np.pad(this_start, 
                                            ((max(window/2-peak,0),max(peak+window/2-spk_log_start.shape[0],0))), 
                                            mode='constant',
                                            constant_values=(np.nan,))
                    SWCpk_spk_start[i,:] = this_start
                mean_start = np.nanmean(SWCpk_spk_start, axis=0)
                sem_start = stats.sem(SWCpk_spk_start,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_start,color='green',label='First Second of Seizure')
                plt.fill_between(xvals, mean_start-sem_start, mean_start+sem_start, alpha=0.2, color='green')
                
                # plot rest in red
                SWCpk_spk_rest = np.zeros((pks_rest.shape[0],window))
                for i in range(pks_rest.shape[0]): # go through each SWC
                    peak = pks_rest[i]
                    this_rest = spk_log_rest[max(peak-window/2,0):min(peak+window/2,spk_log_start.shape[0])]
                    if peak-(window/2) < 0 or peak+(window/2) > spk_log_rest.shape[0]:
                        this_rest = np.pad(this_rest, 
                                           ((max(window/2-peak,0),max(peak+window/2-spk_log_start.shape[0],0))), 
                                           mode='constant',
                                           constant_values=(np.nan,))
                    SWCpk_spk_rest[i,:] = this_rest
                mean_rest = np.nanmean(SWCpk_spk_rest, axis=0)
                sem_rest = stats.sem(SWCpk_spk_rest,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_rest,color='red', label = 'Remainder of Seizure')
                plt.fill_between(xvals, mean_rest-sem_rest, mean_rest+sem_rest, alpha=0.2, color='red') 
                
                # plot bl in black
                fake_pks = np.arange(spk_log.shape[0]/this_frequency)*this_frequency
                pks_bl = fake_pks
                SWCpk_spk_bl = np.zeros((pks_bl.shape[0],window))
                for i in range(pks_bl.shape[0]): # go through each SWC
                    peak = pks_bl[i]
                    this_bl = spk_log_bl[max(peak-window/2,0):min(peak+window/2,dur)]
                    if peak-(window/2) < 0 or peak+(window/2) > spk_log_bl.shape[0]:
                        this_bl = np.pad(this_bl, 
                                           ((max(window/2-peak,0),max(peak+window/2-dur,0))), 
                                           mode='constant',
                                           constant_values=(np.nan,))
                    SWCpk_spk_bl[i,:] = this_bl
                mean_bl = np.nanmean(SWCpk_spk_bl, axis=0)
                sem_bl = stats.sem(SWCpk_spk_bl,axis=0,nan_policy='omit')
                plt.plot(xvals, mean_bl,color='gray', label = 'Baseline', alpha=0.5)
                plt.fill_between(xvals, mean_bl-sem_bl, mean_bl+sem_bl, alpha=0.1, color='gray') 
                    
                # add dashed line at x=0
                plt.axvline(alpha=0.3, linestyle='--', color='black')
                    
                plt.xticks(np.arange(-200,201, step=50))           
                plt.xlabel('Time from SWC Peak (ms)')
                plt.ylabel('Mean Spikes (per second)')
                plt.legend(loc='upper right')
                plt.suptitle(name + '(nseiz = ' + str(nseiz) + ')')
                
                save_path = '/mnt/Data4/MakeFigures/SWC_Triggered_Average/start'
                
                if name in cortical_cells:
                    this_class = cortical_classes[cortical_cells.index(name)]
                    save_path += '/Cortical/' + this_class + '/' + severity
                    pSWCth = pSWCth.append({'Cell':name,'Type':'Cortical','Cluster':this_class,'nSeiz':nseiz,'Start':mean_start,'Rest':mean_rest,'Baseline':mean_bl},ignore_index=True)
                elif name in thalamic_cells:
                    this_class = thalamic_classes[thalamic_cells.index(name)]
                    save_path += '/Thalamic/' + this_class + '/' + severity
                    pSWCth = pSWCth.append({'Cell':name,'Type':'Thalamic','Cluster':this_class,'nSeiz':nseiz,'Start':mean_start,'Rest':mean_rest,'Baseline':mean_bl},ignore_index=True)
               
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
                save_path += '/' + name + '.png'
                plt.savefig(save_path)
                plt.close('all')
                
            counter += 1
        
        classes = cortical_assignments['Classes'].iloc[:].to_list()
        classes = list(set(classes))
        
        for cell_type in ['Cortical', 'Thalamic']:
            for cell_class in classes:
                if cell_type == 'Thalamic' and cell_class == 'Miscellaneous':
                    continue
                temp = pSWCth.loc[pSWCth['Type']==cell_type,:]
                temp = temp.loc[temp['Cluster']==cell_class,:]
                temp.to_pickle('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/' + severity + '/Data.pkl')
    

#%%
def plot_SWC_avg_by_group(period='start',cell_type='Cortical',cell_class='Onset Peak',nseiz_min=5):
    '''
    Plots the SWC triggered mean averaged by cluster.
    
    Args:
        period = 'seiz', 'start'
        cell_type = 'Cortical', 'Thalamic'
        cell_class = 'Onset Peak', 'Sustained Decrease', 'Sustained Increase', 'No change', 'Miscellaneous'
        nseiz_min(int): the minimum number of seizures a cell must have to be included
    '''
    mpl.rcParams.update(mpl.rcParamsDefault)
    
    # generate input path
    input_path = '/mnt/Data4/MakeFigures/SWC_Triggered_Average/' + period + '/' + cell_type + '/' + cell_class + '/Data.pkl'
    
    # load data file
    with open(input_path,'rb') as f:
        data = pk.load(f)

    # apply threshold based on nseiz
    data = data.loc[data['nSeiz']>=nseiz_min,:]
    
    xvals = np.arange(-200,200, step =1)
    
    # extract data
    if period == 'start':
        all_start = np.vstack(data['Start'])
        all_rest = np.vstack(data['Rest'])
        all_bl = np.vstack(data['Baseline'])
        start_label = 'First Second of Seizure'
        rest_label = 'Rest of Seizure'
        bl_label = 'Baseline'
    elif period == 'baseline':
        all_start = np.vstack(data['Pre'])
        all_rest = np.vstack(data['Baseline'])
        start_label = '1s Preictal'
        rest_label = 'Baseline'
    
    # compute mean
    mean_start = np.nanmean(all_start,axis=0)
    mean_rest = np.nanmean(all_rest,axis=0)
    mean_bl = np.nanmean(all_bl,axis=0)
    
    # compute sem
    sem_start = stats.sem(all_start,axis=0,nan_policy='omit')    
    sem_rest = stats.sem(all_rest,axis=0,nan_policy='omit')    
    sem_bl = stats.sem(all_bl,axis=0,nan_policy='omit')
    
    # plot
    # add dashed line at x=0
    plt.axvline(alpha=0.3, linestyle='--', color='black')
    
    plt.plot(xvals, mean_bl,color='gray', label=bl_label, alpha=0.5)
    plt.fill_between(xvals, mean_bl-sem_bl, mean_bl+sem_bl, alpha=0.1, color='gray')
    
    plt.plot(xvals, mean_start,color='green',label=start_label)
    plt.fill_between(xvals, mean_start-sem_start, mean_start+sem_start, alpha=0.2, color='green')
    
    plt.plot(xvals, mean_rest,color='red', label=rest_label)
    plt.fill_between(xvals, mean_rest-sem_rest, mean_rest+sem_rest, alpha=0.2, color='red')
                
    plt.xticks(np.arange(-200,201, step=50))       
    plt.ylim(bottom=-0.005,top=0.08)    
    plt.xlabel('Time from SWC Peak (ms)')
    plt.ylabel('Mean Spikes (per second)')
    plt.legend(loc='upper right')
    plt.suptitle('SWC Triggered Average - ' + cell_type + ' ' + cell_class + ' Cluster (n=' + str(all_start.shape[0]) + ')')
    
    plt.savefig('/mnt/Data4/MakeFigures/SWC_Triggered_Average/' + period + '/' + cell_type + '/' + cell_class + '/Average.png', facecolor='white', transparent=True)
    
    plt.close()
    
    return all_start, all_rest, all_bl
    

def plot_SWC_avg_by_group_main(period='start'):
    '''
    The main function for plot_SWC_avg_by_group().
    '''
    groups = ['Sustained Increase', 'Onset Peak', 'No Change', 'Sustained Decrease', 'Miscellaneous']
    cell_types = ['Cortical', 'Thalamic']
    xvals = np.arange(-200,200, step =1)
    
    for cell_type in cell_types:
        all_start = np.empty((0,400))
        all_rest = np.empty((0,400))
        all_bl = np.empty((0,400))
    
        for group in groups:
            this_start, this_rest, this_bl = plot_SWC_avg_by_group(period=period, cell_type=cell_type, cell_class=group)
            all_start = np.vstack((all_start,this_start))
            all_rest = np.vstack((all_rest,this_rest))
            all_bl = np.vstack((all_bl,this_bl))
            
        # compute mean
        mean_start = np.nanmean(all_start,axis=0)
        mean_rest = np.nanmean(all_rest,axis=0)
        mean_bl = np.nanmean(all_bl,axis=0)
        
        # compute sem
        sem_start = stats.sem(all_start,axis=0,nan_policy='omit')    
        sem_rest = stats.sem(all_rest,axis=0,nan_policy='omit')  
        sem_bl = stats.sem(all_bl,axis=0,nan_policy='omit')  
            
        # plot
        # add dashed line at x=0
        plt.axvline(alpha=0.3, linestyle='--', color='black')
        
        plt.plot(xvals, mean_bl,color='gray',label='Baseline',alpha=0.5)
        plt.fill_between(xvals, mean_bl-sem_bl, mean_bl+sem_bl, alpha=0.1, color='gray')
    
        plt.plot(xvals, mean_start,color='green',label='First Second of Seizure')
        plt.fill_between(xvals, mean_start-sem_start, mean_start+sem_start, alpha=0.2, color='green')
        
        plt.plot(xvals, mean_rest,color='red', label = 'Rest of Seizure')
        plt.fill_between(xvals, mean_rest-sem_rest, mean_rest+sem_rest, alpha=0.2, color='red')
                    
        plt.xticks(np.arange(-200,201, step=50))       
        plt.ylim(top=0.08)    
        plt.ylim(bottom=-0.005)
        plt.xlabel('Time from SWC Peak (ms)')
        plt.ylabel('Mean Spikes (per second)')
        plt.legend(loc='upper right')
        plt.suptitle('SWC-Triggered Avg - ' + cell_type + ' (n=' + str(all_start.shape[0]) + ')')
        
        plt.savefig('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/Average.png')
        
        plt.close()
            
        
            

#%%
def plot_SWC_avg_by_group_small_and_large(cell_type='Cortical',cell_class='Onset Peak',nseiz_min=5):
    '''
    Plots the SWC triggered mean averaged by cluster for small and large.
    
    Args:
        cell_type = 'Cortical', 'Thalamic'
        cell_class = 'Onset Peak', 'Sustained Decrease', 'Sustained Increase', 'No change', 'Miscellaneous'
        nseiz_min(int): the minimum number of seizures a cell must have to be included
    '''
    with open('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/small/Data.pkl','rb') as f:
        spared_data = pk.load(f)
    with open('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/large/Data.pkl','rb') as f:
        impaired_data = pk.load(f)
        
    spared_n = spared_data['nSeiz'].to_numpy()
    impaired_n = impaired_data['nSeiz'].to_numpy()
    
    matched_cell_inds = np.where((spared_n>=nseiz_min)&(impaired_n>=nseiz_min))[0]
        
    this_ylim_top = 0
    for severity in ['small','large']:
        # generate input path
        input_path = '/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/' + severity + '/Data.pkl'
        
        if severity == 'small':
            behav_severity = 'Spared'
        elif severity == 'large':
            behav_severity = 'Impaired'
        
        # load data file
        with open(input_path,'rb') as f:
            data = pk.load(f)
    
        # apply threshold based on nseiz
        data = data.iloc[matched_cell_inds,:]
        
        xvals = np.arange(-200,200, step =1)
        
        # extract data
        all_start = np.vstack(data['Start'])
        all_rest = np.vstack(data['Rest'])
        all_bl = np.vstack(data['Baseline'])
        
        # compute mean
        mean_start = np.nanmean(all_start,axis=0)
        mean_rest = np.nanmean(all_rest,axis=0)
        mean_bl = np.nanmean(all_bl,axis=0)
        
        # compute sem
        sem_start = stats.sem(all_start,axis=0,nan_policy='omit')    
        sem_rest = stats.sem(all_rest,axis=0,nan_policy='omit')  
        sem_bl = stats.sem(all_bl,axis=0,nan_policy='omit')
        
        this_ylim_top = max(this_ylim_top,np.nanmax(mean_start+sem_start))
        this_ylim_top = max(this_ylim_top,np.nanmax(mean_rest+sem_rest))
        this_ylim_top = max(this_ylim_top,np.nanmax(mean_bl+sem_bl))
        
    for severity in ['small','large']:
        # generate input path
        input_path = '/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/' + severity + '/Data.pkl'
        
        if severity == 'small':
            behav_severity = 'Spared'
        elif severity == 'large':
            behav_severity = 'Impaired'
        
        # load data file
        with open(input_path,'rb') as f:
            data = pk.load(f)
    
        # apply threshold based on nseiz
        data = data.iloc[matched_cell_inds,:]
        
        xvals = np.arange(-200,200, step =1)
        
        # extract data
        all_start = np.vstack(data['Start'])
        all_rest = np.vstack(data['Rest'])
        all_bl = np.vstack(data['Baseline'])
        
        # compute mean
        mean_start = np.nanmean(all_start,axis=0)
        mean_rest = np.nanmean(all_rest,axis=0)
        mean_bl = np.nanmean(all_bl,axis=0)
        
        # compute sem
        sem_start = stats.sem(all_start,axis=0,nan_policy='omit')    
        sem_rest = stats.sem(all_rest,axis=0,nan_policy='omit')  
        sem_bl = stats.sem(all_bl,axis=0,nan_policy='omit')
        
        # plot
        # add dashed line at x=0
        plt.axvline(alpha=0.3, linestyle='--', color='black')
        
        plt.plot(xvals, mean_bl,color='gray', label = 'Baseline', alpha=0.5)
        plt.fill_between(xvals, mean_bl-sem_bl, mean_bl+sem_bl, alpha=0.1, color='gray')
    
        plt.plot(xvals, mean_start,color='green',label='First Second of Seizure')
        plt.fill_between(xvals, mean_start-sem_start, mean_start+sem_start, alpha=0.2, color='green')
        
        plt.plot(xvals, mean_rest,color='red', label = 'Rest of Seizure')
        plt.fill_between(xvals, mean_rest-sem_rest, mean_rest+sem_rest, alpha=0.2, color='red')        
                    
        plt.xticks(np.arange(-200,201, step=50))       
        plt.ylim(top=this_ylim_top)    
        plt.ylim(bottom=-0.005)
        plt.xlabel('Time from SWC Peak (ms)')
        plt.ylabel('Mean Spikes (per second)')
        plt.legend(loc='upper right')
        plt.suptitle('SWC-Triggered Avg - ' + cell_type + ' ' + cell_class + ' ' + behav_severity + ' (n=' + str(all_start.shape[0]) + ')')
        
        plt.savefig('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/' + severity + '/Average.png')
        
        plt.close()
        
def plot_SWC_avg_by_group_small_and_large_all(cell_type='Cortical',nseiz_min=5):
    '''
    Plots the SWC triggered mean averaged for all cells for small and large.
    
    Args:
        cell_type: 'Cortical', 'Thalamic'
        nseiz_min(int): the minimum number of seizures a cell must have to be included
    '''
    
    for severity in ['small','large']:
        all_start = np.empty((0,400))
        all_rest = np.empty((0,400))
        all_bl = np.empty((0,400))
        for cell_class in ['Sustained Increase', 'Onset Peak', 'No Change', 'Sustained Decrease']:
            with open('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/small/Data.pkl','rb') as f:
                spared_data = pk.load(f)
            with open('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/large/Data.pkl','rb') as f:
                impaired_data = pk.load(f)
                
            spared_n = spared_data['nSeiz'].to_numpy()
            impaired_n = impaired_data['nSeiz'].to_numpy()
            
            matched_cell_inds = np.where((spared_n>=nseiz_min)&(impaired_n>=nseiz_min))[0]
    
            # generate input path
            input_path = '/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/' + severity + '/Data.pkl'
            
            if severity == 'small':
                behav_severity = 'Spared'
            elif severity == 'large':
                behav_severity = 'Impaired'
            
            # load data file
            with open(input_path,'rb') as f:
                data = pk.load(f)
        
            # apply threshold based on nseiz
            data = data.iloc[matched_cell_inds,:]
            
            xvals = np.arange(-200,200, step =1)
            
            # extract data
            all_start = np.vstack((all_start,np.vstack(data['Start'])))
            all_rest = np.vstack((all_rest,np.vstack(data['Rest'])))
            all_bl = np.vstack((all_bl,np.vstack(data['Baseline'])))
            
        # compute mean
        mean_start = np.nanmean(all_start,axis=0)
        mean_rest = np.nanmean(all_rest,axis=0)
        mean_bl = np.nanmean(all_bl,axis=0)
        
        # compute sem
        sem_start = stats.sem(all_start,axis=0,nan_policy='omit')    
        sem_rest = stats.sem(all_rest,axis=0,nan_policy='omit')  
        sem_bl = stats.sem(all_bl,axis=0,nan_policy='omit')  
            
        # plot
        # add dashed line at x=0
        plt.axvline(alpha=0.3, linestyle='--', color='black')
        
        plt.plot(xvals, mean_bl,color='gray',label='Baseline',alpha=0.5)
        plt.fill_between(xvals, mean_bl-sem_bl, mean_bl+sem_bl, alpha=0.1, color='gray')
    
        plt.plot(xvals, mean_start,color='green',label='First Second of Seizure')
        plt.fill_between(xvals, mean_start-sem_start, mean_start+sem_start, alpha=0.2, color='green')
        
        plt.plot(xvals, mean_rest,color='red', label = 'Rest of Seizure')
        plt.fill_between(xvals, mean_rest-sem_rest, mean_rest+sem_rest, alpha=0.2, color='red')
                    
        plt.xticks(np.arange(-200,201, step=50))       
        plt.ylim(top=0.03)    
        plt.ylim(bottom=-0.001)
        plt.xlabel('Time from SWC Peak (ms)')
        plt.ylabel('Mean Spikes (per second)')
        plt.legend(loc='upper right')
        plt.suptitle('SWC-Triggered Avg - ' + cell_type + ' ' + behav_severity + ' (n=' + str(all_start.shape[0]) + ')')
        
        plt.savefig('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/Average_' + severity + '.png')
        
        plt.close()
    

def plot_SWC_avg_by_group_small_and_large_main():
    '''
    The main function for plot_SWC_avg_by_group_small_and_large().
    '''
    groups = ['Sustained Increase', 'Onset Peak', 'No Change', 'Sustained Decrease', 'Miscellaneous']
    cell_types = ['Cortical', 'Thalamic']
    
    for cell_type in cell_types:
        for group in groups:
            if cell_type == 'Thalamic' and group == 'Miscellaneous':
                continue
            plot_SWC_avg_by_group_small_and_large(cell_type=cell_type, cell_class=group)
        plot_SWC_avg_by_group_small_and_large_all(cell_type)
        

#%% stats
def integral_stats(cycle=[-71,71],peak=[-30,10],nseiz_min=5):
    '''
    Performs paired t tests on spared vs impaired integrals on cells and individual clusters based on severity classification. 
    
    Args:
        cycle: the start and end times of a cycle of firing
        peak: the time points between which to take the peak
        nseiz_min: min number of spared/impaired seizures a cell must have to be included
    '''
    clusters = []
    cell_names = []
    cell_types = []
    start_small = []
    start_large = []
    rest_small = []
    rest_large = []
    bl_small = []
    bl_large = []
    for cell_type in ['Cortical','Thalamic']:
        for cell_class in ['Sustained Increase', 'Onset Peak', 'No Change', 'Sustained Decrease']:
            
            # load data file
            with open('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/small/Data.pkl','rb') as f:
                data_small = pk.load(f)
            with open('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/' + cell_type + '/' + cell_class + '/large/Data.pkl','rb') as f:
                data_large = pk.load(f)
        
            # apply threshold based on nseiz
            nseiz_small = data_small['nSeiz']
            nseiz_large = data_large['nSeiz']
            data_small = data_small.iloc[np.where((nseiz_small>=nseiz_min) & (nseiz_large>=nseiz_min))[0],:]
            data_large = data_large.iloc[np.where((nseiz_small>=nseiz_min) & (nseiz_large>=nseiz_min))[0],:]
            
            clusters += data_small['Cluster'].to_list()
            cell_types += data_small['Type'].to_list()
            cell_names += data_small['Cell'].to_list()
            
            # extract data
            all_start_small = np.vstack(data_small['Start'])
            all_rest_small = np.vstack(data_small['Rest'])
            all_bl_small = np.vstack(data_small['Baseline'])
            
            start_small += list(np.nansum(all_start_small[:,peak[0]+200:peak[1]+200],axis=1) / np.nansum(all_start_small[:,cycle[0]+200:cycle[1]+200],axis=1))
            rest_small += list(np.nansum(all_rest_small[:,peak[0]+200:peak[1]+200],axis=1) / np.nansum(all_rest_small[:,cycle[0]+200:cycle[1]+200],axis=1))
            bl_small += list(np.nansum(all_bl_small[:,peak[0]+200:peak[1]+200],axis=1) / np.nansum(all_bl_small[:,cycle[0]+200:cycle[1]+200],axis=1))
            
            all_start_large = np.vstack(data_large['Start'])
            all_rest_large = np.vstack(data_large['Rest'])
            all_bl_large = np.vstack(data_large['Baseline'])
            
            start_large += list(np.nansum(all_start_large[:,peak[0]+200:peak[1]+200],axis=1) / np.nansum(all_start_large[:,cycle[0]+200:cycle[1]+200],axis=1))
            rest_large += list(np.nansum(all_rest_large[:,peak[0]+200:peak[1]+200],axis=1) / np.nansum(all_rest_large[:,cycle[0]+200:cycle[1]+200],axis=1))
            bl_large += list(np.nansum(all_bl_large[:,peak[0]+200:peak[1]+200],axis=1) / np.nansum(all_bl_large[:,cycle[0]+200:cycle[1]+200],axis=1))
            
    data_df = pd.DataFrame(data={'Cell':cell_names,'Type':cell_types,'Cluster':clusters,
                                 'First Sec Firing (Spared)':start_small, 'Remainder Firing (Spared)':rest_small, 'Baseline Firing (Spared)':bl_small,
                                 'First Sec Firing (Impaired)':start_large, 'Remainder Firing (Impaired)':rest_large, 'Baseline Firing (Impaired)':bl_large})
        
    data_df.to_csv('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/Stats_Data.csv')
    
    # do stats
    cell_types = []
    clusters = []
    periods = []
    mean_spared = []
    mean_impaired = []
    diff = []
    pvals = []
    for cell_type in ['Cortical','Thalamic']:
        all_spared_start = []
        all_impaired_start = []
        all_spared_rest = []
        all_impaired_rest = []
        for cell_class in ['Sustained Increase', 'Onset Peak', 'No Change', 'Sustained Decrease']:
            this_data = data_df[data_df['Type']==cell_type]
            this_data = this_data[this_data['Cluster']==cell_class]
            
            cell_types += [cell_type] * 2
            clusters += [cell_class] * 2
            periods += ['First Sec of Seiz','Rest of Seiz']
            
            # first sec
            this_spared = this_data['First Sec Firing (Spared)']
            this_impaired = this_data['First Sec Firing (Impaired)']
            all_spared_start += list(this_spared)
            all_impaired_start += list(this_impaired)
            mean_spared.append(np.nanmean(this_spared))
            mean_impaired.append(np.nanmean(this_impaired))
            diff.append(np.nanmean(this_impaired)-np.nanmean(this_spared))
            
            t, p = stats.ttest_rel(this_spared, this_impaired, nan_policy='omit')
            pvals.append(p)
            
            # remainder
            this_spared = this_data['Remainder Firing (Spared)']
            this_impaired = this_data['Remainder Firing (Impaired)']
            all_spared_rest += list(this_spared)
            all_impaired_rest += list(this_impaired)
            mean_spared.append(np.nanmean(this_spared))
            mean_impaired.append(np.nanmean(this_impaired))
            diff.append(np.nanmean(this_impaired)-np.nanmean(this_spared))
            
            t, p = stats.ttest_rel(this_spared, this_impaired, nan_policy='omit')
            pvals.append(p)
        
        cell_types.append(cell_type)
        clusters.append('All Clusters')
        periods.append('First Sec of Seiz')
        mean_spared.append(np.nanmean(all_spared_start))
        mean_impaired.append(np.nanmean(all_impaired_start))
        diff.append(np.nanmean(all_impaired_start)-np.nanmean(all_spared_start))
        t, p = stats.ttest_rel(all_spared_start, all_impaired_start, nan_policy='omit')
        pvals.append(p)
        
        cell_types.append(cell_type)
        clusters.append('All Clusters')
        periods.append('Rest of Seiz')
        mean_spared.append(np.nanmean(all_spared_rest))
        mean_impaired.append(np.nanmean(all_impaired_rest))
        diff.append(np.nanmean(all_impaired_rest)-np.nanmean(all_spared_rest))
        t, p = stats.ttest_rel(all_spared_rest, all_impaired_rest, nan_policy='omit')
        pvals.append(p)
            
    data_stats = pd.DataFrame(data={'Type':cell_types,'Cluster':clusters,'Period':periods,'Mean (Spared)':mean_spared,'Mean (Impaired)':mean_impaired,'Impaired - Spared':diff,'p':pvals})
    data_stats.to_csv('/mnt/Data4/MakeFigures/SWC_Triggered_Average/start/Stats_Results.csv')

#%% 
# =============================================================================
# if __name__ == "__main__":
# =============================================================================
# =============================================================================
#     plot_SWC_avg(period='start')
#     plot_SWC_avg_by_group_main()
# =============================================================================
    # small, large = gbp.load_preds(same_cell=True,cutoff=5)
    # plot_SWC_avg_by_small_and_large(small, large)
# =============================================================================
#     plot_SWC_avg_by_group_small_and_large_main()
# =============================================================================
# =============================================================================
#     plot_SWC_avg(period='baseline')
#     plot_SWC_avg_by_group_main(period='baseline')
# =============================================================================
