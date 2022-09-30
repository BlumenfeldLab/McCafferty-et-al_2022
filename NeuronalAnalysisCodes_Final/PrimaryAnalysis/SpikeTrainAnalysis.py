# -*- coding: utf-8 -*-
"""
Created on Wed May 22 10:15:58 2019

@author: Renee Tung & Kohl Swift, 
check = extract_utils.batch_cells('/mnt/Data4/AnnotateSWD/','/mnt/Data4/GAERS_Data/')
"""
from os.path import expanduser
home = expanduser('~')

import sys
sys.path.insert(0, home+'/mnt/Data4/GAERS_Codes/DataExtraction')
import extract_utils as eu
import class_definitions
import pickle as pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.cluster.vq import vq, kmeans, whiten
#from scipy.signal import find_peaks
import build_database as bd
import warnings
import seaborn as sns
from scipy import stats
#from sklearn.mixture import GaussianMixture
#from sklearn.linear_model import LinearRegression
import math 
import os

sys.path.insert(0,home+'/mnt/Data4/GAERS_Codes/AnalysisCodes')
import GenericFuncsRenee as gf
import LoadData as ld


"""
#misc
database = bd.load_in_dataframe()
cell_df = sta.extract_dataframe(database,Type='Cell')
cells_fr = sta.cell_fr(cell_df)
seiz_firing_cells = cells_fr[~np.isnan(cells_fr['seiz_fr'])]

cells_fr = sta.cell_fr(cell_df)
nonseiz_fr = cells_fr['nonseiz_fr']
seiz_fr = cells_fr['seiz_fr']
x = seiz_fr-nonseiz_fr
fr_increase = cells_fr[x>0]; fr_increase_list = []
for i,cell in fr_increase.iterrows():
    fr_increase_list.append(cell['Name'])
fr_increase = sta.pull_select_dataframe(database,fr_increase_list)
fr_decrease = cells_fr[x<0]; fr_decrease_list = []
for i,cell in fr_decrease.iterrows():
    fr_decrease_list.append(cell['Name'])
fr_decrease = sta.pull_select_dataframe(database,fr_decrease_list)

def set_names(num):
    cell_oi = cell_df.iloc[num-1:num]
    cell_name = cell_oi['Name'].values[0]
    return cell_oi,cell_name
def test(num):
    plt.close('all')
    cell_oi,cell_name = set_names(num)
    sta.analyze_cell(cell_name)

for i,cell in cells_fr.iterrows():
    fr_course = np.array([cell['nonseiz_fr'],cell['preseiz_fr'],cell['seiz_fr'],cell['postseiz_fr']])
    plt.plot(range(4),fr_course)
    
cells_fr = sta.fix_df_index(cells_fr)
bl_presz_dif = np.empty([len(cells_fr)])
for i,cell in cells_fr.iterrows():
    bl_presz_dif[i] = cell['preseiz_fr'] - cell['nonseiz_fr']
bl_presz_dif.sort()
plt.figure()
plt.scatter(range(len(cells_fr)),bl_presz_dif)
"""
#%%
def panda_input(func):
    """Decorator that allows decorated functions to use both string, list, and panda input"""
    def wrapper(*args,**kwargs):
        if isinstance(args[0],pd.core.frame.DataFrame):
            cells=args[0]
            cells=cells['Name']#allow for multiple types?
            cells=cells.values[:]
            cells.astype(str)
            print("Panda wrapped")
            if len(args)==1:
                value = func(cells,**kwargs)
            else:
                value = func(cells,*args[1:],**kwargs)
        elif isinstance(args[0],str):
            value = func([args[0]],*args[1:],**kwargs)
        else:
            value = func(*args,**kwargs)
        return value
    return wrapper
#%%
@panda_input
def analyze_cell(cell_names,save_dir = ''):
    database = bd.load_in_dataframe()
    for cell_name in cell_names:
        print('Running Full Cell Analysis on: {}'.format(cell_name))
        cell_df = extract_dataframe(database,Name=cell_name)
        avg_seiz_tc(cell_df,onset_period=5000,offset_period=5000,period='onset',mode='gaussian')
        if save_dir != '': plt.savefig(save_dir + '/AverageSeizTCOnset.png')
        avg_seiz_tc(cell_df,onset_period=5000,offset_period=5000,period='offset',mode='gaussian')
        if save_dir != '': plt.savefig(save_dir + '/AverageSeizTCOffset.png')
#        firing_rate_panda = cell_fr(cell_name)
        cell_isi(cell_name,plot='zoom')
        if save_dir != '': plt.savefig(save_dir + '/ISIHist.png')
        plot_seizs_in_cell(cell_df,onset_period = 1000,offset_period = 1000,mode = 'onset',plot = True,exclusion_period = 50,min_dur=True)
        if save_dir != '': plt.savefig(save_dir + '/SpikeHist.png')
        is_rhythmic(cell_name,mode='seiz', plot=True)
        if save_dir != '': plt.savefig(save_dir + '/SeizIPIHist.png')
        is_rhythmic(cell_name,mode='nonseiz', plot=True)
        if save_dir != '': plt.savefig(save_dir + '/NonSeizIPIHist.png')
        is_rhythmic(cell_name,mode='full', plot=True)
        if save_dir != '': plt.savefig(save_dir + '/FullIPIHist.png')
#%%
def cluster_cell_fr(cell_names):
    """This function takes in a list of cells as a string list and clusters them 
    based on firing rate using kmeans clustering. It also plots centroids and 
    data as a 3D scatterplot."""
    fr = cell_fr(cell_names,False) #calculate firing rates for all cells in list
    fr = fix_df_index(fr)
    data_mat = np.zeros([len(fr),3])
    for i,cell in fr.iterrows():
        base_fr = cell['nonseiz_fr']#try clustering without scaling
        data_mat[i,0] = cell['seiz_fr']/base_fr
        data_mat[i,1] = cell['preseiz_fr']/base_fr
        data_mat[i,2] = cell['postseiz_fr']/base_fr
#        if cell['seiz_fr']/base_fr <1:
#            data_mat[i,0] = 0
#        else:
#            data_mat[i,0] = 1
#        if cell['preseiz_fr']/base_fr <1:
#            data_mat[i,1] = 0
#        else:
#            data_mat[i,1] = 1
#        if cell['postseiz_fr']/base_fr <1:
#            data_mat[i,2] = 0
#        else:
#            data_mat[i,2] = 1
    isnan = (fr['num_seiz'].values)!=0 #=0???
    data_mat = data_mat[isnan]
    centroids,_ = kmeans(data_mat,3) #kmeans clustering of seizure, preseizure, and postseizure firing rate
    idx,_ = vq(data_mat,centroids) #get centroid assignment of each cell 
    from mpl_toolkits.mplot3d import Axes3D #For 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') #3D subplot
    ax.scatter(data_mat[:,0],data_mat[:,1],data_mat[:,2]) #Scatter plot of data
    ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c='r')#Scatter plot of centroids in red
    return data_mat, centroids, idx

#%%%
def recalc_severityfrdf():
    #run if need to recalculate and save fr info for each cell's seizures of a certain severity
    #note: if want this but for all seizures in each cell, run sta.cell_fr(); if want to rerun and save to pkl, use func param
    database = bd.load_in_dataframe()
    cell_df = extract_dataframe(database,Type='Cell') #all cells
    seiz_df = extract_dataframe(database,Type='Seizure') #all seizures
    cort_seizs = pull_seiz_celltype(seiz_df, celltype = 'cortical') #seizures for cortical
    thal_seizs = pull_seiz_celltype(seiz_df, celltype = 'thalamic') #seizures for thalamic
    cort_cell = extract_dataframe(cell_df, label = 'Cortical') #cortical cells
    thal_cell = extract_dataframe(cell_df, label = 'Thalamic') #thalamic cells
    cort_szimp = extract_dataframe(cort_seizs, label = 'Impaired') #impaired seizures cortical
    cort_szspa = extract_dataframe(cort_seizs, label = 'Spared') #spared seizures cortical
    thal_szimp = extract_dataframe(thal_seizs, label = 'Impaired') #impaired seizures thalamic
    thal_szspa = extract_dataframe(thal_seizs, label = 'Spared') #spared seizures thalamic
    
    cortspared_df = cell_fr(cort_cell,seiz_panda=cort_szspa, save_new = 2) #fr for spared seizure, cortical cells
    cortimpaired_df = cell_fr(cort_cell,seiz_panda=cort_szimp, save_new = 2) #fr for impaired seizures, cortical cells
    thalspared_df = cell_fr(thal_cell,seiz_panda=thal_szspa, save_new = 2) #fr for spared seizures, thalamic cells
    thalimpaired_df = cell_fr(thal_cell,seiz_panda=thal_szimp, save_new = 2) #fr for impaired seizures, thalamic cells
    
    #cleaning up nan values
    checks = np.empty(4)
    checks[0] = sum(np.isnan(np.array(cortspared_df['seiz_fr']))) #cort spa check
    checks[1] = sum(np.isnan(np.array(cortimpaired_df['seiz_fr']))) #cort imp check
    checks[2] = sum(np.isnan(np.array(thalspared_df['seiz_fr']))) #thal spa check
    checks[3] = sum(np.isnan(np.array(thalimpaired_df['seiz_fr']))) #thal imp check
    for i in range(4):
        if checks[i] != 0: #if there are nan values in the fr (due to seizures in recording but not during cell time)
            if i ==0:
                nanidx = np.argwhere(np.isnan(np.array(cortspared_df['seiz_fr'])))
                numnan = len(nanidx)
                for j in range(numnan):
                    nanidxint = int(nanidx[j])
                    cortspared_df = remove_from_dataframe(cortspared_df, Name = cortspared_df.iloc[nanidxint]['Name'])
            elif i==1:
                nanidx = np.argwhere(np.isnan(np.array(cortimpaired_df['seiz_fr'])))
                numnan = len(nanidx)
                for j in range(numnan):
                    nanidxint = int(nanidx[j])
                    cortimpaired_df = remove_from_dataframe(cortimpaired_df, Name = cortimpaired_df.iloc[nanidxint]['Name'])
            elif i==2:
                nanidx = np.argwhere(np.isnan(np.array(thalspared_df['seiz_fr'])))
                numnan = len(nanidx)
                for j in range(numnan):
                    nanidxint = int(nanidx[j])
                    thalspared_df = remove_from_dataframe(thalspared_df, Name = thalspared_df.iloc[nanidxint]['Name'])
            elif i==3:
                nanidx = np.argwhere(np.isnan(np.array(thalimpaired_df['seiz_fr'])))
                numnan = len(nanidx)
                for j in range(numnan):
                    nanidxint = int(nanidx[j])
                    thalimpaired_df = remove_from_dataframe(thalimpaired_df, Name = thalimpaired_df.iloc[nanidxint]['Name'])
    
    filename = '/mnt/Data4/GAERS_Data/cortspared.pkl'
    bd.save_dataframe(cortspared_df,save_dir = filename)
    filename = '/mnt/Data4/GAERS_Data/cortimpaired.pkl'
    bd.save_dataframe(cortimpaired_df,save_dir = filename)
    filename = '/mnt/Data4/GAERS_Data/thalspared.pkl'
    bd.save_dataframe(thalspared_df,save_dir = filename)    
    filename = '/mnt/Data4/GAERS_Data/thalimpaired.pkl'
    bd.save_dataframe(thalimpaired_df,save_dir = filename)    

#%%
def avg_seiz_tc(cell_df, seiz_type = 'none', seiz_min_dur = 0, seiz_max_dur = 'max', onset_period = 5000,offset_period = 5000, period = 'onset',mode = 'gaussian', plot=1, show_sem = 0):
    """Calculates the average seizure timecourse given a panda of cells"""
    database = bd.load_in_dataframe()
    analysis_len = onset_period+offset_period
    tc_list = []
   
    ids = cell_df['recording_id'].unique() #find unique recording id's for multiple cells
    
    for rec_id in ids:
#        break
        seizs = extract_dataframe(database,recording_id = rec_id,Type = 'Seizure')
        if len(seizs) == 0:
            continue
        all_seizs = seizs
        if seiz_min_dur != 0 or seiz_max_dur != 'max':
            seizs = sep_seizs_by_dur(seizs,seiz_min_dur, seiz_max_dur)
        if seiz_type == 'spared':
            seizs = extract_dataframe(seizs, label = 'Spared')
        elif seiz_type == 'impaired':
            seizs = extract_dataframe(seizs, label = 'Impaired')
        cells_in_rec = extract_dataframe(cell_df,recording_id = rec_id)
        
        for i,cell in cells_in_rec.iterrows(): #iterate through cells in recording
#            break
            spk_times,spk_log,cell_times= load_cell(cell['Name'])
            cell_end = len(spk_log) #end in cell time
            
            seiz_starts = all_seizs['start'].values #these will be important for postseizure times
            seiz_ends = all_seizs['end'].values #these will be important for preseizure times
            
            for i,seiz in seizs.iterrows(): #iterate through seizures in recording
                if period =='onset':
                    ref = int(seiz['start']-cell_times['start'])#seiz start in cell time
                    if ref < 0 or ref > cell_end: #if the seizure start is outside this cell
                        continue
                    if seiz['start'] == min(seiz_starts): #if this is the first seizure
                        prev_sz_end = cell_times['start']
                    else:
                        prev_sz_end = max(seiz_ends[seiz['start'] - seiz_ends > 0]) #in rec time
                elif period == 'offset':
                    ref = int(seiz['end']-cell_times['start'])#seiz end in cell time
                    if ref > cell_end or ref < 0: #if seizure end is after the cell ends or before the cell starts
                        continue
                    if seiz['end'] == max(seiz_ends): #if this is the last seizure
                        next_sz_start = cell_end
                    else:
                        next_sz_start = min(seiz_starts[seiz_starts - seiz['end'] > 0]) #in rec time
                tc_start = max(ref - onset_period, 0) #if starts before cell starts (<0), set to 0
                tc_end   = min(ref + offset_period, cell_end) #make sure ends if the cell time ends
                tc_template = np.full(analysis_len,np.nan) #empty nan array
                if not ((period != 'onset') & (period != 'offset')):
                    if period == 'onset':
                        tc_end = int(min(ref + seiz['end'] - seiz['start'],tc_end)) #in cell time: end of seizure, or end of period of interest
                        tc_start = int(max(tc_start, ref - (seiz['start'] - prev_sz_end))) #in cell time: end of previous seizure, or full period of interest
                    elif period == 'offset':
                        tc_start = int(max(ref - seiz['end'] + seiz['start'],tc_start)) #in cell time: start of seizure, or start of period of interest within seizure
                        tc_end = int(min(tc_end, ref + (next_sz_start - seiz['end']))) #in cell time: end of postictal period of interest, or start of next seizure
#                    start_idx = tc_start - ref + onset_period #if -onset_period is index 0, where does this firing tc start?
                    start_idx = onset_period - (ref - tc_start) #if -onset_period is index 0, where does this firing tc start?
                    #end_idx = tc_end - tc_start
                    tc = np.insert(tc_template, start_idx, np.array([spk_log[tc_start:tc_end]]).ravel()) #insert the spike train in the middle of the nan template
                    tc = tc[:len(tc_template)] #cut off the extra nan's at the end
                    tc_list.append(tc)
                else:
                    print('Invalid Period')
                    break
#    tc_list = pad_epochs_to_equal_length(data_list = tc_list, pad_val=np.nan, align = period)
    tc_array = np.array(tc_list)
    num_cellseiz = sum(~np.isnan(tc_array))
#    num_seiz = int(np.nanmean(sum(np.isnan(tc_array)!=len(tc_array))))
    num_seiz = len(tc_array)
    avg_tc = np.nanmean(tc_array,axis=0)
    sem_tc = stats.sem(tc_array, axis=0, nan_policy='omit')
    
    #Plotting
    if plot:
        plt.figure()
        if mode == 'gaussian':
            gaussian_tc = sliding_gaussian_window(avg_tc,100)
            gaussian_tc*= 1000
            x = np.array(range(-onset_period,-onset_period + len(gaussian_tc)))#x values for plotting
            plt.plot(x,gaussian_tc)
            plt.xlim(-onset_period,offset_period)
#            plt.xlim(-5000,5000)
            plt.ylim(1,7)
            plt.title('Average Seizure Timecourse')
            plt.ylabel('Firing Rate Spikes/Second')
        elif mode == 'normalized_gaussian':
            if period == 'onset':
                #note that this baseline period does not include the 2 seconds before seizure starts (line below)
                baseline_fr = np.matlib.repmat(np.nanmean(tc_array[:,0:onset_period-2000], axis=1).transpose(),tc_array.shape[1],1)
                plt.title('Average Seizure Timecourse, Normalized to Preseizure Firing Rate')
            if period == 'offset':
                #note that this baseline period does not include the 2 seconds after seizure ends (line below)
                baseline_fr = np.matlib.repmat(np.nanmean(tc_array[:,onset_period+2000:], axis=1).transpose(),tc_array.shape[1],1)
                plt.title('Average Seizure Timecourse, Normalized to Postseizure Firing Rate')
            normalized_tc_array = tc_array - baseline_fr.transpose()
            avg_tc = np.nanmean(normalized_tc_array, axis = 0)
            gaussian_tc = sliding_gaussian_window(avg_tc,100)
            gaussian_tc*= 1000
            x = np.array(range(-onset_period,-onset_period + len(gaussian_tc)))#x values for plotting
            plt.plot(x,gaussian_tc)
            plt.xlim(-onset_period,offset_period)
            plt.ylim(-2, 1)
            plt.ylabel('Firing Rate - Pre-Seizure Baseline Firing Rate (Spikes/Second)')
                
        elif mode == 'bar':
            binWidth = 50
            binx = range(-onset_period,offset_period,binWidth)
            num_bins = len(binx)
            biny = np.zeros(num_bins)
            for i in range(num_bins):
                print(sum(avg_tc[i*binWidth:(i+1)*binWidth]))
                biny[i] = sum(avg_tc[i*binWidth:(i+1)*binWidth])
                biny[i]/=binWidth/1000.0
            plt.bar(binx,biny,width=binWidth,align='edge')
            plt.title('Average Seizure Timecourse')
            plt.ylabel('Firing Rate Spikes/Second')
        plt.axvline(x=0,color = 'red', alpha=0.5)
        plt.xlabel('Time Relative to Seizure {}(ms)'.format(period))
        ax = plt.gca()
        plt.text(0.99,0.99,'Number of Seizures: {}\nNumber of Cells: {}\nMinDur: {} MaxDur: {}'.format(num_seiz,len(cell_df), seiz_min_dur, seiz_max_dur),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
        
        plt.figure()
        plt.title('Histogram of Seizures Included at Each Time Point')
        plt.plot(x,num_cellseiz)
    
    return tc_array, sem_tc

#%%
def severity_frtc(cell_df, szlen = 10000, interest_period = 20000, period = 'onset', smooth_size = 1000):
    #timecourse for seizures split by severity (in cell-seizures)
    if period == 'onset':
        all_tc, _= avg_seiz_tc(cell_df, period = period, onset_period = interest_period, offset_period = szlen, plot=0)
        spared_tc, _ = avg_seiz_tc(cell_df, seiz_type = 'spared', period = period, onset_period = interest_period, offset_period = szlen, plot=0)
        impaired_tc, _ = avg_seiz_tc(cell_df,seiz_type = 'impaired', period = period, onset_period = interest_period, offset_period = szlen, plot=0)
        x = np.array(range(-interest_period, -interest_period + all_tc.shape[1]))
    elif period == 'offset':
        all_tc, _= avg_seiz_tc(cell_df, period = period, onset_period = szlen, offset_period = interest_period, plot=0)
        spared_tc, _ = avg_seiz_tc(cell_df, seiz_type = 'spared', period = period, onset_period = szlen, offset_period = interest_period, plot=0)
        impaired_tc, _ = avg_seiz_tc(cell_df, seiz_type = 'impaired', period = period, onset_period = szlen, offset_period = interest_period, plot=0)
        x = np.array(range(-szlen, -szlen + all_tc.shape[1]))
    avg_all = np.nanmean(all_tc, axis=0)
    avg_spared = np.nanmean(spared_tc, axis=0)
    avg_impaired = np.nanmean(impaired_tc, axis=0)
    num_all = sum(~np.isnan(all_tc))
    num_spared = sum(~np.isnan(spared_tc))
    num_impaired = sum(~np.isnan(impaired_tc))
    gaussian_all = sliding_gaussian_window(avg_all, smooth_size) * 1000
    gaussian_spared = sliding_gaussian_window(avg_spared, smooth_size) * 1000
    gaussian_impaired = sliding_gaussian_window(avg_impaired, smooth_size) * 1000
    
    plt.figure()
    plt.plot(x, gaussian_all, c='k')
    plt.plot(x, gaussian_spared, c='b')
    plt.plot(x, gaussian_impaired, c='r')
    plt.legend(['all n = {}'.format(len(all_tc)),'spared n = {}'.format(len(spared_tc)),\
    'impaired n = {}'.format(len(impaired_tc))])
    plt.title('Average Seizure Timecourse for Seizures of Varying Severity, Cells = {}'.format(len(cell_df)))
    plt.axvline(x=0,color = 'red', alpha=0.5)
    plt.xlabel('Time Relative to Seizure {}(ms)'.format(period))
    plt.ylabel('Firing Rate Spikes/Second')
#    plt.ylim(1,9)
    plt.ylim(1,20)
    
    plt.figure()
    plt.title('Histogram of All Seizures Included at Each Time Point')
    plt.plot(x, num_all, c= 'k')
    plt.plot(x, num_spared, c='b')
    plt.plot(x, num_impaired, c='g')
    plt.axvline(x=0,color = 'red', alpha=0.5)
    plt.legend(['all','spared','impaired'])
    plt.xlabel('Time Relative to Seizure {} (ms)'.format(period))
    plt.ylabel('Count')


#%%
def plot_sep_timecourses(cell_df, szlen = 10000, interest_period = 20000, short_dur = 5000, med_dur_start = 5000, med_dur_end = 10000, long_dur = 10000, period = 'onset', plot_type = 'separate', smooth_size = 1000):
    #This plots the seizure firing time course split by seizure duration.. 
    
    if period == 'onset':
        all_tc, _= avg_seiz_tc(cell_df, period = period, onset_period = interest_period, offset_period = szlen, plot=0)
    elif period == 'offset':
        all_tc, _= avg_seiz_tc(cell_df, period = period, onset_period = szlen, offset_period = interest_period, plot=0)
    all_num_cellseiz = sum(~np.isnan(all_tc))
    avg_tc_all = np.nanmean(all_tc,axis=0)
    sem_tc_all = stats.sem(all_tc, axis=0, nan_policy='omit')
    gaussian_all = sliding_gaussian_window(avg_tc_all, smooth_size) * 1000
    if period == 'onset':
        x_all = np.array(range(-interest_period,-interest_period + len(gaussian_all)))
    elif period == 'offset':
        x_all = np.array(range(-szlen,-szlen + len(gaussian_all)))
    
#    timecourse_len = szlen + interest_period
#    template_timecourse = np.full(timecourse_len,np.nan) #empty nan array
    
    if period == 'onset':
        short_tc, _ = avg_seiz_tc(cell_df, period = period, seiz_min_dur = 0, seiz_max_dur = short_dur, onset_period = interest_period, offset_period = short_dur, plot=0)
        avg_tc_short = np.nanmean(short_tc, axis=0)
#        avg_tc_short = np.insert(template_timecourse, szlen-short_dur, avg_tc_short)
#        avg_tc_short = avg_tc_short[:timecourse_len]
    elif period == 'offset':
        short_tc, _ = avg_seiz_tc(cell_df, period = period, seiz_min_dur = 0, seiz_max_dur = short_dur, onset_period = short_dur, offset_period = interest_period, plot=0)
        avg_tc_short = np.nanmean(short_tc, axis=0)
#        avg_tc_short = np.insert(template_timecourse, short_dur, avg_tc_short)
#        avg_tc_short = avg_tc_short[:timecourse_len]        
    short_num_cellseiz = sum(~np.isnan(short_tc))
    sem_tc_short = stats.sem(short_tc, axis=0, nan_policy='omit')
    gaussian_short = sliding_gaussian_window(avg_tc_short, smooth_size) * 1000
    if period == 'onset':
        x_short = np.array(range(-interest_period,-interest_period + len(gaussian_short)))
    elif period == 'offset':
        x_short = np.array(range(-short_dur,-short_dur + len(gaussian_short)))
        
    if period == 'onset':
        medium_tc, _ = avg_seiz_tc(cell_df, period = period, seiz_min_dur = med_dur_start, seiz_max_dur = med_dur_end, onset_period = interest_period, offset_period = med_dur_end, plot=0)
    elif period == 'offset':
        medium_tc, _ = avg_seiz_tc(cell_df, period = period, seiz_min_dur = med_dur_start, seiz_max_dur = med_dur_end, onset_period = med_dur_end, offset_period = interest_period, plot=0)
    medium_num_cellseiz = sum(~np.isnan(medium_tc))
    avg_tc_medium = np.nanmean(medium_tc, axis = 0)
    sem_tc_medium = stats.sem(medium_tc, axis=0, nan_policy='omit')
    gaussian_medium = sliding_gaussian_window(avg_tc_medium, smooth_size) * 1000
    if period == 'onset':
        x_medium = np.array(range(-interest_period,-interest_period + len(gaussian_medium)))
    elif period == 'offset':
        x_medium = np.array(range(-med_dur_end,-med_dur_end + len(gaussian_medium)))

    if period == 'onset':
        long_tc, _ = avg_seiz_tc(cell_df, period = period, seiz_min_dur = long_dur, seiz_max_dur = 'max', onset_period = interest_period, offset_period = szlen, plot=0)
    elif period == 'offset':
        long_tc, _ = avg_seiz_tc(cell_df, period = period, seiz_min_dur = long_dur, seiz_max_dur = 'max', onset_period = szlen, offset_period = interest_period, plot=0) 
    long_num_cellseiz = sum(~np.isnan(long_tc))
    avg_tc_long = np.nanmean(long_tc,axis=0)
    sem_tc_long = stats.sem(long_tc, axis=0, nan_policy='omit')
    gaussian_long = sliding_gaussian_window(avg_tc_long, smooth_size) * 1000
    if period == 'onset':
        x_long = np.array(range(-interest_period,-interest_period + len(gaussian_long)))
    elif period == 'offset':
        x_long = np.array(range(-szlen,-szlen + len(gaussian_long)))
    
    if plot_type == 'subplots':
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharey=True, sharex=True)
        plt.xlim(-interest_period,szlen)
        plt.ylim(0,5)
        ax4.plot(x_all, gaussian_all); ax4.title.set_text('All Seizures n = {}'.format(len(all_tc)))
        ax1.plot(x_short, gaussian_short); ax1.title.set_text('Short Seizures (0-5s), n = {}'.format(len(short_tc)))
        ax2.plot(x_medium, gaussian_medium); ax2.title.set_text('Medium Seizures, (5-10s) n = {}'.format(len(medium_tc)))
        ax3.plot(x_long, gaussian_long); ax3.title.set_text('Long Seizures (>10s), n = {}'.format(len(long_tc)))
        ax1.axvline(x=0,color = 'red', alpha=0.5); ax2.axvline(x=0,color = 'red', alpha=0.5); ax3.axvline(x=0,color = 'red', alpha=0.5); ax4.axvline(x=0,color = 'red', alpha=0.5)
        fig.suptitle('Average Seizure Timecourse for Seizures of Varying Durations, Cells = {}'.format(len(cell_df)))
        plt.xlabel('Time Relative to Seizure Onset (ms)')
        fig.text(0.06, 0.5, 'Firing Rate Spikes/Second', ha='center', va='center', rotation='vertical')
    else:
        plt.figure()
        plt.plot(x_all, gaussian_all, c='k')
        shaded_error_bar(x_all, gaussian_all, sem_tc_all, c='k')
        plt.plot(x_short, gaussian_short, c='b')
        shaded_error_bar(x_short, gaussian_short, sem_tc_short, c='b')
        plt.plot(x_medium, gaussian_medium, c='g')
        shaded_error_bar(x_medium, gaussian_medium, sem_tc_medium, c='g')
        plt.plot(x_long, gaussian_long, c='r')
        shaded_error_bar(x_long, gaussian_long, sem_tc_long, c='r')
        plt.legend(['all n = {}'.format(len(all_tc)),'short (0-5s) n = {}'.format(len(short_tc)),\
        'medium (5-10s) n = {}'.format(len(medium_tc)),'long (>10s) n = {}'.format(len(long_tc))])
        plt.title('Average Seizure Timecourse for Seizures of Varying Durations, Cells = {}'.format(len(cell_df)))
        plt.axvline(x=0,color = 'red', alpha=0.5)
        plt.xlabel('Time Relative to Seizure {}(ms)'.format(period))
        plt.ylabel('Firing Rate Spikes/Second')
        plt.ylim(1,9)
        
        plt.figure()
        plt.title('Histogram of All Seizures Included at Each Time Point')
        plt.plot(x_all, all_num_cellseiz, c= 'k')
        plt.plot(x_short, short_num_cellseiz, c='b')
        plt.plot(x_medium, medium_num_cellseiz, c='g')
        plt.plot(x_long, long_num_cellseiz, c='r')
        plt.axvline(x=0,color = 'red', alpha=0.5)
        plt.legend(['all','short', 'medium','long'])
        plt.xlabel('Time Relative to Seizure {} (ms)'.format(period))
        plt.ylabel('Count')
        
    return x_all, all_tc, x_short, short_tc, x_medium, medium_tc, x_long, long_tc

#%%
#def get_seizs_in_cell(cell,ref_panda = None):
#    if ref_panda == None: # prevents repetitive loading of database
#        ref_panda = bd.load_in_dataframe()
#    rec_id = cell['recording_id'].values[0]
#    seizs = extract_dataframe(ref_panda,recording_id = rec_id,Type = 'Seizure')
#    return seizs
#%%
@panda_input
def cell_fr(cell_names,seiz_panda=False, save_new = 0, **kwargs):
    #this function is for getting the df with avg fr for each cell in dif periods
    #note: if want to adjust the period the fr is taken from, need to edit load_seizures parameters
    #note: can set save_new to a random non-0/1 number to calculate but not save new df
    firing_rate_panda = pd.DataFrame()
    for cell_name in cell_names:
        print('Cell_FR: '+cell_name)
        data = 0
        spk_times,spk_log,cell_times,props= load_cell(cell_name,return_props=True,dictionary='spike_rate_analysis')
        labels = ["seiz_fr","earlyseiz_fr","lateseiz_fr", "nonseiz_fr","preseiz_fr","postseiz_fr","num_seiz_in_cell","seiz_duration","recording_time","num_seiz_in_rec"]
        if save_new == 0:
            if all (keys in props for keys in labels):
                data = [cell_name,props['seiz_fr'],props['earlyseiz_fr'], props['lateseiz_fr'], props['nonseiz_fr'],props['preseiz_fr'],props['postseiz_fr'],props['num_seiz_in_rec'],props['num_seiz_in_cell'],props['seiz_duration'],props['recording_time']]
        if not data:
            seiz_times,seiz_logs = load_seizures(cell_name,seiz_df = seiz_panda)
            if isinstance(seiz_logs, int): #this means returned 0 and no seizures in that recording
                continue 
            
            seiz_fr         = fr(spk_log,seiz_logs['seiz'])
            earlyseiz_fr    = fr(spk_log,seiz_logs['earlyseiz'])
            lateseiz_fr     = fr(spk_log, seiz_logs['lateseiz'])
            nonseiz_fr      = fr(spk_log,seiz_logs['nonseiz'])
            preseiz_fr      = fr(spk_log,seiz_logs['preseiz'])
            postseiz_fr     = fr(spk_log,seiz_logs['postseiz'])
            
            num_seiz_in_rec    = len(seiz_times)
            num_seiz_in_cell        = len(seiz_in_range(seiz_times,cell_times['start'],cell_times['end']))
            seiz_duration   = sum(seiz_logs['seiz'])/1000.0
            recording_time  = len(seiz_logs['seiz'])/1000.0
            data = [cell_name,seiz_fr,earlyseiz_fr,lateseiz_fr, nonseiz_fr,preseiz_fr,postseiz_fr,num_seiz_in_rec,num_seiz_in_cell,seiz_duration,recording_time]
            if save_new == 1:
                save_cell(cell_name,seiz_fr=seiz_fr,earlyseiz_fr = earlyseiz_fr,lateseiz_fr = lateseiz_fr,nonseiz_fr=nonseiz_fr,preseiz_fr=preseiz_fr,postseiz_fr=postseiz_fr,num_seiz_in_cell=num_seiz_in_cell,seiz_duration=seiz_duration,recording_time=recording_time,num_seiz_in_rec=num_seiz_in_rec)
        headers=['Name','seiz_fr','earlyseiz_fr','lateseiz_fr', 'nonseiz_fr','preseiz_fr','postseiz_fr','num_seiz_in_rec','num_seiz_in_cell','seiz_duration','recording_duration']
        cell_firing_rate = pd.DataFrame([data],columns = headers)
        firing_rate_panda = pd.concat([firing_rate_panda,cell_firing_rate],)
        firing_rate_panda = fix_df_index(firing_rate_panda)

    return firing_rate_panda


#%%
def seiz_fr(seiz_df, spk_logs = False, duration= 'full', mode = 'seiz', preseiz = 10000):
    database = bd.load_in_dataframe()
    ids = seiz_df['recording_id'].unique()
    seiz_fr_df = pd.DataFrame()
    seiz_fr = []
    for rec_id in ids:
        rec_df = pd.DataFrame()
        seizs = extract_dataframe(seiz_df,recording_id = rec_id)
        cells = extract_dataframe(database,recording_id = rec_id,Type = "Cell")
        cells = cells['Name'].values[:]
        cells = cells.astype(str)
        print(len(cells))
        for cell in cells:
            cell_df = pd.Series()
            spk_times,spk_log,cell_times = load_cell(cell)
            cell_max = len(spk_log)
            for i,seiz in seizs.iterrows():
                if mode == 'seiz':
                    start = int(seiz['start']-cell_times['start']) #seiz start
                    end = int(seiz['end']-cell_times['start']) #seiz end
                elif mode == 'preseiz':
                    end = int(seiz['start']-cell_times['start']) #seiz start
                    start = int(end - preseiz) #before seizure start
                if start > 0 and end < cell_max:
                    spk_log_subset = spk_log[start:end]
                elif start < 0 and end < cell_max and end > 0:
                    spk_log_subset = spk_log[:end]
                elif start > 0 and end > cell_max and start < cell_max:
                    spk_log_subset = spk_log[start:]
                elif start < 0 and end > cell_max:
                    spk_log_subset = spk_log[:]
                else:
                    spk_log_subset = [float('nan')]
                if spk_logs:
                    cell_df[str(seiz['Name'])]=spk_log_subset
                else:
                    if isinstance(duration,basestring) != 1:
                        if len(spk_log_subset) > int(duration*1000):
                            spk_log_subset = spk_log_subset[:int(duration*1000)]
                    fr = sum(spk_log_subset)/(len(spk_log_subset)/1000.0)
                    cell_df[str(seiz['Name'])]=fr
                    seiz_fr.append(fr)
            rec_df[cell] = cell_df
        seiz_fr_df = seiz_fr_df.append(rec_df,sort=False)
        
    return seiz_fr_df,seiz_fr

#%%
def seiz_portion_fr_tc(cell_names,seiz_df, period = 'onset', trunc = 'shortest'):
    tc_list = []
    min_tc = float('inf')
    for cell_name in cell_names:
        spk_times,spk_log,cell_times= load_cell(cell_name)
        for i,seiz in seiz_df.iterrows(): #iterate through seizures in recording
            tc_start = int(seiz['start']-cell_times['start'])#seiz start in cell time
            tc_end   = int(seiz['end']-cell_times['start'])#seiz end in cell time
            tc = np.array([spk_log[tc_start:tc_end]]) #array of single timecourse
            if tc.size == 0:
                continue
            tc = tc.ravel()
            min_tc = min(min_tc,len(tc))
            tc_list.append(tc)
    tc_list = pad_epochs_to_equal_length(data_list = tc_list, pad_val=np.nan, align = period)
    tc_array = np.array(tc_list)
    if trunc == 'shortest':
        if period == 'onset':
            tc_array = tc_array[:,:min_tc]
        elif period == 'offset':
            length = len(tc_array)
            tc_array = tc_array[:,length - min_tc:]
    avg_tc = np.nanmean(tc_array,axis=0)
    return avg_tc

#%%
@panda_input
def plot_cell_fr_changes(cell_names, period = 'earlyseiz_fr'):
    #period can be 'earlyseiz_fr' or 'seiz_fr' for entire seizure
    data_dir = '/mnt/Data4/GAERS_Data/'
    cell_frs = np.zeros([len(cell_names),2])
    for i in range(len(cell_names)):
        cell_name = cell_names[i]
        cell_file = data_dir + cell_name
        with open(cell_file,'r') as p:
            cell_info = pickle.load(p)
            cell_fr = cell_info.properties['spike_rate_analysis']
            cell_frs[i,0] = cell_fr['preseiz_fr']
            cell_frs[i,1] = cell_fr[period]
    cell_frs = cell_frs[~np.isnan(cell_frs).any(axis=1)]
    cell_frs = cell_frs.transpose()
    
    plt.figure()
    ax = plt.gca()
    plt.plot(cell_frs)
    plt.title('Change in Cell Firing Rate between Pre-Seizure Control and {}'.format(period))
    plt.ylabel('Firing Rate (Spikes/s)')
    plt.xticks([0, 1], ['Pre-Seizure', period])
    plt.text(0.80,0.99,"Number of Cells: {}".format(cell_frs.shape[1]),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    
    x = np.reshape(cell_frs[0,:],(-1,1)) #This is the control firing rate
    y = np.reshape(cell_frs[1,:] - cell_frs[0,:],(-1,1)) #Ictal fr - control fr
    plt.figure()
    ax = plt.gca()
    plt.scatter(x, y)
    plt.title('Difference in Cell Firing Rate between Control and {} based on Control Firing'.format(period))
    plt.ylabel('Firing Rate Difference between Ictal and Control Periods(Spikes/s)')
    plt.xlabel('Control Firing Rate (Spikes/s)')
    plt.text(0.70,0.99,"Number of Cells: {}".format(cell_frs.shape[1]),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
#    reg = LinearRegression().fit(x,y)
#    r_sq = reg.score(x,y)
#    m = reg.coef_
#    b = reg.intercept_
#    plt.plot(x,(m*x + b))
#    plt.legend(['Linear Regression'])
#    plt.text(0.80,0.89,"Linear Regression:\nR-squared = {}\nm = {}".format(r_sq,m[0][0]),
#                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    pearson = stats.pearsonr(x,y)
    plt.text(0.80,0.69,"Pearson Correlation:\nCoefficient = {}\np-value = {}".format(pearson[0][0],pearson[1][0]),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    
    x = cell_frs[0,:] #This is the control firing rate
    y = (cell_frs[1,:] -  cell_frs[0,:]) / cell_frs[0,:] #Percent difference of ictal from control
    plt.figure()
    ax = plt.gca()
    plt.scatter(x, y)
    plt.title('Percent Difference in Cell Firing Rate between Control and {} based on Control Firing'.format(period))
    plt.ylabel('Percent Firing Rate Difference between Ictal and Control Periods(Spikes/s)')
    plt.xlabel('Control Firing Rate (Spikes/s)')
    plt.text(0.80,0.99,"Number of Cells: {}".format(cell_frs.shape[1]),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    
    plt.figure() #histogram of pre-seizure firing rates
    plt.hist(cell_frs[0,:], bins=100, range = [0,30])
    plt.title('Histogram of Cell Pre-Seizure Firing Rates')
    plt.ylabel('Number of Cells'); plt.xlabel('Firing Rate (spikes/s)')
    plt.ylim(0,25)
    plt.text(0.80,0.99,"Number of Cells: {}".format(cell_frs.shape[1]),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    
    plt.figure() #histogram of seizure firing rates
    plt.hist(cell_frs[1,:], bins=100, range = [0,30])
    plt.title('Histogram of Cell {}'.format(period))
    plt.ylabel('Number of Cells'); plt.xlabel('Firing Rate (spikes/s)')
    plt.ylim(0,25)
    plt.text(0.80,0.99,"Number of Cells: {}".format(cell_frs.shape[1]),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    
    '''
    for i in range(2):
        print(np.nanmean(cell_frs[i,:]))
        print(np.nanstd(cell_frs[i,:]))
        print(stats.sem(cell_frs[i,:], nan_policy = 'omit'))
        print(sta.mean_confidence_interval(cell_frs[i,:]))
    print(stats.ttest_rel(a = cell_frs[0,:], b = cell_frs[1,:]))
    '''
    
    return cell_frs

#%% 
def cell_seiz_impairment(plot = 1, plotted_cell = 'both', plot_none = 0):
    cell_seiztimes = bd.load_in_dataframe('/mnt/Data4/GAERS_Data/Cell_Seizure_Dataframe.pkl') #use this to get which seizs cells fire in
    cell_list = cell_seiztimes['cell_name'].unique()
    num_cells = len(cell_list)
    thal_seizs = pull_seiz_celltype(cell_seiztimes, celltype = 'thalamic')
    cort_seizs = pull_seiz_celltype(cell_seiztimes, celltype = 'cortical')
    spared_num = np.empty(num_cells); spared_num[:] = np.nan;
    impaired_num = np.empty(num_cells); impaired_num[:] = np.nan;
    none_num = np.empty(num_cells); none_num[:] = np.nan;
    idx = 0
    cell_counted = []
    for cell_name in cell_list:
#        break
#        cell_type = np.nan
        #this_cell_thal_seizs = extract_dataframe(thal_seizs, cell_name = cell_name)
        this_cell_cort_seizs = extract_dataframe(cort_seizs, cell_name = cell_name)
        
        #if len(this_cell_thal_seizs) == 0: #that would mean this is a cortical cell
        this_cell_labels = list(this_cell_cort_seizs['seiz_label'])
#            cell_type = 0 #0 can mean cortical
            #if plotted_cell == 'thalamic': #if we're only looking to plot thalamic cells
            #    continue
        #else: #this is otherwise a thalamic cell
            #this_cell_labels = list(this_cell_thal_seizs['seiz_label'])
#            cell_type = 1 #1 can mean thalamic
           # if plotted_cell == 'cortical': #if we're only looking to plot cortical cells
             #   continue
            
        cell_counted.append(idx)
        spared_num[idx] = this_cell_labels.count('Spared')
        impaired_num[idx] = this_cell_labels.count('Impaired')
        none_num[idx] = this_cell_labels.count('None')
        
        
#        this_cell_fr = extract_dataframe(firing_rate_panda, Name = cell_name).iloc[0]
#        seiz_fr[idx] = this_cell_fr['seiz_fr']
#        earlyseiz_fr[idx] = this_cell_fr['earlyseiz_fr']
#        nonseiz_fr[idx] = this_cell_fr['nonseiz_fr']
#        preseiz_fr[idx] = this_cell_fr['preseiz_fr']
#        postseiz_fr[idx] = this_cell_fr['postseiz_fr']
                
        idx +=1
    
    cell_names = [cell_list[i] for i in cell_counted]
    num_counted = len(cell_names)
    spared_num = spared_num[~np.isnan(spared_num)]
    impaired_num = impaired_num[~np.isnan(impaired_num)]
    none_num = none_num[~np.isnan(none_num)]
    
    fig, ax1 = plt.subplots()
    ind = np.arange(num_counted)
    p1 = ax1.bar(ind, spared_num)
    p2 = ax1.bar(ind, impaired_num, bottom = spared_num)
    if plot_none:
        p3 = ax1.bar(ind, none_num, bottom = spared_num+impaired_num)
        ax1.legend((p1[0], p2[0], p3[0]), ('Spared n = {}'.format(np.nansum(spared_num)), 'Impaired n = {}'.format(np.nansum(impaired_num)), 'None n = {}'.format(np.nansum(none_num))), loc = "upper left")
        plt.ylim([0,350])
    else:
        ax1.legend((p1[0], p2[0]), ('Impaired n = {}'.format(np.nansum(spared_num)), 'Spared n = {}'.format(np.nansum(impaired_num))), loc = "upper left")
        plt.ylim([0,150])
    plt.title('Seizures of different severities')
    ax1.set_ylabel('Seizures')
    plt.xticks(ind, cell_list)
    


#%%
def cell_seiz_duration(plot = 1, orderby = np.nan, proportions=0):
    from sklearn.linear_model import LinearRegression
    cell_seiztimes = bd.load_in_dataframe('/mnt/Data4/GAERS_Data/Cell_Seizure_Dataframe.pkl') #use this to get which seizs cells fire in
    firing_rate_panda = bd.load_in_dataframe('/mnt/Data4/GAERS_Data/firing_rate_panda.pkl') #use this to compare to fr
    cell_list = cell_seiztimes['cell_name'].unique()
    num_cells = len(cell_list)
#    cell_seizdur_dict = {}
    short_num = np.empty(num_cells)
    med_num = np.empty(num_cells)
    long_num = np.empty(num_cells)
    seiz_fr = np.empty(num_cells)
    earlyseiz_fr = np.empty(num_cells)
    nonseiz_fr = np.empty(num_cells)
    preseiz_fr = np.empty(num_cells)
    postseiz_fr = np.empty(num_cells)
    idx = 0
    points = []
    for cell_name in cell_list:
        this_cell_seizs = extract_dataframe(cell_seiztimes, cell_name = cell_name)
        this_cell_seizdurs = np.array(this_cell_seizs['seiz_end']) - np.array(this_cell_seizs['seiz_start'])
#        cell_seizdur_dict[cell_name] =  this_cell_seizdurs #originally wanted to do a dict.. but nah
        short_num[idx] = sum(this_cell_seizdurs <= 5000)
        long_num[idx] = sum(this_cell_seizdurs > 10000)
        short_enough = this_cell_seizdurs[this_cell_seizdurs <= 10000] #extra lines because I'm bad at logic
        long_enough = short_enough[short_enough > 5000] #extra lines because I'm bad at logic
        med_num[idx] = len(long_enough)
        this_cell_fr = extract_dataframe(firing_rate_panda, Name = cell_name).iloc[0]
        seiz_fr[idx] = this_cell_fr['seiz_fr']
        earlyseiz_fr[idx] = this_cell_fr['earlyseiz_fr']
        nonseiz_fr[idx] = this_cell_fr['nonseiz_fr']
        preseiz_fr[idx] = this_cell_fr['preseiz_fr']
        postseiz_fr[idx] = this_cell_fr['postseiz_fr']
        
        xs = np.repeat(this_cell_fr['nonseiz_fr'], len(this_cell_seizdurs))
        if idx == 0:
            points = np.array([xs, this_cell_seizdurs])
        else:
            points = np.concatenate((points, np.array([xs, this_cell_seizdurs])), axis = 1)
        plt.scatter(xs, this_cell_seizdurs)
        plt.title('Cell Firing Rate and Seizure Durations')
        plt.xlabel('Cell Nonseizure Firing Rate')
        plt.ylabel('Seizure Duration (ms)')
                
        idx +=1
    
    ax = plt.gca()
    x = np.reshape(points[0,],(-1,1))
    y = np.reshape(points[1,],(-1,1))
    reg = LinearRegression().fit(x,y)
    r_sq = reg.score(x,y)
    m = reg.coef_
    b = reg.intercept_
    plt.plot(x,(m*x + b))
    plt.legend(['Linear Regression'])
    plt.text(0.80,0.89,"Linear Regression:\nR-squared = {}\nm = {}".format(r_sq,m[0][0]),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    pearson = stats.pearsonr(x,y)
    plt.text(0.80,0.69,"Pearson Correlation:\nCoefficient = {}\np-value = {}".format(pearson[0][0],pearson[1][0]),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    
    if proportions:
        sums = short_num + med_num + long_num
        short_num = short_num / sums
        med_num = med_num / sums
        long_num = long_num / sums
    
    if isinstance(orderby, basestring):
        #stupid code for ordering in the most inefficient way possible
        if orderby == 'long_num':
            order = np.argsort(long_num)
        elif orderby == 'short_num':
            order = np.argsort(short_num)
        elif orderby == 'nonseiz_fr':
            order = np.argsort(nonseiz_fr)
        elif orderby == 'seiz_fr':
            order = np.argsort(seiz_fr)
        ord_short_num = np.empty(num_cells)
        ord_med_num = np.empty(num_cells)
        ord_long_num = np.empty(num_cells)
        ord_seiz_fr = np.empty(num_cells)
        ord_earlyseiz_fr = np.empty(num_cells)
        ord_nonseiz_fr = np.empty(num_cells)
        ord_preseiz_fr = np.empty(num_cells)
        ord_postseiz_fr = np.empty(num_cells)
        ord_cell_list = []
        idx=0
        for i in order:
            ord_short_num[idx] = short_num[i]
            ord_med_num[idx] = med_num[i]
            ord_long_num[idx] = long_num[i]
            ord_seiz_fr[idx] = seiz_fr[i]
            ord_earlyseiz_fr[idx] = earlyseiz_fr[i]
            ord_nonseiz_fr[idx] = nonseiz_fr[i]
            ord_preseiz_fr[idx] = preseiz_fr[i]
            ord_postseiz_fr[idx] = postseiz_fr[i]
            ord_cell_list.append(cell_list[i])
            idx+=1

    if plot:
        if not isinstance(orderby, basestring):
            #unordered, both on same plot
            fig, ax1 = plt.subplots()
            ind = np.arange(num_cells)
            p1 = ax1.bar(ind, short_num)
            p2 = ax1.bar(ind, med_num, bottom = short_num)
            p3 = ax1.bar(ind, long_num, bottom = short_num+med_num)
            plt.title('Seizures of different durations')
            ax1.set_ylabel('Seizures')
            plt.xticks(ind, cell_list)
            ax1.legend((p1[0], p2[0], p3[0]), ('Short (<5s) n = {}'.format(sum(short_num)), 'Medium (5-10s) n = {}'.format(sum(med_num)), 'Long (>10s) n = {}'.format(sum(long_num))), loc = "upper left")
            #note: these numbers count certain seizures multiple times if multiple cells fired during them
            ax2 = ax1.twinx()
            ax2.set_ylabel('Firing Rate (spikes/s)')
            ax2.plot(ind, seiz_fr, linewidth = 2, c='tab:purple')
            ax2.plot(ind, earlyseiz_fr, linewidth = 2, c='tab:red')
            ax2.plot(ind, nonseiz_fr, linewidth = 2, c='tab:gray')
            ax2.plot(ind, preseiz_fr, linewidth = 2, c='tab:pink')
            ax2.plot(ind, postseiz_fr, linewidth = 2, c='tab:cyan')
            ax2.legend(['seiz fr', 'early seiz fr (1s of seizure)', 'nonseiz fr', 'preseiz fr (-15 to -5s pre-seizure)', 'postseiz fr (5-15s post-seizure)'], loc = "upper right")
            
            #just the cell histogram
            plt.figure()
            p1 = plt.bar(ind, short_num)
            p2 = plt.bar(ind, med_num, bottom = short_num)
            p3 = plt.bar(ind, long_num, bottom = short_num+med_num)
            plt.title('Seizures of different durations')
            plt.ylabel('Seizures')
            plt.xticks(ind, cell_list)
            plt.legend((p1[0], p2[0], p3[0]), ('Short (<5s) n = {}'.format(sum(short_num)), 'Medium (5-10s) n = {}'.format(sum(med_num)), 'Long (>10s) n = {}'.format(sum(long_num))), loc = "upper left")
            #note: these numbers count certain seizures multiple times if multiple cells fired during them
            
            #just the firing rate lines
            plt.figure()
            plt.ylabel('Firing Rate (spikes/s)')
            plt.plot(ind, seiz_fr, linewidth = 2, c='tab:purple')
            plt.plot(ind, earlyseiz_fr, linewidth = 2, c='tab:red')
            plt.plot(ind, nonseiz_fr, linewidth = 2, c='tab:gray')
            plt.plot(ind, preseiz_fr, linewidth = 2, c='tab:pink')
            plt.plot(ind, postseiz_fr, linewidth = 2, c='tab:cyan')
            plt.legend(('seiz fr', 'early seiz fr (1s of seizure)', 'nonseiz fr', 'preseiz fr (-15 to -5s pre-seizure)', 'postseiz fr (5-15s post-seizure)'), loc = "upper left")
            plt.xticks(ind, cell_list)
            plt.title("Firing Rate in Cells")
            
            #subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            p1 = ax1.bar(ind, short_num)
            p2 = ax1.bar(ind, med_num, bottom = short_num)
            p3 = ax1.bar(ind, long_num, bottom = short_num+med_num)
            plt.title('Seizures of different durations')
            ax1.set_ylabel('Seizures')
            plt.xticks(ind, cell_list)
            ax1.legend((p1[0], p2[0], p3[0]), ('Short (<5s) n = {}'.format(sum(short_num)), 'Medium (5-10s) n = {}'.format(sum(med_num)), 'Long (>10s) n = {}'.format(sum(long_num))), loc = "upper left")
            ax2.set_ylabel('Firing Rate (spikes/s)')
            ax2.plot(ind, seiz_fr, linewidth = 2, c='tab:purple')
            ax2.plot(ind, earlyseiz_fr, linewidth = 2, c='tab:red')
            ax2.plot(ind, nonseiz_fr, linewidth = 2, c='tab:gray')
            ax2.plot(ind, preseiz_fr, linewidth = 2, c='tab:pink')
            ax2.plot(ind, postseiz_fr, linewidth = 2, c='tab:cyan')
            ax2.legend(['seiz fr', 'early seiz fr (1s of seizure)', 'nonseiz fr', 'preseiz fr (-15 to -5s pre-seizure)', 'postseiz fr (5-15s post-seizure)'])
        else:
            #ordered, both on same plot
            fig, ax1 = plt.subplots()
            ind = np.arange(num_cells)
            p1 = ax1.bar(ind, ord_short_num)
            p2 = ax1.bar(ind, ord_med_num, bottom = ord_short_num)
            p3 = ax1.bar(ind, ord_long_num, bottom = ord_short_num+ord_med_num)
            plt.title('Seizures of different durations')
            ax1.set_ylabel('Seizures')
            plt.xticks(ind, ord_cell_list)
            ax1.legend((p1[0], p2[0], p3[0]), ('Short (<5s) n = {}'.format(sum(short_num)), 'Medium (5-10s) n = {}'.format(sum(med_num)), 'Long (>10s) n = {}'.format(sum(long_num))), loc = "upper left")
            #note: these numbers count certain seizures multiple times if multiple cells fired during them
            ax2 = ax1.twinx()
            ax2.set_ylabel('Firing Rate (spikes/s)')
            ax2.plot(ind, ord_seiz_fr, linewidth = 2, c='tab:purple')
            ax2.plot(ind, ord_earlyseiz_fr, linewidth = 2, c='tab:red')
            ax2.plot(ind, ord_nonseiz_fr, linewidth = 2, c='tab:gray')
            ax2.plot(ind, ord_preseiz_fr, linewidth = 2, c='tab:pink')
            ax2.plot(ind, ord_postseiz_fr, linewidth = 2, c='tab:cyan')
            ax2.legend(['seiz fr', 'early seiz fr (1s of seizure)', 'nonseiz fr', 'preseiz fr (-15 to -5s pre-seizure)', 'postseiz fr (5-15s post-seizure)'], loc = "upper right")
            
            #ordered, just seiz and nonseiz
            fig, ax1 = plt.subplots()
            ind = np.arange(num_cells)
            p1 = ax1.bar(ind, ord_short_num)
            p2 = ax1.bar(ind, ord_med_num, bottom = ord_short_num)
            p3 = ax1.bar(ind, ord_long_num, bottom = ord_short_num+ord_med_num)
            plt.title('Seizures of different durations')
            ax1.set_ylabel('Seizures')
            plt.xticks(ind, ord_cell_list)
            ax1.legend((p1[0], p2[0], p3[0]), ('Short (<5s) n = {}'.format(sum(short_num)), 'Medium (5-10s) n = {}'.format(sum(med_num)), 'Long (>10s) n = {}'.format(sum(long_num))), loc = "upper left")
            #note: these numbers count certain seizures multiple times if multiple cells fired during them
            ax2 = ax1.twinx()
            ax2.set_ylabel('Firing Rate (spikes/s)')
            ax2.plot(ind, ord_seiz_fr, linewidth = 5, c='tab:red')
            ax2.plot(ind, ord_nonseiz_fr, linewidth = 5, c='y')
            ax2.legend(['seiz fr', 'nonseiz fr'], loc = "upper right")
            
            #three distributions
            plt.figure()
            plt.hist(short_num, 40)
            plt.hist(med_num, 40)
            plt.hist(long_num, 40)
            plt.title('Proportion of Short/Med/Long Seizures in Cells')
            plt.legend(['Short', 'Medium', 'Long'])
            plt.xlabel('Proportion of SWDs in Cell')
            plt.ylabel('Number of Cells')
        
    return short_num, med_num, long_num, seiz_fr, nonseiz_fr, cell_list
        
#%%
def hist_all_seiz_durations():
    database = bd.load_in_dataframe()
    seiz_df = extract_dataframe(database, Type = 'Seizure')
    seiz_starts = np.array(seiz_df['start'])
    seiz_ends = np.array(seiz_df['end'])
    durs = seiz_ends - seiz_starts
    
    short_num = sum(durs <= 5000)
    long_num = sum(durs > 10000)
    short_enough = durs[durs <= 10000] #extra lines because I'm bad at logic
    long_enough = short_enough[short_enough > 5000] #extra lines because I'm bad at logic
    med_num = len(long_enough)
    
    plt.figure()
    ax = plt.gca()
    plt.hist(durs, bins = 500)
    plt.title('All Seizure Durations (ms)')
    plt.ylabel('Count')
    plt.xlabel('Seizure Duration (ms)')
    plt.axvline(x=5000, c = 'r')
    plt.axvline(x=10000, c = 'r')
    plt.axvline(x=1000, c = 'r')
    plt.text(0.80,0.99,"Number of Short Seizures (<5s): {}".format(short_num),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    plt.text(0.80,0.94,"Number of Medium Seizures (5-10s): {}".format(med_num),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    plt.text(0.80,0.89,"Number of Long Seizures (>10s): {}".format(long_num),
                         horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
    #plt.xlim(0, 110000) 
    #note: these are all seizures, each counted once

#%%
@panda_input
def cell_isi(cell_names,exclude_abnormal=False,**kwargs):
#returns a panda of interspike intervals, plots, and updates saved cell instances
    isi_panda = pd.DataFrame()
    for cell_name in cell_names:
        print('Cell_ISI: '+cell_name)
        spk_times,spk_log,cell_times= load_cell(cell_name)
        isi_array = np.zeros(len(spk_times)-1)
        for i in range(len(isi_array)):
            isi_array[i] = spk_times.iloc[i+1]-spk_times.iloc[i]
        mean_isi = np.mean(isi_array)
        num_isi = len(isi_array)
        num_abnormal_isi = sum(isi_array<=2)
        perc_abnormal_isi = 100*(float(num_abnormal_isi)/num_isi)
        if 'plot' in kwargs:
            plt.figure()
            ax = plt.subplot(111)
            if kwargs['plot']=="zoom":
                ax.hist(isi_array,range(0,500,1))
            elif kwargs['plot'] == "density":
                sns.distplot(isi_array, hist=True, kde=True, bins = range(0,200,1))
#                plt.xlim(0,6000) #calculate kernel density and use to determine if isi is homongeneous or inhomogeneous poisson distribution 
            else:
                ax.hist(isi_array,range(0,int(max(isi_array)),200))
            plt.title('Interspike Interval Histogram')
            plt.xlabel('Interspike Interval (ms)')
            plt.ylabel('Number of Interspike Intervals')
            plt.text(0.3,0.98,"Cell Name: {}\nTotal Number of Interspike Intervals: {}\nMean Interspike Interval(ms): {}".format(cell_name,len(isi_array),mean_isi),
                     horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
            
        data = [[cell_name,isi_array,mean_isi,num_isi,num_abnormal_isi,perc_abnormal_isi]]
        headers=['Name','isi_array','mean_isi','num_isi','num_abnormal_isi','perc_abnormal_isi']
        cell_isi = pd.DataFrame(data,columns = headers)
        isi_panda = pd.concat([isi_panda,cell_isi])
        save_cell(cell_name,mean_isi=mean_isi)
    return isi_panda
#%%
#@panda_input
def plot_seizs_in_cell(cell_df,onset_period = 1000,offset_period = 1000,mode = 'onset',plot = True,exclusion_period = 50,min_dur=True,**kwargs):
    #Plots pre- or post- ictal spiketrains aligned by seizure onset or offset for a given cell
    if min_dur == True and mode =='onset':
        min_dur = offset_period #set minimum duration based on analysis period
    elif min_dur == True and mode == 'offset':
        min_dur = onset_period
    database = bd.load_in_dataframe()      
    for i,cell in cell_df.iterrows():
        seizs_in_rec = extract_dataframe(database,recording_id = cell['recording_id'],Type = 'Seizure')
        spk_times,spk_log,cell_times= load_cell(cell['Name'])
        tc_list = []
        order = np.array([]) #initialize order array with single nan
        for i,seiz in seizs_in_rec.iterrows():
#            break
            if (seiz['start']-onset_period>cell_times['start'] and seiz['end']+offset_period<cell_times['end']) and (seiz['end']-seiz['start'])>=min_dur:
                if mode == 'offset':
                    ref   = int(seiz['end']-cell_times['start']) #Seizure offset in cell time
                else:
                    ref   = int(seiz['start']-cell_times['start'])#Seizure onset in cell time
                start = ref - onset_period
                end = ref + offset_period;
                if mode == 'offset':
                    start = int(min(ref - seiz['end'] + seiz['start'],start))
                else:
                    end = int(min(ref + seiz['end'] - seiz['start'],end))
#                first_spk = next((i for i,x in np.ndenumerate(seiz_tc[0,onset_period+exclusion_period:]) if x!=0), offset_period)
#                try:
#                    first_spk[0]
#                except:
#                    first_spk = [first_spk]
#                order = np.append(first_spk[0],order)
                    
                seiz_tc = np.array([spk_log[start:end]])
                seiz_tc = seiz_tc.ravel()
                tc_list.append(seiz_tc)

            elif (seiz['end']-seiz['start'])<min_dur:
                print('Seiz Duration Below {} ms'.format(min_dur))
            else:
                print('Seiz Not Fully In Cell Firing Time')
        tc_list = pad_epochs_to_equal_length(data_list = tc_list, pad_val=np.nan, align = mode)
        seiz_tc_array = np.array(tc_list)
        temp_seiz_tc_array = seiz_tc_array[:,onset_period + exclusion_period:]
        for row in range(temp_seiz_tc_array.shape[0]):
            first_spk = np.nonzero(temp_seiz_tc_array[row])[0][0]
            order = np.append(order,first_spk)
        order += exclusion_period
        
        num_seiz = seiz_tc_array.shape[0]#number of used seizures = length of seiz_tc_array
        seiz_tc_df = pd.DataFrame()
        for i in range(num_seiz):
            new_df = pd.DataFrame([[seiz_tc_array[i],order[i]]],columns=['seiz_tc_array','first_spk'])
            seiz_tc_df = seiz_tc_df.append(new_df,ignore_index = True)
        seiz_tc_df = seiz_tc_df.sort_values(by = 'first_spk')
#        x = seiz_tc_df['seiz_tc_array'].values[:]
#        x = seiz_tc_df['seiz_tc_array'].to_numpy()
#        x = x.astype(np.dtype(np.int32))
        if plot == True:
            plt.figure(figsize = [20,20])
            rasterplot_series(seiz_tc_df['seiz_tc_array'],onset_period)
            plt.ylim(-1,num_seiz+1)
            plt.xlim(-onset_period,offset_period)
            plt.axvline(x=0,color = 'red', alpha=0.5)
            plt.title(cell['Name'])
            plt.ylabel('Seizure Number')
            plt.xlabel('Time(ms)')
#    return seiz_tc_array #make larger seiz_tc_array for data from each cell
#%%
def seiz_ST_rasterplot(seiz_df,mode = 'onset',onset_time = 4000,offset_time = 5000):
    """Plots cell spiketrains for a given seizure or dataframe of seizures"""
    database = bd.load_in_dataframe()
    for i,seiz in seiz_df.iterrows():
        cells = extract_dataframe(database,recording_id = seiz['recording_id'],Type = 'Cell')
        cells = cells['Name'].values[:]
        cells = cells.astype(str)
        if mode == 'offset':
            ref   = seiz['end']
        else:
            ref   = seiz['start']
        start = ref - onset_time
        end   = ref + offset_time
        plt.figure(figsize = [20,20])
        plt.axvline(x=ref,color = 'red')
        print("Number of Cells:{}".format(len(cells)))
        for i in range(len(cells)):
            if i!=0:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax = plt.subplot(len(cells),1,i+1,sharex = ax)
            else:
                ax = plt.subplot(len(cells),1,i+1)
            cell = cells[i]
            spk_times,spk_log,cell_times = load_cell(cell)
            spk_times = seiz_in_range(spk_times,start,end)
            rasterplot_ax(spk_times,ax)
#            ax.eventplot(spk_times)
            ax.axvline(x=ref,color = 'red', alpha=0.5)
            plt.yticks([])# don't display y ticks
        plt.xlim(start,end)
#%%
def cross_corr_rec(cell_ref,bin_size = 5,analysis_time = 1000,mode = 'seiz',output='notnorm',**kwargs):
    cells_in_rec = find_cell_fam(cell_ref)
    cells_in_rec = remove_from_dataframe(cells_in_rec,Name = cell_ref)
    for i,cell_trg in cells_in_rec.iterrows():
        cell_trg_name = cell_trg['Name']
        print(cell_trg_name)
#       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        cross_corr(cell_ref,cell_trg_name,bin_size=bin_size,analysis_time=analysis_time,mode = 'seiz',output =output,ax = ax1)
        cross_corr(cell_ref,cell_trg_name,bin_size=bin_size,analysis_time=analysis_time,mode = 'nonseiz',output =output,ax=ax2)
#%%
def cross_corr(cell_name1,cell_name2,bin_size = 10,analysis_time = 1000,mode = 'full',output='notnorm',**kwargs):
    #Check Recording Ids
    database = bd.load_in_dataframe()
    recording_id1 = (extract_dataframe(database,Name = cell_name1)['recording_id'].values[0])
    recording_id2 = (extract_dataframe(database,Name = cell_name2)['recording_id'].values[0])
    if recording_id1 != recording_id2:
        print('Cells are from different recordings')
        return None
    #Load Spikes and Seizures
    spk_times1,spk_log1,cell_times1= load_cell(cell_name1)
    spk_times2,spk_log2,cell_times2= load_cell(cell_name2) 
    seiz_times2,seiz_logs2 = load_seizures(cell_name2)
    
    ref_spikes = np.zeros(max(cell_times1['end'],cell_times2['end']))
    ref_spikes[cell_times1['start']-1:cell_times1['end']] = spk_log1# array of reference cell spike times
    target_spikes = np.zeros_like(ref_spikes)
    if mode=='seiz':
        target_spikes[cell_times2['start']-1:cell_times2['end']] = spk_log2*seiz_logs2['seiz'] #array of target cell spike times with spikes during nonseizure times masked out
    elif mode == 'nonseiz':
        target_spikes[cell_times2['start']-1:cell_times2['end']] = spk_log2*seiz_logs2['nonseiz'] #array of target cell spike times with spikes during seizures masked out
    else:
        target_spikes[cell_times2['start']-1:cell_times2['end']] = spk_log2 #array of target cell spike times without masking
    simultaneous_mask = np.zeros_like(ref_spikes)
    simultaneous_mask[cell_times2['start']-1:cell_times2['end']] = 1 #mask of times target cell is recording
    ref_spikes*=simultaneous_mask #mask out times target cell is not recording
    
    num_bins = analysis_time/bin_size
    count_log = np.zeros(2*num_bins)#pre and post spike analysis
    for ref_time in spk_times1:
        if ref_spikes[ref_time-1]==1:
            for bin_num in range(2*num_bins):
                start_bin = ref_time-1 - analysis_time + bin_num*bin_size
                end_bin   = start_bin + bin_size
                count_log[bin_num] += sum(target_spikes[start_bin:end_bin])#does this overlap??
    if 'ax' in kwargs:
        ax = kwargs['ax']
        print(type(ax))
    else:
        fig = plt.figure()
    if output == 'norm':
        count_log/=sum(ref_spikes) #calculates mean spikes per bin
        #spk_times1[spk_times1.duplicated()].index[:] #check for duplicates
        bins_per_sec = 1000.0/bin_size
        count_log *= (bins_per_sec) #calculate each bin as rate
        cell2_nonseiz_fr = fr(spk_log2,seiz_logs2['nonseiz'])
#        cell2_panda = sta.cell_fr(cell_name2)
#        cell2_nonseiz_fr = cell2_panda['nonseiz_fr'].values[0]
        count_log -= cell2_nonseiz_fr
        count_log /= cell2_nonseiz_fr #calculate each bin as perfect change compared to nonseiz firing rate
        count_log *= 100.0
        plt.ylabel('% Change in Target Cell Firing Rate Compared to Non-Seizure Baseline')
    x = np.arange(-analysis_time,analysis_time,bin_size) +bin_size/2   
    ax.bar(x,count_log,width = bin_size)
    plt.xlabel('Time relative to Reference Cell Spike (ms)')
    plt.title('Crosscorrelogram of {} vs. {}'.format(cell_name1,cell_name2))
#    ax.axvline(0,ymax=max(count_log))
    return count_log
#%%
@panda_input
def is_rhythmic(cell_names,mode='seiz', plot=True,**kwargs):
    for cell_name in cell_names:
        print('Is_Rhythmic ' + mode + ': '+str(cell_name))
        spk_times,spk_log,cell_times= load_cell(cell_name)
        seiz_times,seiz_logs = load_seizures(cell_name)
        cell_seizs = seizs_in_cell(cell_name)
        
        ipi_list = []
        ipi_array = np.array([])
        if mode == 'full':
            ipi_array = calc_ipis(spk_log[:],ipi_array,window_size=10)
        elif cell_seizs.size ==0: #if cell fires in no seizures
                if mode=='seiz':
                    ipi_array = []
                elif mode == 'nonseiz':
                    ipi_array = calc_ipis(spk_log,ipi_array,window_size=10)
        elif fr(spk_log,seiz_logs[mode])>0:
            old_end =0;
            cell_seizs = cell_seizs.sort_values(by ='start')
            cell_seizs = fix_df_index(cell_seizs)
            for i,seiz in cell_seizs.iterrows():
                seiz_start = int(seiz['start'] - cell_times['start'])
                seiz_end   = int(seiz['end'] - cell_times['start'])
#                print('{}:{}'.format(seiz_start,seiz_end))
                if mode== 'seiz':
                    ipi_array = calc_ipis(spk_log[seiz_start:seiz_end],ipi_array,window_size=10)#with window size of 10, isi is essentially calculated
                elif mode == 'nonseiz':
                    ipi_array = calc_ipis(spk_log[old_end:seiz_start],ipi_array,window_size=10)
                    old_end = seiz_end
                    if i==len(cell_seizs)-1:#get last nonseizure period
                        ipi_array = calc_ipis(spk_log[old_end:],ipi_array)
#                        print('{}:{}'.format(old_end,seiz_start))
            if not ipi_array.size:
                print('Not Enough Spikes During {} Period'.format(mode))
                ipi_list.append(list([np.nan]))
                continue
        else:
            print('No Spikes During {} Period'.format(mode))
            continue
        if plot:
            if 'ax' in kwargs:
                ax = kwargs['ax']
                print(type(ax))
            else:
                fig = plt.figure()
                ax = plt.gca()
            ax.hist(ipi_array,range(0,200,1))
            plt.text(0.99,0.99,'Number of Interpeak Intervals: {}'.format(len(ipi_array)),horizontalalignment='right',verticalalignment='top',transform = ax.transAxes)
            plt.title(cell_name + ' ' + mode)
            plt.xlabel('Interspike Interval (ms)')
            plt.ylabel('Number of Interspike Intervals')
            
        ipi_list.append(ipi_array)
    if isinstance(ipi_array,list):
        ipi_list.append(list([np.nan]))
        print('No IPIs')
    return ipi_list

def plot_all_ipi_types(cell_names):
    for cell_name in cell_names:
        fig = plt.figure()
        ax1 = fig.add_subplot(131); ax2 = fig.add_subplot(132); ax3 = fig.add_subplot(133); 
        
        seiz_ipi = is_rhythmic(cell_name,mode='seiz',plot=True,ax=ax1)
        nonseiz_ipi = is_rhythmic(cell_name,mode='nonseiz',plot=True,ax=ax2)
        all_ipi = is_rhythmic(cell_name,mode='full',plot=True,ax=ax3)
        
        ax1.title.set_text('Seizure IPI')
        ax2.title.set_text('Nonseizure IPI')
        ax3.title.set_text('All IPI')
        plt.suptitle(cell_name)

#%%
@panda_input
def ipi_KLdivergence(cell_names, mode = 'model'):
    #KL-Divergence, does not look amazing using sampling 
    KL_dict = {}
    if mode == 'model':
        for n_cell in range(len(cell_names)):
            cur_dict = {}
            cur_dict['Name'] = cell_names[n_cell]
            _,full_ipi_model = ipi_EM(cell_names[n_cell],mode='full')
            if isinstance(full_ipi_model,list):
                cur_dict['kl_seiz'] = np.nan
                cur_dict['kl_nonseiz'] = np.nan
                KL_dict[n_cell] = cur_dict
                continue
            _,seiz_ipi_model = ipi_EM(cell_names[n_cell],mode='seiz')
            if isinstance(seiz_ipi_model,list):
                cur_dict['kl_seiz'] = np.nan
            else:
                cur_dict['kl_seiz'] = stats.entropy(seiz_ipi_model, full_ipi_model)
            _,nonseiz_ipi_model = ipi_EM(cell_names[n_cell],mode='nonseiz')
            if isinstance(nonseiz_ipi_model,list):
                cur_dict['kl_nonseiz'] = np.nan
            else:
                cur_dict['kl_nonseiz'] = stats.entropy(nonseiz_ipi_model, full_ipi_model)
            KL_dict[n_cell] = cur_dict
        KL_df = pd.DataFrame.from_dict(KL_dict)
        KL_df = KL_df.transpose()
#    if mode == 'sample':
#        from random import sample
#        min_ipis=500
#        for n_cell in range(len(cell_names)):
#            seiz_ipi = is_rhythmic(cell_names[n_cell], mode='seiz', plot=0)[0]
#            if len(seiz_ipi) < min_ipis:
#                continue
#            seiz_ipi = seiz_ipi[sample(range(len(seiz_ipi)),min_ipis)]
#            nonseiz_ipi = is_rhythmic(cell_names[n_cell], mode='nonseiz', plot=0)[0]
#            nonseiz_ipi = nonseiz_ipi[sample(range(len(nonseiz_ipi)),min_ipis)]
#            all_ipi = is_rhythmic(cell_names[n_cell], mode='full', plot=0)[0]
#            all_ipi = all_ipi[sample(range(len(all_ipi)),min_ipis)]
#            kl_seiz[n_cell] = stats.entropy(seiz_ipi, all_ipi)
#            kl_nonseiz[n_cell] = stats.entropy(nonseiz_ipi, all_ipi)
    return KL_df

#%%
def ipi_EM(cell_name,mode='full', plot=0):
    #Expectation Maximization
    ipi_array = is_rhythmic(cell_name,mode=mode,plot=0)[0]
    if isinstance(ipi_array,list):
        return np.nan, list([np.nan])
    ipi_array = ipi_array[ipi_array < 200]
    if not ipi_array.size > 2:
        print('Not enough short IPIs')
        return np.nan, list([np.nan])
    GMM = GaussianMixture(n_components=2).fit(ipi_array.reshape(-1,1))
    print('Converged: ' + str(GMM.converged_))
#    weights = GMM.weights_
    means = GMM.means_
    covariances = GMM.covariances_
    x = range(0,200,1)
    y1 = stats.multivariate_normal.pdf(x,mean = means[0], cov = covariances[0])
    y2 = stats.multivariate_normal.pdf(x,mean = means[1], cov = covariances[1])
    if plot:
        plt.figure()
        plt.hist(ipi_array,range(0,200,1))
        plt.plot(x,y1*500)
        plt.plot(x,y2*500)
        plt.plot(x,y1*500 + y2*500)
    model = y1 + y2
    return GMM, model

#%%
#def szproportion_EMweights(database = bd.load_in_dataframe(), mode = 'time'):
#    cells = extract_dataframe(database, Type = 'Cell')
#    cells = fix_df_index(cells)
#    plt.figure()
#    plt.xlabel('EM weight')
#    weights = np.empty(len(cells))
#    seiz_proportions = np.empty(len(cells))
#    for index, cell in cells.iterrows():
#        seiz_in_cell = seizs_in_cell(cell['Name'])
#        seiz_log = np.zeros(int(cell['end']- cell['start']+1)) #empty logicals in cell time
#        for idx,seiz in seiz_in_cell.iterrows():#plot seizures
#            start = int(seiz['start']- cell['start']-1)
#            end   = int(seiz['end']- cell['start']-1)
#            seiz_log[start:end]=1
#        if mode == 'time':
#            seiz_duration   = sum(seiz_log)
#            recording_time  = cell['end'] - cell['start']
#            seiz_proportions[index] = seiz_duration / recording_time
#        if mode == 'spks':
#            _,spk_log,_ = load_cell(cell['Name'])
#            sz_spklog = spk_log[seiz_log == 1]
#            seiz_proportions[index] = np.nansum(sz_spklog) / np.nansum(spk_log)
#        weight = ipi_EM(cell['Name']).weights_
#        weights[index] = weight[0]
#        if index == 40:
#            break
#    
#    plt.scatter(weights,seiz_proportions)
#    if mode == 'time':
#        plt.ylabel('Proportion of Cell Recording Time in Seizure')
#    elif mode == 'spks':
#        plt.ylabel('Proportion of Cell Spikes in Seizure')
#    return weights, seiz_proportions

#%%
@panda_input
def spk_to_dur(cell_names):
#    database = bd.load_in_dataframe()
#    this_cell = extract_dataframe(database, Name = cell_name)
    ratios = []
    for cell_name in cell_names: 
        seiz_in_cell = seizs_in_cell(cell_name)
        these_ratios = np.zeros(seiz_in_cell.shape[0])
        spk_times,_,_= load_cell(cell_name)
        spk_times = np.array(spk_times)
        for index,seiz in seiz_in_cell.iterrows():
            seiz_start = seiz['start']
            seiz_dur = seiz['end'] - seiz_start
            rel_spks = spk_times - seiz_start
            first_spk = np.amin(rel_spks[rel_spks > 0])
            if first_spk < seiz_dur:
                these_ratios[index] = first_spk / seiz_dur
        angles = ratio_to_angle(these_ratios)
        ratios.append(these_ratios)
        rose_plot(angles)
        plt.title(cell_name)
    return ratios

#%%
def interseiz_hist():
#this function will plot a histogram of the interseizure intervals
    database = bd.load_in_dataframe()
    seiz_df = extract_dataframe(database,Type="Seizure")
    ids = seiz_df['recording_id'].unique() #find unique recording id's
    intervals_list = []
    for rec_id in ids:
        seizs = extract_dataframe(database,recording_id = rec_id,Type = 'Seizure')
        seiz_starts = np.sort(seizs['start'].values) #all seizure start times
        seiz_ends = np.sort(seizs['end'].values) #all seizure end times
        these_intervals = np.empty(len(seiz_starts)-1)
        for i in range(len(seiz_starts)-1):
            these_intervals[i] = seiz_starts[i+1] - seiz_ends[i]
        intervals_list.append(these_intervals)
    into_array = np.array(pad_epochs_to_equal_length(intervals_list, np.nan))
    into_1d = into_array.ravel()
    interval_array = into_1d[~np.isnan(into_1d)]
    plt.figure()
    plt.title('Histogram of All Interseizure Intervals')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Count')
    plt.hist(interval_array, bins = 700)
    return interval_array

#%%
def plot_dur_avgpreszfr(cell_df, preseiz = 5000, plot=0, seiz_min_dur = 0, seiz_max_dur = 'max', plot_secs = 15):
    database = bd.load_in_dataframe()
    ids = cell_df['recording_id'].unique() #find unique recording id's
    #all_seiz_params = 'fluff'
    all_seiz_params = np.zeros([1,2])
    for rec_id in ids:
#        break
        seizs = extract_dataframe(database,recording_id = rec_id,Type = 'Seizure')
        if seiz_min_dur != 0 or seiz_max_dur != 'no':
            seizs = sep_seizs_by_dur(seizs,seiz_min_dur, seiz_max_dur)
        if len(seizs) == 0:
            continue
        cells_in_rec = extract_dataframe(cell_df,recording_id = rec_id)
        if len(cells_in_rec) == 0:
            continue
#        cell_seiz_params = 'fluff'
        for i,cell in cells_in_rec.iterrows(): #iterate through cells in recording
#            break
            spk_times,spk_log,cell_times= load_cell(cell['Name'])
            cell_end = len(spk_log)
            seiz_params = np.empty([len(seizs),2])
            for i,seiz in seizs.iterrows(): #iterate through seizures in recording
#                break
                ref = int(seiz['start']-cell_times['start'])#seiz start in cell time
                tc_start = ref - preseiz
                tc_end   = ref
                tc = np.full((1,preseiz),np.nan) #empty nan array
                if not ((tc_start<0) | (tc_end>cell_end)):
                    tc_end = int(min(ref + seiz['end'] - seiz['start'],tc_end))
                    tc = np.array([spk_log[tc_start:tc_end]])
                    tc = tc.ravel()
                seiz_params[i,0] = seiz['end'] - seiz['start']
                seiz_params[i,1] = np.mean(tc) * 1000
            all_seiz_params = np.concatenate((all_seiz_params, seiz_params))
    all_seiz_params = np.delete(all_seiz_params,0, axis=0)
    if plot:
        seconds_params = np.floor(all_seiz_params[:,0] / 1000)
        mean_values = np.zeros([plot_secs])
        sem_values = np.zeros([plot_secs])
        num_values = np.zeros([plot_secs])
        for sec in range(plot_secs):
            mask = seconds_params == sec
            these_fr = all_seiz_params[mask,1]
            mean_values[sec] = np.nanmean(these_fr, )
            sem_values[sec] = stats.sem(these_fr, nan_policy = 'omit')
            num_values[sec] = len(these_fr)
        plt.figure()
        plt.plot(range(plot_secs), mean_values)
        plt.fill_between(range(plot_secs),mean_values - sem_values,
                         mean_values + sem_values, alpha = 0.2, color = 'r')
        plt.title('Pre-Seizure Firing Rate for Seizures of Different Durations')
        plt.xlabel('Seizure Duration (1s bins)')
        plt.ylabel('Average Pre-Seizure (5s) Riring Rate')
        plt.ylim([0.5, 5])
    return all_seiz_params


#%%
@panda_input
def find_duplicates(cell_names,data_dir = '/mnt/Data4/GAERS_Data/',**kwargs):
    duplicate_df = pd.DataFrame()
    for cell_name in cell_names:
        print(cell_name)
        cell_file = data_dir + cell_name #load cell
        with open(cell_file,'r') as p:
            cell = pickle.load(p)
        cell_dataframe = pd.DataFrame(cell.cell_data)
        spk_times = cell_dataframe['Spk_Time']
        unique_spks = len(np.unique(spk_times.to_list()))
        total_spks  = len(spk_times.to_list())
        total_indxs = len(np.unique(cell_dataframe.index.values))
        diff = total_spks-unique_spks
        
        data = [[cell_name,diff,unique_spks,total_spks,total_indxs]]
        headers=['Name','Difference','Unique Spikes','Total Spikes','Total Indexes']
        cell_duplicates = pd.DataFrame(data,columns = headers)
        duplicate_df = pd.concat([duplicate_df,cell_duplicates])
    return duplicate_df
#%%
def calc_ipis(seiz_log,ipi_array,window_size = 10):
    new_ipi_array = ipi_array
    if sum(seiz_log)>0:
        tc = sliding_gaussian_window(seiz_log,window_size)#combines close spikes
        peaks,_ = find_peaks(tc,height = 0,distance = window_size/2)#distance = 50 use if using larger gaussian window
        if len(peaks) < 2: #need at least 2 peaks to have an ipi
            return new_ipi_array
        ipi = np.zeros(len(peaks)-1)
        for i in range(len(ipi)):
            ipi[i] = int(peaks[i+1]-peaks[i])
        new_ipi_array = np.append(new_ipi_array,ipi,axis=0)
        if(len(ipi)>(sum(seiz_log)-1)):#should not run
            print('Exception')
            print('{}:{}'.format(len(ipi),sum(seiz_log)))
            plt.figure()
            plt.eventplot(peaks)
            plt.plot(tc*10)
    return new_ipi_array
#%%
def sliding_gaussian_window(spk_log,time_window=100):
    normVector = np.zeros(time_window)#create a normal distribution vector
    for i in range(0,time_window):
        normVector[i] = norm.pdf(i-time_window/2,0,time_window/4)
    gaussian_tc = np.convolve(spk_log,normVector,'same')  #convolve spike function with normal distribution
    return gaussian_tc

#%%
def fr(spk_log,period_log):
    total_spikes = np.matmul(spk_log,period_log)
    total_time   = sum(period_log)/1000.0
    fr           = total_spikes/total_time
    return fr
#%%
def seiz_in_range(seiz_times,start,end): #Add logical functionality?
    new_seiz_times = pd.DataFrame()
    for i,seiz in seiz_times.iterrows():
        if seiz['start']>=start and seiz['end']<=end:
            new_seiz_times = new_seiz_times.append(seiz)
    return new_seiz_times
#%%
def seizs_in_cell(cell_name):
    database = bd.load_in_dataframe()
    cell_df = extract_dataframe(database,Name=cell_name)
    start = int(cell_df['start'].values[0])
    end= int(cell_df['end'].values[0])
    cell_id = cell_df['recording_id'].values[0]
    seizs_in_rec = extract_dataframe(database,Type="Seizure",recording_id=cell_id)#load seizures from same recording
    seizs = seiz_in_range(seizs_in_rec,start,end)
    seizs = fix_df_index(seizs)
    return seizs

#%%
def sep_seizs_by_dur(seiz_df, min_dur, max_dur):
    durations = seiz_df['end'] - seiz_df['start']
    durations = durations.reset_index()
    durations = durations.to_numpy()
    if min_dur != 0:
        durations = durations[durations[:,1] > min_dur,:]
    if max_dur != 'max':
        durations = durations[durations[:,1] <= max_dur,:]
    indices = durations[:,0]
    seiz_df = seiz_df.loc[indices,:]
    seiz_df = fix_df_index(seiz_df)
    return seiz_df

#%%
@panda_input
def plot_spiketrain(cell_names, **kwargs):
    for cell_name in cell_names:
        print(cell_name)
        spk_times,spk_log,cell_times= load_cell(cell_name)
        seiz_times,seiz_logs = load_seizures(cell_name)
        
        #Spike-Train Plot
        fig = plt.figure(figsize = [20,20])
        ax1 = plt.subplot(511)
        ax1.eventplot(spk_times)#rasterplot(spk_times,ax1)
        plt.setp(ax1.get_xticklabels(), fontsize=6)
        plt.xlim(cell_times["start"],cell_times["end"])
        plt.ylim(0.50,1.50)
        plt.yticks([])# don't display y ticks
        plt.title('Spike raster plot')
        plt.ylabel('Spikes')
        plt.setp(ax1.get_xticklabels(), visible=False)
        for index,seiz in seiz_times.iterrows():#plot seizures
            plt.axvspan(seiz['start'],seiz['end'],facecolor='r', alpha=0.5)
        
        #Discrete Bin Calculations
        binWidth = 3000
        ax2 = plt.subplot(512, sharex=ax1)
        #plt.hist(spk_log, bins=range(min(spk_log), max(spk_log) + binWidth, binWidth))
        binx = range(cell_times["start"],cell_times["end"],binWidth)
        numBins = len(binx)
        biny = np.zeros(numBins)
        for i in range(numBins):
            biny[i]= sum(spk_log[binx[i]-cell_times["start"]:binx[i]+binWidth-cell_times["start"]])
            biny[i]/=binWidth/1000
        plt.bar(binx,biny,width=binWidth,align='edge')
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.ylabel('Rate (Hz)')
        for index,seiz in seiz_times.iterrows():#plot seizures
            plt.axvspan(seiz['start'],seiz['end'],facecolor='r', alpha=0.5)
        plt.title('Discrete 3-Second Time Bins')
    
        #Sliding Bin Calc
        binWidth = 3000
        binOverlap = 10
        ax3 = plt.subplot(513, sharex=ax1)
        binx = range(cell_times["start"],cell_times["end"],binWidth/binOverlap)
        numBins = len(binx)
        biny = np.zeros(numBins)
        for i in range(numBins):
            biny[i]= sum(spk_log[binx[i]-binWidth/2-cell_times["start"]:binx[i]+binWidth/2-cell_times["start"]])##fix this line?!?1?1?1?1?1!/1/
            biny[i]/=binWidth/1000
        plt.bar(binx,biny,width=binWidth/binOverlap,align='edge')
        plt.setp(ax3.get_xticklabels(), visible=False)
        plt.ylabel('Rate (Hz)')
        plt.setp(ax3.get_xticklabels(), visible=False)
        for index,seiz in seiz_times.iterrows():#plot seizures
            plt.axvspan(seiz['start'],seiz['end'],facecolor='r', alpha=0.5)
        plt.title('Overlapping 3-Second Time Bins')
    
        #Gaussian Window
#        timeWindow = 100
#        normVector = np.zeros(timeWindow)         #create a normal distribution vector
#        for i in range(0,timeWindow):
#            normVector[i] = norm.pdf(i-timeWindow/2,0,timeWindow/4)
#        gaussianWindow = np.convolve(spk_log,normVector,'same')  #convolve spike function with normal distribution
        gaussian_window = sliding_gaussian_window(spk_log,100) #convolve spike train with gaussian
        x = np.array(range(cell_times["start"],cell_times["end"]+1))#x values for plotting
        ax4 = plt.subplot(514, sharex=ax1)
        plt.plot(x,gaussian_window)
        plt.setp(ax4.get_xticklabels(), visible=False)
        for index,seiz in seiz_times.iterrows():#plot seizures
            plt.axvspan(seiz['start'],seiz['end'],facecolor='r', alpha=0.5)
        plt.title('Sliding Gaussian Window')
    
        #Half-Wave Rectification
        timeWindow = 3000
        alpha = 1.0/timeWindow #500ms time window
        alphaVector = np.zeros(timeWindow)
        for i in range(0,len(alphaVector)):
            alphaVector[i] = (alpha*alpha)*(i**(-1*i*alpha))
            if alphaVector[i]<0.0:
                alphaVector[i] = 0.0
        alphaVector/=sum(alphaVector)
        alphaWindow = np.convolve(spk_log,alphaVector,'same')
        ax5 = plt.subplot(515, sharex=ax1)
        plt.plot(alphaWindow)
        plt.xlabel('Time(ms)')
        for index,seiz in seiz_times.iterrows():#plot seizures
            plt.axvspan(seiz['start'],seiz['end'],facecolor='r', alpha=0.5)
        plt.xlim(cell_times["start"],cell_times["end"])
        plt.title('Half-Wave Rectification')

#%% Helper Functions
def extract_dataframe(dataframe,**kwargs):
    """this function takes in a general dataframe of recording data and returns a
    new dataframe that only contains data for specific columns"""
    if dataframe.empty:
        raise Exception('Dataframe argument is empty')
    for name, value in kwargs.items():
        if name not in dataframe.columns:
            warnings.warn('Dataframe does not contain column: {}'.format(name))
        dataframe = dataframe.loc[dataframe[name]==value]
    dataframe = fix_df_index(dataframe)
    return dataframe

def remove_from_dataframe(dataframe,**kwargs):
    """this function takes in a general dataframe returns a
    new dataframe with inputted rows removed."""
    if dataframe.empty:
        raise Exception('Dataframe argument is empty')
    for name, value in kwargs.items():
        dataframe = dataframe[dataframe[name]!=value]
    dataframe = fix_df_index(dataframe)
    return dataframe

def exclude(df,bad_cells=['2018-08-24_12-32-21_1_1','2018-08-25_19-18-28_1_1','2018-08-24_15-18-20_1_1','2018-10-23_14-41-10_1_1','2018-08-23_15-44-00_1_1','2018-10-08_10-16-37_1_1','2018-10-06_13-39-58_1_1','2018-10-09_09-43-16_1_1','2018-10-09_14-29-36_1_1','2018-10-08_13-33-43_1_1','2018-10-08_13-33-43_2_1','2018-10-08_10-57-15_1_1','2018-10-09_09-43-16_1_2','2018-10-10_09-04-46_1_1','2018-10-09_14-29-36_1_2'
],col='recording_id'):
    for cell in bad_cells:
        df = remove_from_dataframe(df,**{col:cell})
    return df

def pull_select_dataframe(dataframe, names):
    new_dataframe = pd.DataFrame()
    for name in names:
        this_dataframe = extract_dataframe(dataframe, Name = name)
        new_dataframe = pd.concat([new_dataframe,this_dataframe],)
    new_dataframe = fix_df_index(new_dataframe)
    return new_dataframe   

def pull_select_recid_dataframe(dataframe, recids):
    new_dataframe = pd.DataFrame()
    for recid in recids:
        this_dataframe = extract_dataframe(dataframe, recording_id = recid)
        new_dataframe = pd.concat([new_dataframe,this_dataframe],)
    new_dataframe = fix_df_index(new_dataframe)
    return new_dataframe

def pull_seiz_celltype(dataframe, celltype = 'cortical'):
    from itertools import compress
#    dataframe = extract_dataframe(dataframe, Type = 'Seizure')
    new_dataframe = pd.DataFrame()
    rat_names = list(dataframe['Rat'].unique())
    type_log = np.zeros(len(rat_names))
    for i in range(len(rat_names)):
        type_log[i] = rat_names[i].startswith('Sil') #1 for cortical cells
    type_log = type_log == 1
    if celltype == 'cortical':
        rat_idx = list(compress(range(len(type_log)), type_log))
    elif celltype == 'thalamic':
        rat_idx = list(compress(range(len(~type_log)), ~type_log))
    else:
        print('invalid celltype')
        return
    rats = [rat_names[i] for i in rat_idx]
    for rat in rats:
        this_dataframe = extract_dataframe(dataframe, Rat = rat)
        new_dataframe = pd.concat([new_dataframe, this_dataframe],)
    new_dataframe = fix_df_index(new_dataframe)
    return new_dataframe
    
#%%
def df_overlap(df1,df2,column = 'Name',inverse=False):
    if not inverse:
        overlap = df1[df1[column].isin(df2[column])]
    else:
        overlap = df1[~df1[column].isin(df2[column])]
    return overlap
#%%
def save_cell(cell_name,dictionary='spike_rate_analysis',data_dir = '/mnt/Data4/GAERS_Data/',**kwargs):
    cell_file = data_dir + cell_name
    with open(cell_file,'rb') as p:
        cell = pickle.load(p)
        p.close()
    if cell.properties.get(dictionary)==None:
        cell.properties[dictionary] = {}
    for name, value in kwargs.items():
        (cell.properties[dictionary])[name] = value
    with open(cell_file,'wb') as s:
        pickle.dump(cell,s)
        s.close()
    
#%%
def find_cell_fam(cell_name):
    database = bd.load_in_dataframe()
    rec_id = extract_dataframe(database,Name=cell_name)['recording_id'].values[0]
    cells_in_rec = extract_dataframe(database,recording_id = rec_id,Type = "Cell")
    return cells_in_rec

#%%
def fix_df_index(dataframe):
    dataframe = dataframe.reset_index()
    dataframe = dataframe.drop(columns=['index'])
    return dataframe

#%%
#def read_cell_props(cell_name,properties,data_dir = '/mnt/Data4/GAERS_Data/',**kwargs):
#    ##read properties from saved cells
#    if type(properties)=='str':#convert single property into properties list
#        properties = [properties]
#    cell_file = data_dir + cell_name #load cell
#    with open(cell_file,'r') as p:
#        cell = pickle.load(p)
#    return_props = 
##    cell_props = pd.DataFrame(cell.properties)
#    for prop in properties:
#        return_props[prop] = cell.properties[prop]
#    return return_props


#%%
def exclude_sleep_time(recording_id, spk_log, cell_start, cell_end):
    '''
    Removes the spikes that occured during sleep time. 
    
    Args:
        recording_id: recording id or session id
        spk_times: a list of spike times
        
    Returns:
        spk_logs with sleep times set to nan
    '''
    slptms = ld.sess_slp_load(recording_id)
            
    for slptm in slptms:
        start = slptm[0]
        end = slptm[1]
        if start >= cell_end or end <= cell_start:
            continue
        spk_log[max(0,int(start-cell_start)):min(cell_end,int(end-cell_start))] = np.nan
            
    return spk_log
        

#%%
def identify_gone_times(cell_name, spk_times, rec_id, gone_time=60000, save_times=False, save_dir=home+'/mnt/Data4/AnnotateSWD/'):
    '''
    Identifies the times when the cell is gone. 
    
    Args:
        cell_name: name of cell, string
        spk_times: time stamps of cellular spikes relative to session start
        rec_id: recording id
        gone_time: the amount of time in ms for a cell to be considered gone if no spikes
        
    Returns:
        list of start and end time pairs indicating when the cell is gone, relative to session start
    '''
    spk_times_diff = np.array(spk_times[1:]) - np.array(spk_times[:spk_times.shape[0]-1])
    gone_times_start = np.array(spk_times)[np.where(spk_times_diff>gone_time)[0]]
    gone_times_end = np.array(spk_times)[np.where(spk_times_diff>gone_time)[0]+1]
    gone_times_dur = (gone_times_end - gone_times_start)/1000
    gone_times = np.array([gone_times_start,gone_times_end,gone_times_dur]).T
    if save_times:
        with open(save_dir+rec_id+'/'+cell_name+'_gonetms.txt','w') as f:
            f.write(str(gone_times))
    return gone_times

#%%
def load_cell(cell_name,return_props=False,dictionary = 'spike_rate_analysis',data_dir = home+'/mnt/Data4/GAERS_Data/',exclude_sleep=False):
    cell_file = data_dir + cell_name #load cell
    with open(cell_file,'r') as p:
        cell = pickle.load(p)
    cell_dataframe = pd.DataFrame(cell.cell_data)
    rec_id = extract_dataframe(bd.load_in_dataframe(),Name=cell_name)['recording_id'].values[0]
    spk_times = cell_dataframe['Spk_Time'].to_numpy().astype(int)
    gone_times = identify_gone_times(cell_name,spk_times,rec_id,save_times=True)
    spk_log = np.zeros(cell.end-cell.start+1)   #create a log spike vector
    for i in spk_times:
        spk_log[i-cell.start] = 1
    if exclude_sleep:
        spk_log = exclude_sleep_time(rec_id, spk_log, cell.start, cell.end)
    for gone_time in gone_times:
        spk_log[gone_time[0]-cell.start:gone_time[1]-cell.start] = np.nan # remove gone time
    spk_times = np.where(spk_log==1)[0] + cell.start
    if return_props:
        props = cell.properties.get(dictionary)
        return spk_times,spk_log,{"start":cell.start,"end":cell.end},props;
    return spk_times,spk_log,{"start":cell.start,"end":cell.end}



#%%
def load_seizures(cell_name,preseiz_period=10000, earlyseiz_period = 1000, lateseiz_period = 1000, 
                  postseiz_period=10000, offset_aroundseiz = 2000, seiz_df=False, valid_szs=[]):
    database = bd.load_in_dataframe()
    cell_df = extract_dataframe(database,Name=cell_name)
    cell_start = int(cell_df['start'].values[0]);cell_end= int(cell_df['end'].values[0])
    cell_id = cell_df['recording_id'].values[0]
    if not isinstance(seiz_df,pd.DataFrame):
        seiz_df = extract_dataframe(database,Type="Seizure",recording_id=cell_id)#load seizures from same recording
    else:
        seiz_df = extract_dataframe(seiz_df,Type="Seizure",recording_id=cell_id)#load seizures from same recording
#        if len(seiz_df) == 0:
#            return 0, 0 #set output to 0 for ones that don't have this kind of sz in recording
    if len(valid_szs) == 0:
        seiz_times = seiz_df[['start','end']] #get seizure times from seizures
    else:
        seiz_times = valid_szs
    if isinstance(seiz_times,str):
        return 'no', 'no'
    spk_times, spk_log, cell_times = load_cell(cell_name)
    nan_inds = np.isnan(spk_log)
    seiz_log = np.zeros(cell_end-cell_start+1) #empty logicals in cell time
    earlyseiz_log = np.zeros_like(seiz_log)
    lateseiz_log = np.zeros_like(seiz_log)
    preseiz_log = np.zeros_like(seiz_log)
    start_log = np.zeros_like(seiz_log)
    rest_log = np.zeros_like(seiz_log)
    postseiz_log = np.zeros_like(seiz_log)
    baseline_log  = np.ones_like(seiz_log)
    slp_times = ld.sess_slp_load(cell_id)
    if len(slp_times) == 0: # no sleep here
        slp_log = np.zeros_like(seiz_log)
    else:
        slp_times -= cell_start
        slp_log = gf.create_periodlog(slp_times, len(seiz_log))
    for index,seiz in seiz_times.iterrows():#plot seizures
#        break
        start = int(seiz['start']-cell_start-1)
        end   = int(seiz['end']-cell_start-1)
        seiz_log[start:end]=1
        if end-start >= 1000:
            start_log[start:start+1000] = 1
            rest_log[start+1000:end] = 1
        else: 
            start_log[start:end] = 1
        baseline_log[start:end] = 0
        early_end = min(end-start,earlyseiz_period)
        earlyseiz_log[start:start+early_end]=1
        late_start = min(end-start, lateseiz_period)
        lateseiz_log[end-late_start:end] = 1
        if (start-preseiz_period-offset_aroundseiz)>0:
#            preseiz_log[start-preseiz_period:start-1]=1
            preseiz_log[start-preseiz_period-offset_aroundseiz:start-offset_aroundseiz-1]=1
#        elif (start>0):
        elif (start-offset_aroundseiz > 0):
            preseiz_log[0:start-offset_aroundseiz-1]=1
        if (end+offset_aroundseiz+postseiz_period)<len(seiz_log):
            postseiz_log[end+offset_aroundseiz+1:end+offset_aroundseiz+postseiz_period]=1
        elif (end<len(seiz_log)):
            postseiz_log[end+offset_aroundseiz+1:len(seiz_log)]=1
    nonseiz_log = (seiz_log<1)
    postseiz_log = nonseiz_log*postseiz_log #make sure that none of these periods encompass seizure
    preseiz_log = nonseiz_log*preseiz_log
    baseline_log = baseline_log*(preseiz_log<1)*(postseiz_log<1)*(slp_log<1)
    seiz_log[nan_inds] = np.nan
    nonseiz_log[nan_inds] = np.nan
    start_log[nan_inds] = np.nan
    rest_log[nan_inds] = np.nan
    baseline_log[nan_inds] = np.nan
    postseiz_log[nan_inds] = np.nan
    preseiz_log[nan_inds] = np.nan
    earlyseiz_log[nan_inds] = np.nan
    lateseiz_log[nan_inds] = np.nan
    seiz_logs = {"seiz":seiz_log,"nonseiz":nonseiz_log,'start':start_log,'rest':rest_log,"baseline":baseline_log,"postseiz":postseiz_log,"preseiz":preseiz_log,"earlyseiz":earlyseiz_log, "lateseiz":lateseiz_log}
    return seiz_times,seiz_logs;

#%%
def ratio_to_angle(ratio):
    degrees = 360 * ratio
    return degrees

def rose_plot(angles):
    """
    Compass draws a graph that displays the vectors with
    components `u` and `v` as arrows from the origin.

    Examples
    --------
    >>> import numpy as np
    >>> u = [+0, +0.5, -0.50, -0.90]
    >>> v = [+1, +0.5, -0.45, +0.85]
    >>> compass(u, v)
    Modified code from Github
    """
    
    radii = np.ones(len(angles))
    
    fig, ax = plt.subplots(subplot_kw=dict(polar=True))

    kw = dict(arrowstyle="->", color='k')
    [ax.annotate("", xy=(angle, radius), xytext=(0, 0),
                 arrowprops=kw) for
    angle, radius in zip(angles, radii)]

    ax.set_ylim(0, np.max(radii))

    return fig, ax

#%%
def rasterplot_ax(times,ax):
    for time in times:
        ax.plot(time,0,'.k')
def rasterplot_series(series,onset):
    series.reset_index(inplace=True, drop=True)
    for i,tc_array in series.iteritems():
        for j in range(len(tc_array)):
            if tc_array[j] != 0 :
                plt.plot(j-onset,i,'.k')

#%%
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

def calc_discrete_deriv(data_array):
    #This only does 1ms steps
    data_array_shifted = np.insert(data_array,0,np.nan)
    data_array = np.append(data_array,np.nan)
    diff = data_array_shifted - data_array
    diff = diff[1:len(diff)-1]
    return diff

def shaded_error_bar(x_array, mean_array, sem_array, c = 'r', plot_mean = 0):
    if plot_mean:
        plt.plot(x_array, mean_array)
    plt.fill_between(x_array, mean_array - sem_array,
                         mean_array + sem_array, alpha = 0.2, color = c)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.nanmean(a), stats.sem(a, nan_policy = 'omit')
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

#database = bd.load_in_dataframe()
#cell_name = 'Cell_201810151352491212_7.pkl'
#cell_name = 'Cell_20181012092035115_1100350.pkl'
#cell_df = extract_dataframe(database,Name = cell_name)
#is_rhythmic(cell_df)
#plot_seizs_in_cell(cell_df,1000,1000)
#avg_seiz_tc(cell_df)
#cell_name = 'Cell_20180831082714113_115.pkl'#"Cell_20180824123221111_3017.pkl"
#plot_spiketrain(cell_name)
#is_rhythmic(cell_name)
#z = cell_fr(cell_df)
#seiz_ST_rasterplot(cell_df)


#import os
#def get_immediate_subdirectories(a_dir):
#    return [name for name in os.listdir(a_dir)
#            if os.path.isdir(os.path.join(a_dir, name))]
#for par_dir in sub_dirs:
#    if (not par_dir.startswith('Null')) and (not any(fil.endswith('.txt') for fil in os.listdir(par_dir))):
#        print(par_dir)
