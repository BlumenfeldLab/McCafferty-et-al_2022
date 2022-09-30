#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 17:35:03 2020

These functions will be for studying the pre-seizure state for the unit data.

@author: rjt37
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
import scipy.stats as stats
import scipy.signal as signal
import pickle as pk
import math
import pdb
import time

import GenericFuncsRenee as gf

sys.path.insert(0, '/mnt/Data4/GAERS_Codes/DataExtraction')
import build_database as bd
import trim_database as td
import TimecourseAnalysis as ta
import SpikeTrainAnalysis as sta
import LoadData as ld

sys.path.insert(0, '/mnt/Data4/BehavData/StateEffects')
import bhv_spectrograms as bs

fS = 1000

# power band segregation
vals = np.array([[1,4],[4,10],[8,13],[12,30],[30,55],[65,500]])
band_names = ['delta','theta','alpha','beta','gamma1','gamma2']
#vals = np.array([[1,4],[4,10],[8,13],[12,30],[30,55],[65,500],[15,100],[5,12],[300,500]])
#band_names = ['delta','theta','alpha','beta','gamma1','gamma2','spike_power','wave_power','high_gamma']
bands = pd.DataFrame(vals,index=band_names,columns=['low','high'])
del vals, band_names

# time segregation
timebins = [[-60,-50],[-50,-40],[-40,-30],[-30,-20],[-20,-10],[-10,-5],[-5,0]]
timelabels = []
for i in range(len(timebins)):
    label = '{} to {}'.format(timebins[i][0],timebins[i][1])
    timelabels.append(label)

#%%

def baselinefr_hist(cell_label = 'Cortical'):
    cell_df = ta.get_database(classtype='Cell',label=cell_label)
    
    try:
        cell_fr = sta.cell_fr(cell_df)
    except:
        cell_fr = sta.cell_fr(cell_df, save_new=1)
    
    plt.figure(figsize=(12,8))
    plt.hist(cell_fr['nonseiz_fr'],bins=30)
    plt.xlabel('non-seizure firing rate')
    plt.ylabel('number of cells')
    plt.title('{} cell baseline firing rate histogram'.format(cell_label))
    
    plt.figure(figsize=(12,8))
    plt.hist(cell_fr['preseiz_fr'][~np.isnan(cell_fr['preseiz_fr'])],bins=30)
    plt.xlabel('10s pre-seizure firing rate (-12 to -2s)')
    plt.ylabel('number of cells')
    plt.title('{} cell pre-seizure firing rate histogram'.format(cell_label))

'''
%matplotlib qt
spect_dict = sa.calc_spect_mean(cell_label = 'Cortical', presz_dur = 120, sz_dur = 10,
                                baseline = 20, window_size = 1, overlap = 0.5, save=0, plot=1, movie=0, filt=1)
spect_dict = sa.calc_spect_mean(cell_label = 'Cortical', presz_dur = 120, sz_dur = 10,
                                baseline = 20, window_size = 2, overlap = 0.5, save=0, plot=1, movie=0, filt=1)

gf.plot_spect(spect_dict['s'], spect_dict['t']-presz_dur, 'mean of all seizures, mean-normalized, n={}'.format(spect_dict['idx']))
gf.plot_spect(spect_dict['s'][:,40:], spect_dict['t'][40:]-presz_dur, 'mean of all seizures, mean-normalized, n={}'.format(spect_dict['idx'])) # cut out early


bs.plot_pwr_bands(spect_dict['bands'], presz_dur, spect_dict['f'], spect_dict['t']-presz_dur, 'all ')
bs.plot_timebin_freqpower(spect_dict['freqpwr'], presz_dur, spect_dict['f'], overlap=0.5, label= 'all ')

'''

def calc_spect_mean(cell_label = 'Cortical', presz_dur = 120, sz_dur = 10, 
                    baseline = 20, window_size = 1, overlap = 0.5, save=0, plot=0, movie=0, filt=1):
    #test = []
    presz_dur *= fS; sz_dur *= fS; window_size *= fS;
    data = bd.load_in_dataframe(); data = td.extract_dataframe(data,Type='Seizure')
    data = td.pull_seiz_celltype(data, celltype = cell_label)
#    spects = np.zeros([fS/2+1, int((presz_dur/fS+sz_dur/fS)/overlap)-1])
    spects = np.zeros([fS/2+1, int((presz_dur/fS+sz_dur/fS)/overlap)-1])
    spect_bands = np.zeros([len(bands),spects.shape[1], 10000]); spect_bands.fill(np.nan)
    spect_freqpwrs = np.zeros([len(timebins),spects.shape[0], 10000]); spect_freqpwrs.fill(np.nan)
    idx = 0
    sessions = data['recording_id'].to_list()
    sessions_unique = list(set(sessions))
    if movie:
        _, t, _ = gf.calc_spect(np.ones(presz_dur+sz_dur), window_size = window_size, overlap = overlap, sampfreq = fS, plot=0)
        fig = plt.figure(figsize=(16,12))
        #plt.colorbar()
        plt.xlabel('time (s)')
        plt.ylabel('frequency')
        xs = range(0,t.shape[0], 20)
        plt.xticks(xs, np.array(list(t-presz_dur/fS)[0::20]) - 0.5)
        ims = []
    for sess in sessions_unique:
#        break
        # load eeg
        eeg = ld.sess_eeg_load(sess)

        sess_data = td.extract_dataframe(data, recording_id = sess)
        #ends = np.array(sess_data['end'])
        
        # make a sztms and logs
        sztms = sess_data[['start','end','Duration']].as_matrix() # this is new to this script
        if sztms.size == 0: # if there are not seizures in session, skip
            continue
        elif sztms.size < 5: # if there's only a single seizure in the session
            sztms = np.reshape(sztms,(1,-1))
        #sztms = np.round(sztms[:,1:]*fS).astype(int)
        szlog = gf.create_periodlog(sztms,len(eeg))
        szless_eeg = eeg.copy(); szless_eeg[np.where(szlog)[0]] = np.nan
        #pdb.set_trace()
        sztms = np.concatenate((np.zeros([1,3]),sztms)).astype(int) #for convenience in finding prev seiz

        this_sess_inds = [i for i,s in enumerate(sessions) if s == sess]
        sess_data = data.iloc[this_sess_inds,:]
        
        for row_ind, row in sess_data.iterrows():
#            break

            # skipping over seizures
            szstart = int(row['start'])
            start = int(np.amax([szstart - presz_dur,0])) # sz start - preseiz duration, or 0 if not enough time
            end = int(np.amin([row['end'], szstart + sz_dur])) # sz end or truncate
            
            this_eeg = szless_eeg.copy()
            this_eeg[szstart:end] = eeg[szstart:end] # fill in this seizure
            #pdb.set_trace()
            avail_eeg = np.empty(presz_dur+sz_dur); avail_eeg.fill(np.nan)
            avail_eeg[presz_dur-szstart+start:presz_dur-szstart+start+end-start] = this_eeg[start:end] # put the eeg values in, aligned
            
            f,t,s = gf.calc_spect(avail_eeg, window_size = window_size, overlap = overlap, sampfreq = fS, plot=0)
            
#            f,t,s = gf.calc_spect(avail_eeg, window_size = window_size, overlap = overlap, sampfreq = fS, plot=1)
#            pdb.set_trace()
            
            '''
            plt.figure(); plt.imshow(10*np.log(s), origin='lower', aspect = 'auto'); plt.colorbar(); #plt.title('raw spect {}'.format(row['Name'])); 
            plt.plot(freqsum*50,c='w'); plt.axhline(1.5*50,c='w'); plt.show();

            '''
            
            if cell_label == 'Cortical':
                if row['Name'] == 'SWD_20190502122745119.pkl': # temp removal of horrible sz
                    continue
            
            if filt:
                raw_s = s.copy() # copy of raw spect, for movie comparison
                orig_blmean = np.nanmean(raw_s[:,:int(baseline/overlap)],axis=1) # baseline values
                orig_s = raw_s / orig_blmean[:,np.newaxis] # this is normalizing using the original method
                
                # filter pass 1 via seizure power
                szidx = np.where(t-presz_dur/fS>0)[0][0]
                timefilt, s_power = seiz_filt(s,szidx,std_adj=-1)
                
                # filter pass 2
                freqsum = stats.zscore(np.nansum(s_power,0)) # z-scored sum of all frequencies
                timefilt = np.unique(np.concatenate((timefilt, (np.where(np.abs(freqsum) > 1.5)[0])))) # chose a 1.5 SD cutoff
                timefilt_diff = np.diff(timefilt)
                timefilt = np.concatenate((timefilt,timefilt[np.where(timefilt_diff==2)[0]] + 1)) # fill in gaps of 1 bin
                
                s = filt_passthresh(s,timefilt, thresh=20) # check that difference is great enough#
                       
    #            if len(timefilt) > 10:
    #                print('{} timefilt {}'.format(row['Name'], len(timefilt)))
            
            nanless_s = s[np.where(~np.isnan(s))].reshape((s.shape[0],-1))
            baseline_mean = np.mean(nanless_s[:,:int(baseline/overlap)],axis=1) # uses first non-nan baseline (eg20) seconds for baseline
#            baseline_mean = np.nanmean(s[:,:int(baseline/overlap)],axis=1) # recalculate baseline values
            s /= baseline_mean[:,np.newaxis] # normalize the new spect
            # gf.plot_spect(s,t-presz_dur/fS, row['Name'])
            '''
            if filt:
                if cell_label == 'Cortical':
                    # filter pass 3 because life is hard (this filter kills a lot, but does make cleaner)
                    timefilt,_ = seiz_filt(s,szidx,std_adj=1.5)
                    s[:,timefilt] = np.nan
                    if np.sum(np.isnan(np.mean(s,axis=0))) > szidx*0.33: # if over 33% presz is noise, throw out
                        print('tossed {}'.format(row['Name']))
                        continue
            '''

            if movie:
#                plt.title('Spectrogram of seizure {}'.format(row['Name']))
                stacked = np.hstack((orig_s, s))
                ttl = plt.text(0.5, 1.01, row['Name'], horizontalalignment='center',verticalalignment = 'bottom')
                im = plt.imshow(10*np.log(stacked), animated=True, origin='lower', aspect = 'auto',
                                vmin=-30,vmax=50)
                #im.set_label(row['Name'])
                ims.append([im,ttl])
            
            # let's get band values
            these_bands = np.zeros([len(bands),len(t)]);
            k=0
            for _, cutoff in bands.iterrows():
                these_bands[k,:] = gf.calc_pwr(f, t, s, sampfreq=fS, lowfreq = cutoff['low'], highfreq = cutoff['high'], plot=0)
                k+=1
            these_freqpwrs = bs.calc_timebin_freqpower(s, presz_dur/fS, f, overlap=0.5)
            
#            pdb.set_trace()
            spects = np.nansum(np.dstack((spects,s)),2)
            spect_bands[:,:,idx] = these_bands # if error, adjust hard cutoff by top of func
            spect_freqpwrs[:,:,idx] = these_freqpwrs
            idx += 1
   
    presz_dur /= 1000; window_size /= 1000     
    
    spects /= idx
    spect_bands = spect_bands[:,:,:idx]
    spect_freqpwrs = spect_freqpwrs[:,:,:idx]
    spect_dict = {'s':spects,'f':f,'t':t,'idx':idx, 'bands':spect_bands, 'freqpwr':spect_freqpwrs}
        
    if save:
        path = '/mnt/Data4/GAERS_Data/'
        save_name = path + 'all_'+ str(presz_dur)+'pre_' + str(baseline) + 'bl_' + str(window_size) + 'win_' + \
            str(overlap) + 'lap_unitdata.pkl'
        with open(save_name,'wb') as save_file:
            pk.dump(spect_dict, save_file)
        
    if plot:
        gf.plot_spect(spects, t-presz_dur, 'mean of all seizures, mean-normalized, n={}'.format(idx))
        bs.plot_pwr_bands(spect_bands, presz_dur, f, t-presz_dur, 'all ')
        bs.plot_timebin_freqpower(spect_freqpwrs, presz_dur, f, overlap=0.5, label= 'all ')

    if movie:
        ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True,
                                  repeat_delay=1000)
        movsave = '/mnt/Data4/MakeFigures/StateEffects/allseiz_filt2.mp4'
        ani.save(movsave)
                
    return spect_dict

def seiz_filt(s,szidx,std_adj=-1):
    s_power = 10*np.log(s[:,:szidx-1]) # log of power, up to seizure start
    sz_power = np.nanmean(10*np.log(s[7:13,szidx:szidx+1])) # mean power in sz band at seizure start
    sz_power_std = np.nanstd(10*np.log(s[7:13,szidx:szidx+1]))
    high_power = np.amax(s_power[30:,:],0)
    timefilt = np.where(high_power > sz_power + sz_power_std*std_adj)[0] # sz power should be highest because seizure
    return timefilt, s_power

def filt_passthresh(s,timefilt, thresh=20):
    if len(timefilt) > 0: # if there are places with high variance
        sfilt = s.copy(); sfilt[:,timefilt] = np.nan # put nan's where high variance
        dif = np.nanmean(10*np.log(s[:,timefilt]),0) - np.nanmean(10*np.log(sfilt)) # check power dif of tossed points
        if np.sum(dif > thresh) > 0: # a defined threshold for necessary difference
            s[:,timefilt[dif > thresh]] = np.nan # if passed threshold, use spect with tossed values
    return s

'''
filename = '/mnt/Data4/GAERS_Data/' + 'all_120pre_20bl_1win_0.5lap_unitdata.pkl'
with open(filename,'rb') as savedfile:
    spect_dict = pk.load(savedfile)
bs.plot_timebin_freqpower(spect_dict['freqpwr'],120,spect_dict['f'], label='All Sz')
bs.plot_pwr_bands(spect_dict['bands'], 120, spect_dict['f'], spect_dict['t']-120, label = 'All')
presz_dur = 120
gf.plot_spect(spect_dict['s'][:,40:], spect_dict['t'][40:]-presz_dur, 'mean of all seizures, mean-normalized, n={}'.format(spect_dict['idx'])) # cut early
gf.plot_spect(spect_dict['s'], spect_dict['t']-presz_dur, 'all seizures: mean-normalized, n={}'.format(spect_dict['idx']),vmin=-30,vmax=30)

'''


#%%
'''
CELL TYPE TIMECOURSES FOR LONG BEFORE SEIZURE
'''


def class_presz_fr(cell_label='Cortical', presz = 120, smooth_size=500, win_overlap=0.5):
    if cell_label=='Cortical':
        with open('/mnt/Data4/CtxCellClasses.pkl') as f:
            cell_class = pk.load(f)
    elif cell_label=='Thalamic':
         with open('/mnt/Data4/ThalCellClasses.pkl') as f:
            cell_class = pk.load(f)
    classes = list(set(cell_class['Classes'])); classes.remove('Miscellaneous')
    cell_df = td.extract_dataframe(bd.load_in_dataframe(), Type = 'Cell', label=cell_label)
    seiz_df = td.extract_dataframe(bd.load_in_dataframe(), Type = 'Seizure')
    for i, this_class in enumerate(classes):
        these_cells = td.extract_dataframe(cell_class, Classes = this_class)
        this_cell_df = td.pull_select_dataframe(cell_df, these_cells['Cells'])
        this_seiz_df = td.pull_select_recid_dataframe(seiz_df, list(set(this_cell_df['recording_id'])))
        tcs = gf.get_fr_tcs(this_cell_df,this_seiz_df,onset_period=120000,period='onset')
        x = np.array(range(-presz*fS, -presz*fS + tcs.shape[1], int(smooth_size*(1-win_overlap))))
        mean, sem = gf.plot_single_timecourse(x, tcs, cell_label, this_class, smooth_size, win_overlap,plot_n=True)
        plt.figure(20); plt.plot(x, mean); plt.fill_between(x, mean - sem, mean + sem, alpha = 0.2)
    plt.legend(classes); plt.title('Pre-seizure Firing Trends by Cell Class')














