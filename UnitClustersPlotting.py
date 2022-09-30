# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 06:55:19 2021

This function is for plotting measures of cell activity and rhythmicity grouped
by cell firing patterns

It requires data generated in Phase_analysis.py (pSWCth)

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
import seaborn as sns

# First load group attributions of each location

with open(home+'/mnt/Data4/CtxCellClasses.pkl','rb') as f1:
    cortical_assignments = pk.load(f1)
    
cortical_assignments = cortical_assignments.rename(columns={'Cells':'Cell'})
            
with open(home+'/mnt/Data4/ThalCellClasses.pkl','rb') as f2:
    thalamic_assignments = pk.load(f2)
    
thalamic_assignments = thalamic_assignments.rename(columns={'Cells':'Cell'})

assignments = cortical_assignments.append(thalamic_assignments)

    
# First, pSWCth
# Load dataframes (as created by plot_SWC_triggered_avg.py, function plot_SWC_avg)

with open(home+'/mnt/Data4/MakeFigures/TestForOD/baseline_pSWCth.pkl') as f3:
    pSWCth_bl = pk.load(f3)

with open(home+'/mnt/Data4/MakeFigures/TestForOD/seizure_pSWCth.pkl') as f4:
    pSWCth_sz = pk.load(f4)

with open(home+'/mnt/Data4/MakeFigures/TestForOD/szstart_pSWCth.pkl') as f5:
    pSWCth_strt = pk.load(f5)


pSWCth_bl = pSWCth_bl[['Cell','Baseline']]
pSWCth = pSWCth_sz.merge(pSWCth_bl,left_on = 'Cell', right_on = 'Cell')

pSWCth_strt = pSWCth_strt[['Cell','Rest','Start']]
pSWCth = pSWCth.merge(pSWCth_strt,left_on = 'Cell', right_on = 'Cell')

locations = pSWCth.Type.unique()
patterns = cortical_assignments.Classes.unique()
patterns = patterns[0:4] # get rid of miscellaneous group for this analysis

# And then set up peak and troughs (as derived from figures)

PT_ctx_1s = np.array([78,103,173,198,104,172])
PT_ctx_all = np.array([58,77,179,198,78,178])
PT_thl_1s = np.array([68,89,173,194,90,172])
PT_thl_all = np.array([43,68,173,192,69,172])

PT_ctx_SD = np.array([62,63,190,197,64,189])
PT_ctx_SI = np.array([55,93,168,190,94,167])
PT_ctx_OP = np.array([58,79,170,194,80,169])
PT_ctx_NC = np.array([62,88,177,195,89,176])
PT_thl_SD = np.array([47,59,175,193,60,174])
PT_thl_SI = np.array([41,82,176,190,83,175])
PT_thl_OP = np.array([53,85,148,190,86,147])
PT_thl_NC = np.array([44,79,172,192,80,171])

PT = pd.DataFrame(index=locations,columns=['Sustained Decrease','Sustained Increase','Onset Peak','No Change','All Seizure'])
PT.set_value('Cortical','Sustained Decrease',PT_ctx_SD)
PT.set_value('Cortical','Sustained Increase',PT_ctx_SI)
PT.set_value('Cortical','Onset Peak',PT_ctx_OP)
PT.set_value('Cortical','No Change',PT_ctx_NC)
PT.set_value('Thalamic','Sustained Decrease',PT_thl_SD)
PT.set_value('Thalamic','Sustained Increase',PT_thl_SI)
PT.set_value('Thalamic','Onset Peak',PT_ctx_OP)
PT.set_value('Thalamic','No Change',PT_ctx_NC)
PT.set_value('Cortical','All Seizure',PT_ctx_all)
PT.set_value('Thalamic','All Seizure',PT_thl_all)

jitter = 0.05

for location in locations:
    fig, ax = plt.subplots()
    
    offset = 0


    loc_pSWCth = pSWCth.loc[pSWCth.Type == location]
    Sz_pSWCth = loc_pSWCth.AllSeizures
    Sz_pSWCth = np.array(Sz_pSWCth.tolist())
       
    # and baseline (sham) pSWCth
    Bl_pSWCth = loc_pSWCth.Baseline
    Bl_pSWCth = np.array(Bl_pSWCth.tolist())

    
    # and first second of seizure pSWCth
    Strt_pSWCth = loc_pSWCth.Start
    Strt_pSWCth = np.array(Strt_pSWCth.tolist())
    
    
    # and rest of seizure pSWCth
    Rest_pSWCth = loc_pSWCth.Rest
    Rest_pSWCth = np.array(Rest_pSWCth.tolist())
        
    # now calculate the total firing (sum) during designated peak and trough periods in each cell's pSWCth 
    
    troughs = np.nansum(Sz_pSWCth[:,PT['All Seizure'].loc[location][4]:PT['All Seizure'].loc[location][5]],axis=1)
    basetrgh = np.nansum(Bl_pSWCth[:,PT['All Seizure'].loc[location][4]:PT['All Seizure'].loc[location][5]],axis=1)
    peaks = np.nansum(np.hstack([Sz_pSWCth[:,PT['All Seizure'].loc[location][0]:PT['All Seizure'].loc[location][1]],Sz_pSWCth[:,PT['All Seizure'].loc[location][2]:PT['All Seizure'].loc[location][3]]]),axis=1)
    basepk = np.nansum(np.hstack([Bl_pSWCth[:,PT['All Seizure'].loc[location][0]:PT['All Seizure'].loc[location][1]],Bl_pSWCth[:,PT['All Seizure'].loc[location][2]:PT['All Seizure'].loc[location][3]]]),axis=1)
    
    # the next lines calculate the surplus firing during peaks and deficit in firing during troughs for each cell
    surplus = peaks-basepk
    deficit = (troughs-basetrgh)*-1
    
    x_jitter1 = pd.Series(np.random.normal(loc=0, scale=jitter, size=surplus.shape))+offset
    ax.plot(x_jitter1, surplus, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='r')
    ax.plot([offset-0.2,offset+0.2],[np.mean(surplus),np.mean(surplus)],'k-')
    ax.plot([offset-0.1,offset+0.1],[np.mean(surplus)+ss.sem(surplus),np.mean(surplus)+ss.sem(surplus)],'k-')
    ax.plot([offset-0.1,offset+0.1],[np.mean(surplus)-ss.sem(surplus),np.mean(surplus)-ss.sem(surplus)],'k-')
    ax.plot([offset,offset],[np.mean(surplus)-ss.sem(surplus),np.mean(surplus)+ss.sem(surplus)],'k-')
    
    offset+=1
    
    x_jitter2 = pd.Series(np.random.normal(loc=0, scale=jitter, size=deficit.shape))+offset
    ax.plot(x_jitter2, deficit, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='b')
    ax.plot([offset-0.2,offset+0.2],[np.mean(deficit),np.mean(deficit)],'k-')
    ax.plot([offset-0.1,offset+0.1],[np.mean(deficit)+ss.sem(deficit),np.mean(deficit)+ss.sem(deficit)],'k-')
    ax.plot([offset-0.1,offset+0.1],[np.mean(deficit)-ss.sem(deficit),np.mean(deficit)-ss.sem(deficit)],'k-')
    ax.plot([offset,offset],[np.mean(deficit)-ss.sem(deficit),np.mean(deficit)+ss.sem(deficit)],'k-')


    for pattern in patterns:
        pat_pSWCth = loc_pSWCth.loc[loc_pSWCth.Cluster == pattern]
        
        Sz_pSWCth = pat_pSWCth.AllSeizures
        Sz_pSWCth = np.array(Sz_pSWCth.tolist())
        
#        yvals = np.nanmean(Sz_pSWCth,axis=0)*1000
#        yerr = ss.sem(Sz_pSWCth,axis=0)*1000
#        xvals = np.arange(-200,200,1)
#        
#        plt.figure()
#        plt.plot(xvals,yvals,color='r')
#        plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='pink')
#        plt.ylabel('Spikes per Second')
#        plt.xlabel('Time from SWC Peak (ms)')
#        
#        # and baseline (sham) pSWCth
        Bl_pSWCth = pat_pSWCth.Baseline
        Bl_pSWCth = np.array(Bl_pSWCth.tolist())
#        
#        yvals = np.nanmean(Bl_pSWCth,axis=0)*1000
#        yerr = ss.sem(Bl_pSWCth,axis=0)*1000
#        xvals = np.arange(-200,200,1)
#        
#        plt.plot(xvals,yvals,color='orange')
#        plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='gold')
#        plt.ylabel('Spikes per Second')
#        plt.xlabel('Time from SWC Peak (ms)')
        
#        # and first second of seizure pSWCth
        Strt_pSWCth = pat_pSWCth.Start
        Strt_pSWCth = np.array(Strt_pSWCth.tolist())
#        
#        yvals = np.nanmean(Strt_pSWCth,axis=0)*1000
#        yerr = ss.sem(Strt_pSWCth,axis=0)*1000
#        xvals = np.arange(-200,200,1)
#        
#        plt.plot(xvals,yvals,color='seagreen')
#        plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='springgreen')
#        plt.ylabel('Spikes per Second')
#        plt.xlabel('Time from SWC Peak (ms)')
#        plt.title([location+ ' ' + pattern + ' pSWCth'])
        
#        # and rest of seizure pSWCth
        Rest_pSWCth = pat_pSWCth.Rest
        Rest_pSWCth = np.array(Rest_pSWCth.tolist())
        
#        yvals = np.nanmean(Rest_pSWCth,axis=0)*1000
#        yerr = ss.sem(Rest_pSWCth,axis=0)*1000
#        xvals = np.arange(-200,200,1)
#        
#        plt.plot(xvals,yvals,color='black')
#        plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='lightgray')
#        plt.ylabel('Spikes per Second')
#        plt.xlabel('Time from SWC Peak (ms)')
#        
#        plt.yticks(np.arange(0,35,10))
#        plt.xticks(np.arange(-200,210,100))
        
        # now calculate the total firing (sum) during designated peak and trough periods in each cell's pSWCth 
        
        troughs = np.nansum(Sz_pSWCth[:,PT[pattern].loc[location][4]:PT[pattern].loc[location][5]],axis=1)
        basetrgh = np.nansum(Bl_pSWCth[:,PT[pattern].loc[location][4]:PT[pattern].loc[location][5]],axis=1)
        peaks = np.nansum(np.hstack([Sz_pSWCth[:,PT[pattern].loc[location][0]:PT[pattern].loc[location][1]],Sz_pSWCth[:,PT[pattern].loc[location][2]:PT[pattern].loc[location][3]]]),axis=1)
        basepk = np.nansum(np.hstack([Bl_pSWCth[:,PT[pattern].loc[location][0]:PT[pattern].loc[location][1]],Bl_pSWCth[:,PT[pattern].loc[location][2]:PT[pattern].loc[location][3]]]),axis=1)
        
        # the next lines calculate the surplus firing during peaks and deficit in firing during troughs for each cell
        surplus = peaks-basepk
        deficit = (troughs-basetrgh)*-1
        
        offset+=1

        x_jitter1 = pd.Series(np.random.normal(loc=0, scale=jitter, size=surplus.shape))+offset
        ax.plot(x_jitter1, surplus, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='r')
        ax.plot([offset-0.2,offset+0.2],[np.mean(surplus),np.mean(surplus)],'k-')
        ax.plot([offset-0.1,offset+0.1],[np.mean(surplus)+ss.sem(surplus),np.mean(surplus)+ss.sem(surplus)],'k-')
        ax.plot([offset-0.1,offset+0.1],[np.mean(surplus)-ss.sem(surplus),np.mean(surplus)-ss.sem(surplus)],'k-')
        ax.plot([offset,offset],[np.mean(surplus)-ss.sem(surplus),np.mean(surplus)+ss.sem(surplus)],'k-')
        
        offset+=1
        
        x_jitter2 = pd.Series(np.random.normal(loc=0, scale=jitter, size=deficit.shape))+offset
        ax.plot(x_jitter2, deficit, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='b')
        ax.plot([offset-0.2,offset+0.2],[np.mean(deficit),np.mean(deficit)],'k-')
        ax.plot([offset-0.1,offset+0.1],[np.mean(deficit)+ss.sem(deficit),np.mean(deficit)+ss.sem(deficit)],'k-')
        ax.plot([offset-0.1,offset+0.1],[np.mean(deficit)-ss.sem(deficit),np.mean(deficit)-ss.sem(deficit)],'k-')
        ax.plot([offset,offset],[np.mean(deficit)-ss.sem(deficit),np.mean(deficit)+ss.sem(deficit)],'k-')
        
        data = pd.melt(pd.DataFrame({"Surplus":surplus, "Deficit":deficit}), var_name = 'Location', value_name = 'Firing')
            
#        plt.figure()
#        sns.swarmplot(data = data, x = 'Location', y = 'Firing', hue = 'Location')
#        sns.boxplot(data=data, x='Location', y = 'Firing')
    
        w,p=ss.ttest_rel(surplus,deficit)
        p = p*4 # Bonferroni correction for 4 subgroups
        
        print(location + pattern + ' Full Seizure peak surplus = ' + 
            str(np.around(np.nanmean(surplus),decimals=2)) + ', SEM = ' +
            str(np.around(ss.sem(surplus,nan_policy='omit'),decimals=2)))
        print(location + pattern + ' Full Seizure trough deficit = ' +
            str(np.around(np.nanmean(deficit),decimals=2)) + ', SEM = ' +
            str(np.around(ss.sem(deficit,nan_policy='omit'),decimals=2)))
        print(location + pattern + ' Surplus v Deficit Wilcoxon p = ' +
            str(np.around(p,decimals=100)) + ', rank sum = ' +
            str(np.around(w,decimals=2)))

        # The next section (first second peaks & troughs) is only relevant for Onset Peak groups - should usually be commented out
        troughs1 = np.nansum(Strt_pSWCth[:,88:156],axis=1)
        basetrgh1 = np.nansum(Bl_pSWCth[:,88:156],axis=1)
        peaks1 = np.nansum(np.hstack([Strt_pSWCth[:,55:87],Strt_pSWCth[:,157:192]]),axis=1)
        basepk1 = np.nansum(np.hstack([Bl_pSWCth[:,55:87],Bl_pSWCth[:,157:192]]),axis=1)
        
        # the next lines calculate the surplus firing during peaks and deficit in firing during troughs for each cell
        surplus1 = peaks1-basepk1
        deficit1 = (troughs1-basetrgh1)*-1  # Note there was a mistake here - it used to calculate the deficit as the absolute value of the difference, which inflated it because
        #even occasions on which there was a trough surplus were converted to deficits
        
        data = pd.melt(pd.DataFrame({"Surplus":surplus1, "Deficit":deficit1}), var_name = 'Location', value_name = 'Firing')
         
        fig1,ax1 = plt.subplots()
        offset1 = 0
        
        sz_jitter = pd.Series(np.random.normal(loc=0, scale=jitter, size=surplus.shape))
        ax1.plot(sz_jitter+offset1,surplus,'o',alpha=0.4,zorder=1,ms=8,mew=1,color = 'b')
        ax1.plot([offset1-0.2,offset1+0.2],[np.mean(surplus),np.mean(surplus)],'k-')
        ax1.plot([offset1-0.1,offset1+0.1],[np.mean(surplus)+ss.sem(surplus),np.mean(surplus)+ss.sem(surplus)],'k-')
        ax1.plot([offset1-0.1,offset1+0.1],[np.mean(surplus)-ss.sem(surplus),np.mean(surplus)-ss.sem(surplus)],'k-')
        ax1.plot([offset1,offset1],[np.mean(surplus)-ss.sem(surplus),np.mean(surplus)+ss.sem(surplus)],'k-')
        
        offset1 += 1
        
        strt_jitter = pd.Series(np.random.normal(loc=0, scale=jitter, size=surplus1.shape))
        ax1.plot(strt_jitter+offset1,surplus1,'o',alpha=0.4,zorder=1,ms=8,mew=1,color='r')
        ax1.plot([offset1-0.2,offset1+0.2],[np.mean(surplus1),np.mean(surplus1)],'k-')
        ax1.plot([offset1-0.1,offset1+0.1],[np.mean(surplus1)+ss.sem(surplus1),np.mean(surplus1)+ss.sem(surplus1)],'k-')
        ax1.plot([offset1-0.1,offset1+0.1],[np.mean(surplus1)-ss.sem(surplus1),np.mean(surplus1)-ss.sem(surplus1)],'k-')
        ax1.plot([offset1,offset1],[np.mean(surplus1)-ss.sem(surplus1),np.mean(surplus1)+ss.sem(surplus1)],'k-')

        offset1 += 1
        sz_jitter2 = pd.Series(np.random.normal(loc=0, scale=jitter, size=deficit.shape))
        ax1.plot(sz_jitter2+offset1,deficit,'o',alpha=0.4,zorder=1,ms=8,mew=1,color = 'b')
        ax1.plot([offset1-0.2,offset1+0.2],[np.mean(deficit),np.mean(deficit)],'k-')
        ax1.plot([offset1-0.1,offset1+0.1],[np.mean(deficit)+ss.sem(deficit),np.mean(deficit)+ss.sem(deficit)],'k-')
        ax1.plot([offset1-0.1,offset1+0.1],[np.mean(deficit)-ss.sem(deficit),np.mean(deficit)-ss.sem(deficit)],'k-')
        ax1.plot([offset1,offset1],[np.mean(deficit)-ss.sem(deficit),np.mean(deficit)+ss.sem(deficit)],'k-')
        
        offset1 += 1
        
        strt_jitter2 = pd.Series(np.random.normal(loc=0, scale=jitter, size=deficit1.shape))+offset1
        ax1.plot(strt_jitter2,deficit1,'o',alpha=0.4,zorder=1,ms=8,mew=1,color='r')
        ax1.plot([offset1-0.2,offset1+0.2],[np.mean(deficit1),np.mean(deficit1)],'k-')
        ax1.plot([offset1-0.1,offset1+0.1],[np.mean(deficit1)+ss.sem(deficit1),np.mean(deficit1)+ss.sem(deficit1)],'k-')
        ax1.plot([offset1-0.1,offset1+0.1],[np.mean(deficit1)-ss.sem(deficit1),np.mean(deficit1)-ss.sem(deficit1)],'k-')
        ax1.plot([offset1,offset1],[np.mean(deficit1)-ss.sem(deficit1),np.mean(deficit1)+ss.sem(deficit1)],'k-')
        
        ax1.set_xticks([0.5,2.5])
        ax1.set_xticklabels(['Peak Surplus','Trough Deficit'])
          
        w,p=ss.ttest_rel(surplus,deficit)
        wp,pp = ss.ttest_rel(surplus,surplus1,nan_policy='omit')
        wt,pt = ss.ttest_rel(deficit,deficit1,nan_policy='omit')

        
        print(location + ' First Second peak surplus = ' + 
            str(np.around(np.nanmean(surplus1),decimals=2)) + ', SEM = ' +
            str(np.around(ss.sem(surplus1,nan_policy='omit'),decimals=2)))
        print(location + ' First Second trough deficit = ' +
            str(np.around(np.nanmean(deficit1),decimals=2)) + ', SEM = ' +
            str(np.around(ss.sem(deficit1,nan_policy='omit'),decimals=2)))
        print(location + ' Surplus v Deficit Wilcoxon p = ' +
            str(np.around(p,decimals=1000)) + ', rank sum = ' +
            str(np.around(w,decimals=2)))
        print(location + ' First v Entire Surplus p = ' +
            str(np.around(pp,decimals=10)) + ', state = ' +
            str(np.around(wp,decimals=10)))
        print(location + ' First v Entire Deficit p = ' +
            str(np.around(pt,decimals=10)) + ', state = ' +
            str(np.around(wt,decimals=10)))

        # This ends the Onset Peak-only first second section

        # This section is designed to find the frequency of peaks in each cell's distribution
        #by first smoothing the distribution, finding peaks (with a minimum interval of 100ms)
        #and then getting the difference between the peaks
   
        Sz_freqs = np.empty([1,1])
        Sz_cycs = np.empty([1,1])
        Sz_pks = np.empty([1,3])
        for cell in Sz_pSWCth:
            smoothsz = sig.savgol_filter(cell,21,5)# smooths each individual total seizure distribution
            pks,_ = sig.find_peaks(smoothsz,distance=110) # assuming the minimum peak difference is 120 ms
            if pks.size==3 and np.abs(np.diff(np.diff(pks)))[0] < 10: # check that it can detect the expected number of peaks and that the inter-peak difference remains consistent within 10 ms
                frq = 1000/np.nanmean(np.diff(pks))
                cycle = np.nansum(cell[pks[0]:pks[1]])
            else:
#                print('Wrong Peak Detection')
                pks = np.empty((1,3))
                pks[:]= np.nan
                cycle = np.nan
                frq = np.nan
            Sz_pks = np.vstack([Sz_pks,pks])
            Sz_freqs= np.vstack([Sz_freqs,frq])
            Sz_cycs = np.vstack([Sz_cycs,cycle])
                
    
        Sz_freqs = np.delete(Sz_freqs, (0), axis=0)
        Sz_cycs = np.delete(Sz_cycs, (0), axis=0)
        Sz_pks = np.delete(Sz_pks, (0), axis=0)
        #finds the locations of peaks
        #finds the intervals of peaks in ms and then divides 1000 by the result to get freq in Hz
                    
        print(location + pattern + ' Seizure Frequency = ' +
            str(np.around(np.nanmean(Sz_freqs),decimals=2)) + ', SEM = ' +
            str(np.around(ss.sem(Sz_freqs,nan_policy='omit'),decimals=2)))
        print(location + ' Seizure Cycle Firing = ' +
            str(np.around(np.nanmean(Sz_cycs),decimals=2))+ ', SEM = ' +
            str(np.around(ss.sem(Sz_cycs,nan_policy='omit'),decimals=2)))
 
        # Again the following is for onset peak group only
        Strt_freqs = np.empty([1,1])
        Strt_cycs = np.empty([1,1])
        Strt_pks = np.empty([1,3])
        for cell in Strt_pSWCth:
            smoothsz = sig.savgol_filter(cell,21,5)# smooths each individual total seizure distribution
            pks,_ = sig.find_peaks(smoothsz,distance=110) # assuming the minimum peak difference is 120 ms
            if pks.size==3 and np.abs(np.diff(np.diff(pks)))[0] < 10:
                frq = 1000/np.nanmean(np.diff(pks))
                cycle = np.nansum(cell[pks[0]:pks[1]])
            else:
#                print('Wrong Peak Detection')
                pks = np.empty((1,3))
                pks[:]= np.nan
                cycle = np.nan
                frq = np.nan
            Strt_pks = np.vstack([Strt_pks,pks])
            Strt_freqs= np.vstack([Strt_freqs,frq])
            Strt_cycs = np.vstack([Strt_cycs,cycle])
                
    
        Strt_freqs = np.delete(Strt_freqs, (0), axis=0)
        Strt_cycs = np.delete(Strt_cycs, (0), axis=0)
        Strt_pks = np.delete(Strt_pks, (0), axis=0)
        #finds the locations of peaks
        #finds the intervals of peaks in ms and then divides 1000 by the result to get freq in Hz
        
        w,pf = ss.ttest_rel(np.squeeze(Sz_freqs),np.squeeze(Strt_freqs),nan_policy='omit')
        w,pc = ss.ttest_rel(np.squeeze(Sz_cycs),np.squeeze(Strt_cycs),nan_policy='omit')
        
        #Next we will plot the frequencies in the first second compared to full seizure for onset peak group
        fig2,ax2 = plt.subplots()
        offset1 = 0
        
        sz_jitter = pd.Series(np.random.normal(loc=0, scale=jitter, size=Sz_freqs.size))+offset1        
        ax2.plot(sz_jitter,Sz_freqs,'o',alpha=0.4,zorder=1,ms=8,mew=1,color = 'b')
        ax2.plot([offset1-0.2,offset1+0.2],[np.nanmean(Sz_freqs),np.nanmean(Sz_freqs)],'k-')
        ax2.plot([offset1-0.1,offset1+0.1],[np.nanmean(Sz_freqs)+ss.sem(Sz_freqs,nan_policy='omit'),np.nanmean(Sz_freqs)+ss.sem(Sz_freqs,nan_policy='omit')],'k-')
        ax2.plot([offset1-0.1,offset1+0.1],[np.nanmean(Sz_freqs)-ss.sem(Sz_freqs,nan_policy='omit'),np.nanmean(Sz_freqs)-ss.sem(Sz_freqs,nan_policy='omit')],'k-')
        ax2.plot([offset1,offset1],[np.nanmean(Sz_freqs)-ss.sem(Sz_freqs,nan_policy='omit'),np.nanmean(Sz_freqs)+ss.sem(Sz_freqs,nan_policy='omit')],'k-')
        
        offset1 += 1
        
        strt_jitter = pd.Series(np.random.normal(loc=0, scale=jitter, size=surplus1.shape))+offset1
        ax2.plot(strt_jitter,Strt_freqs,'o',alpha=0.4,zorder=1,ms=8,mew=1,color='r')
        ax2.plot([offset1-0.2,offset1+0.2],[np.nanmean(Strt_freqs),np.nanmean(Strt_freqs)],'k-')
        ax2.plot([offset1-0.1,offset1+0.1],[np.nanmean(Strt_freqs)+ss.sem(Strt_freqs,nan_policy='omit'),np.nanmean(Strt_freqs)+ss.sem(Strt_freqs,nan_policy='omit')],'k-')
        ax2.plot([offset1-0.1,offset1+0.1],[np.nanmean(Strt_freqs)-ss.sem(Strt_freqs,nan_policy='omit'),np.nanmean(Strt_freqs)-ss.sem(Strt_freqs,nan_policy='omit')],'k-')
        ax2.plot([offset1,offset1],[np.nanmean(Strt_freqs)-ss.sem(Strt_freqs,nan_policy='omit'),np.nanmean(Strt_freqs)+ss.sem(Strt_freqs,nan_policy='omit')],'k-')
        
        ax2.set_xticks([0.5])
        ax2.set_xticklabels(['Frequency'])

        
        print(location + pattern + ' First Second Frequency = ' + 
            str(np.around(np.nanmean(Strt_freqs),decimals=3)) + ', SEM = ' +
            str(np.around(ss.sem(Strt_freqs,nan_policy='omit'),decimals=2)))
        print(location + pattern + ' Full Seizure Frequency = ' +
              str(np.around(np.nanmean(Sz_freqs),decimals=3)) + ', SEM = ' +
              str(np.around(ss.sem(Sz_freqs,nan_policy='omit'),decimals=2)))
        print(location + pattern + ' First Second Cycle Firing = ' +
              str(np.around(np.nanmean(Strt_cycs),decimals=2))+ ', SEM = ' +
              str(np.around(ss.sem(Strt_cycs,nan_policy='omit'),decimals=2)))
        print(location + pattern + ' Full Seizure Cycle Firing = ' +
              str(np.around(np.nanmean(Sz_cycs),decimals=2))+ ', SEM = ' +
              str(np.around(ss.sem(Sz_cycs,nan_policy='omit'),decimals=2)))
        print(location + pattern + ' test of frequency = ' +
              str(np.around(pf,decimals=100)))
        print(location + pattern + ' test of cycle firing = ' +
              str(np.around(pc,decimals=50)))
        
    ax.set_xticks(np.arange(0,offset+1)[0::2]+0.5)
    ax.set_xticklabels(np.insert(patterns,0,'All Cells',axis=0))
                    


# Then XCorrs

CellXCorrs = CellXCorrs.merge(cortical_assignments,left_on='Cell',right_on='Cell')
# Matching the assignment to each cell

SzAutoCorr_OP = CellXCorrs.SeizXCorr.loc[CellXCorrs.Classes == 'Onset Peak']
SzAutoCorr_OP = np.array(SzAutoCorr_OP.tolist())
SzAutoCorr_OP = SzAutoCorr_OP/np.nanmean(SzAutoCorr_OP,axis=0)[0]
SzAutoCorr_OP = SzAutoCorr_OP[:,1:]

yvals = np.concatenate((np.flip(np.nanmean(SzAutoCorr_OP,axis=0)),np.nanmean(SzAutoCorr_OP,axis=0)))
yerr = np.concatenate((np.flip(ss.sem(SzAutoCorr_OP,axis=0)),ss.sem(SzAutoCorr_OP,axis=0)))
xvals = np.arange(-200,200,1)

plt.figure()
plt.plot(xvals,yvals,color='r')
plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='pink')
plt.ylabel('Spike Autocorrelation (proportion of ms bins with spike)')
plt.xlabel('Time offset (ms)')
plt.title('Onset Peak')


SzAutoCorr_SD = CellXCorrs.SeizXCorr.loc[CellXCorrs.Classes == 'Sustained Decrease']
SzAutoCorr_SD = np.array(SzAutoCorr_SD.tolist())
SzAutoCorr_SD = SzAutoCorr_SD/np.nanmean(SzAutoCorr_SD,axis=0)[0]
SzAutoCorr_SD = SzAutoCorr_SD[:,1:]

yvals = np.concatenate((np.flip(np.nanmean(SzAutoCorr_SD,axis=0)),np.nanmean(SzAutoCorr_SD,axis=0)))
yerr = np.concatenate((np.flip(ss.sem(SzAutoCorr_SD,axis=0)),ss.sem(SzAutoCorr_SD,axis=0)))
xvals = np.arange(-200,200,1)

plt.figure()
plt.plot(xvals,yvals,color='r')
plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='pink')
plt.ylabel('Spike Autocorrelation (proportion of ms bins with spike)')
plt.title('Sustained Decrease')


SzAutoCorr_SI = CellXCorrs.SeizXCorr.loc[CellXCorrs.Classes == 'Sustained Increase']
SzAutoCorr_SI = np.array(SzAutoCorr_SI.tolist())
SzAutoCorr_SI = SzAutoCorr_SI/np.nanmean(SzAutoCorr_SI,axis=0)[0]
SzAutoCorr_SI = SzAutoCorr_SI[:,1:]

yvals = np.concatenate((np.flip(np.nanmean(SzAutoCorr_SI,axis=0)),np.nanmean(SzAutoCorr_SI,axis=0)))
yerr = np.concatenate((np.flip(ss.sem(SzAutoCorr_SI,axis=0)),ss.sem(SzAutoCorr_SI,axis=0)))
xvals = np.arange(-200,200,1)

plt.figure()
plt.plot(xvals,yvals,color='r')
plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='pink')
plt.ylabel('Spike Autocorrelation (proportion of ms bins with spike)')
plt.xlabel('Time offset (ms)')


SzAutoCorr_NC = CellXCorrs.SeizXCorr.loc[CellXCorrs.Classes == 'No Change']
SzAutoCorr_NC = np.array(SzAutoCorr_NC.tolist())
SzAutoCorr_NC = SzAutoCorr_NC/np.nanmean(SzAutoCorr_NC,axis=0)[0]
SzAutoCorr_NC = SzAutoCorr_NC[:,1:]

yvals = np.concatenate((np.flip(np.nanmean(SzAutoCorr_NC,axis=0)),np.nanmean(SzAutoCorr_NC,axis=0)))
yerr = np.concatenate((np.flip(ss.sem(SzAutoCorr_NC,axis=0)),ss.sem(SzAutoCorr_NC,axis=0)))
xvals = np.arange(-200,200,1)

plt.figure()
plt.plot(xvals,yvals,color='r')
plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='pink')
plt.ylabel('Spike Autocorrelation (proportion of ms bins with spike)')
plt.xlabel('Time offset (ms)')

# Then baseline autocorrs

BlAutoCorr_OP = CellXCorrs.BaselineXCorr.loc[CellXCorrs.Classes == 'Onset Peak']
BlAutoCorr_OP = np.array(BlAutoCorr_OP.tolist())
BlAutoCorr_OP = BlAutoCorr_OP/np.nanmean(BlAutoCorr_OP,axis=0)[0]
BlAutoCorr_OP = BlAutoCorr_OP[:,1:]

yvals = np.concatenate((np.flip(np.nanmean(BlAutoCorr_OP,axis=0)),np.nanmean(BlAutoCorr_OP,axis=0)))
yerr = np.concatenate((np.flip(ss.sem(BlAutoCorr_OP,axis=0)),ss.sem(BlAutoCorr_OP,axis=0)))
xvals = np.arange(-200,200,1)

plt.figure()
plt.plot(xvals,yvals,color='b')
plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='lightblue')
plt.ylabel('Spike Autocorrelation (proportion of ms bins with spike)')
plt.xlabel('Time offset (ms)')
plt.title('Onset Peak')


BlAutoCorr_SD = CellXCorrs.BaselineXCorr.loc[CellXCorrs.Classes == 'Sustained Decrease']
BlAutoCorr_SD = np.array(BlAutoCorr_SD.tolist())
BlAutoCorr_SD = BlAutoCorr_SD/np.nanmean(BlAutoCorr_SD,axis=0)[0]
BlAutoCorr_SD = BlAutoCorr_SD[:,1:]

yvals = np.concatenate((np.flip(np.nanmean(BlAutoCorr_SD,axis=0)),np.nanmean(BlAutoCorr_SD,axis=0)))
yerr = np.concatenate((np.flip(ss.sem(BlAutoCorr_SD,axis=0)),ss.sem(BlAutoCorr_SD,axis=0)))
xvals = np.arange(-200,200,1)

plt.figure()
plt.plot(xvals,yvals,color='b')
plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='lightblue')
plt.ylabel('Spike Autocorrelation (proportion of ms bins with spike)')
plt.title('Sustained Decrease')


BlAutoCorr_SI = CellXCorrs.BaselineXCorr.loc[CellXCorrs.Classes == 'Sustained Increase']
BlAutoCorr_SI = np.array(BlAutoCorr_SI.tolist())
BlAutoCorr_SI = BlAutoCorr_SI/np.nanmean(BlAutoCorr_SI,axis=0)[0]
BlAutoCorr_SI = BlAutoCorr_SI[:,1:]

yvals = np.concatenate((np.flip(np.nanmean(BlAutoCorr_SI,axis=0)),np.nanmean(BlAutoCorr_SI,axis=0)))
yerr = np.concatenate((np.flip(ss.sem(BlAutoCorr_SI,axis=0)),ss.sem(BlAutoCorr_SI,axis=0)))
xvals = np.arange(-200,200,1)

plt.figure()
plt.plot(xvals,yvals,color='b')
plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='lightblue')
plt.ylabel('Spike Autocorrelation (proportion of ms bins with spike)')
plt.xlabel('Time offset (ms)')


BlAutoCorr_NC = CellXCorrs.BaselineXCorr.loc[CellXCorrs.Classes == 'No Change']
BlAutoCorr_NC = np.array(BlAutoCorr_NC.tolist())
BlAutoCorr_NC = BlAutoCorr_NC/np.nanmean(BlAutoCorr_NC,axis=0)[0]
BlAutoCorr_NC = BlAutoCorr_NC[:,1:]

yvals = np.concatenate((np.flip(np.nanmean(BlAutoCorr_NC,axis=0)),np.nanmean(BlAutoCorr_NC,axis=0)))
yerr = np.concatenate((np.flip(ss.sem(BlAutoCorr_NC,axis=0)),ss.sem(BlAutoCorr_NC,axis=0)))
xvals = np.arange(-200,200,1)

plt.figure()
plt.plot(xvals,yvals,color='b')
plt.fill_between(xvals,yvals-yerr,yvals+yerr,color='lightblue')
plt.ylabel('Spike Autocorrelation (proportion of ms bins with spike)')
plt.xlabel('Time offset (ms)')



## Then, firing rate timecourses ##

# First line just re-loads the pickled file (start from here if possible)
SzTransFiringRates = pd.read_pickle(home+'/mnt/Data4/MakeFigures/TestForOD/SzTransFiringRates_SzOnOff.pkl')
SzTransISIcov = pd.read_pickle(home+'/mnt/Data4/MakeFigures/TestForOD/SzTransISIcov_SzOnOff.pkl')

window = 10000
bin_size = 500
# For the (hopefully) final two versions of the dataframes, the resolutions are as follows:
# SzOnOff (peri-seizure): 10000 ms window with 500 ms resolution
# StateMedRes (pre-seiz state)" 120000 ms window with 2000 ms resolution

locations = SzTransFiringRates.Type.unique()
patterns = cortical_assignments.Classes.unique()


# Get the x values for all timecourse plotting in seconds relative to transition time
xmarkinterval = 20 # the space between tick marks on the final x axis


steps = float(bin_size)/fs # intervals between x values in seconds (number of samples per point divided by number of samples per second)
bincount = window/fs # number of seconds in the data (length of window/epoch in ms divided by number of ms per bin)
start = -(bincount) # first x value relative to seizure start time ()
stop = bincount+1 # last x value relative to seizure start time
xvals = np.arange(start,stop,steps) # all x values (from first to last)

szrange = np.min([20,window/fs]) # we want to plot only 20s of seizure at most, as over that the sample size plummets

xvalson = np.arange(0,bincount+szrange,steps) # seizure on x values start at zero and run for one window's worth of bins (pre-seiz) plus the range within seizure we want to plot (in bins)
xvalsoff = np.arange(np.max(xvalson)+xmarkinterval,np.max(xvalson)+xmarkinterval+szrange+bincount,steps)
# Above: seiz off x vals start one interval above the last on value, and again run for one seizure range's worth of bins plus one window's worth of bins
zeroon = xvalson[int(bincount/steps)] # seizure on will be one window into the "on" x values
zerooff = xvalsoff[int(szrange/steps)] # while seizure off will be one seizure range into the "off" x values


slopewin = [bincount-80,bincount-10] # windows over which correlation coefficients will be calculated
# note that these only apply to the state variables (not short window peri-seizure)


xticklocs = np.hstack([np.arange(xvalson[0],xvalson[-1]+xmarkinterval,xmarkinterval),np.arange(xvalsoff[0],xvalsoff[-1]+xmarkinterval,xmarkinterval)])
# locations for x axis ticks will be every nth value from the on and off x vals, where n is the desired tick interval divided by the size of each step
xtickvals = np.hstack([np.arange(start,szrange+1,xmarkinterval),np.arange(-szrange,stop,xmarkinterval)])
# values for x axis ticks will be 


for location in locations:
#    TransFR = SzTransFiringRates.loc[SzTransFiringRates.Type == location]
#    TransFR = SzTransISIcov.loc[SzTransISIcov.Type == location]
    TransFR = TransFR.merge(assignments,left_on='Cell',right_on='Cell')
    fig, ax = plt.subplots()
    offset = 0
    
    SzStartRawFR = TransFR.Start
    SzStartRawFR = np.array(SzStartRawFR.tolist()) # These are just some shenanigans to get values into numpy array
    
    SzEndRawFR = TransFR.End
    SzEndRawFR = np.array(SzEndRawFR.tolist()) # These are just some shenanigans to get values into numpy array
    
#    yvals = np.nanmean(SzStartRawFR[:,0:int((bincount+szrange)/steps)],axis=0)
#    yerr = ss.sem(SzStartRawFR[:,0:int((bincount+szrange)/steps)],axis=0,nan_policy='omit')
#    plt.figure()
#    plt.plot(xvalson,yvals,color='b')
#    plt.fill_between(xvalson,yvals-yerr,yvals+yerr,color='lightblue')
#    plt.axvline(x=zeroon,color='r');
##        plt.ylim([np.nanmin(yvals-yerr)-2,np.nanmax(yvals+yerr)+2])
#    
#    yvals = np.nanmean(SzEndRawFR[:,-int((bincount+szrange)/steps):],axis=0)
#    yerr = ss.sem(SzEndRawFR[:,-int((bincount+szrange)/steps):],axis=0,nan_policy='omit')
#    plt.plot(xvalsoff,yvals,color='b')
#    plt.fill_between(xvalsoff,yvals-yerr,yvals+yerr,color='lightblue')
#    plt.axvline(x=zerooff,color='r')
##    plt.ylabel('Firing Rate (Hz)')
#    plt.ylabel('Inter-Spike Interval variability (ms)')
#    plt.xlabel('Time from Seizure End (s)')
#    plt.title([location + ' by seiz'])
#    plt.xticks(xticklocs,xtickvals)
#    plt.gca().invert_yaxis() # use only for ISI coefficient of variation
    
    # Calculating mean firing rates up to 5 s pre-seizure (transitions only, not state)
#    preszrate = np.nanmean(SzStartRawFR[:,0:int((bincount-5)/steps)],axis=1) # From start of data to 5s before seizure start time
#    szrate = np.nanmean(SzStartRawFR[:,int(bincount/steps):int((bincount+szrange)/steps)],axis=1)
#    preszsem = ss.sem(preszrate,nan_policy='omit')
#    szsem = ss.sem(szrate,nan_policy='omit')

#    w,p = ss.ttest_rel(preszrate,szrate,nan_policy='omit')

#    print(location + ' baseline = ' + 
#          str(np.around(np.nanmean(preszrate),decimals=3)) + ', SEM = ' +
#          str(np.around(preszsem,decimals=3)))
#    print(location + ' seizure = ' + 
#          str(np.around(np.nanmean(szrate),decimals=3)) + ', SEM = ' +
#          str(np.around(szsem,decimals=3)))
#    print(location + ' base v seiz p = ' +
#        str(p))
 

    CellStartRawFR = np.empty([1,np.size(xvalson,0)])
    CellEndRawFR = np.empty([1,np.size(xvalson,0)])
    CellStartNormFR = np.empty([1,np.size(xvalson,0)])
    CellEndNormFR = np.empty([1,np.size(xvalson,0)])

    for cell in TransFR.Cell.unique():
        # Use first definitions of oncellfr and offcellfr (1-line) if looking at FR, second definitions (3-line) if looking at ISI covar
        oncellfr = np.nanmean(np.array(SzTransFiringRates.loc[SzTransFiringRates.Cell==cell].Start.tolist())[:,0:int((bincount+szrange)/steps)],axis=0)
#        oncellfr = np.nanmean(np.array(SzTransISIcov.loc[SzTransISIcov.Cell==cell].Start.tolist())[:,0:int((bincount+szrange)/steps)],axis=0)
#        oncellfr[oncellfr==0]= np.nan
#        oncellfr = 1/(oncellfr)
        CellStartRawFR=np.vstack([CellStartRawFR, oncellfr])
        offcellfr = np.nanmean(np.array(SzTransFiringRates.loc[SzTransFiringRates.Cell==cell].End.tolist())[:,-int((bincount+szrange)/steps):],axis=0)
#        offcellfr = np.nanmean(np.array(SzTransISIcov.loc[SzTransISIcov.Cell==cell].End.tolist())[:,-int((bincount+szrange)/steps):],axis=0)
#        offcellfr[offcellfr==0]= np.nan
#        offcellfr = 1/(offcellfr)
        CellEndRawFR=np.vstack([CellEndRawFR,offcellfr])
        CellStartNormFR=np.vstack([CellStartNormFR, oncellfr/np.nanmax(np.abs(np.append(oncellfr,offcellfr)),axis=0)])
        CellEndNormFR = np.vstack([CellEndNormFR, offcellfr/np.nanmax(np.abs(np.append(oncellfr,offcellfr)),axis=0)])
        
    CellStartRawFR = np.delete(CellStartRawFR, (0), axis=0)    
    CellEndRawFR = np.delete(CellEndRawFR, (0), axis=0)    
    CellStartNormFR = np.delete(CellStartNormFR, (0), axis=0)    
    CellEndNormFR = np.delete(CellEndNormFR, (0), axis=0)    
       
#    yvals = np.nanmean(CellStartNormFR,axis=0)
#    yerr = ss.sem(CellStartNormFR,axis=0,nan_policy='omit')
#    plt.figure()
#    plt.plot(xvalson,yvals,color='b')
#    plt.fill_between(xvalson,yvals-yerr,yvals+yerr,color='lightblue')
#    plt.axvline(x=zeroon,color='r')
#    
#    yvals = np.nanmean(CellEndNormFR,axis=0)
#    yerr = ss.sem(CellEndNormFR,axis=0,nan_policy='omit')
#    plt.plot(xvalsoff,yvals,color='b')
#    plt.fill_between(xvalsoff,yvals-yerr,yvals+yerr,color='lightblue')
#    plt.axvline(x=zerooff,color='r')
##    plt.ylabel('Firing Rate (Proportion of Cell Max)')
#    plt.ylabel('Inter-Spike Interval Covariance (Proportion of Cell Max)')
#    plt.xlabel('Time from Seizure Start (s)')
#    plt.title([location+' normalized by cell'])
#    plt.xticks(xticklocs,xtickvals)
##    plt.gca().invert_yaxis()
#
#    # Testing correlation coefficients between pre-seizure  rates and time
#    preYs = np.nanmean(CellStartNormFR[:,int(slopewin[0]/steps):int(slopewin[1]/steps)],axis=0)
#    r,p = ss.pearsonr(xvals[int(slopewin[0]/steps):int(slopewin[1]/steps)],preYs)
#    plt.text(1, 0.05, ['p = '+ str(p)])
#    
#    print(location + ' ' + pattern + ' -80 to -10 R = ' + 
#      str(np.around(r,decimals=3)) + ', p = ' +
#      str(p))
#
#    
#    yvals = np.nanmean(CellStartRawFR,axis=0)
#    yerr = ss.sem(CellStartRawFR,axis=0,nan_policy='omit')
#    plt.figure()
#    plt.plot(xvalson,yvals,color='b')
#    plt.fill_between(xvalson,yvals-yerr,yvals+yerr,color='lightblue')
#    plt.axvline(x=zeroon,color='r')
##        plt.ylim([np.nanmin(yvals-yerr)-2,np.nanmax(yvals+yerr)+2])
#    
#    yvals = np.nanmean(CellEndRawFR,axis=0)
#    yerr = ss.sem(CellEndRawFR,axis=0,nan_policy='omit')
#    plt.plot(xvalsoff,yvals,color='b')
#    plt.fill_between(xvalsoff,yvals-yerr,yvals+yerr,color='lightblue')
#    plt.axvline(x=zerooff,color='r')
#    plt.ylabel(' Firing Rate')
##    plt.ylabel('Inter-Spike Interval Covariance')
#    plt.xlabel('Time from Seizure End (s)')
#    plt.title([location + ' by cell'])
#    plt.xticks(xticklocs,xtickvals)
#    plt.gca().invert_yaxis()
    
    # Calculating mean firing rates up to 5 s pre-seizure (transitions only, not state) for cell-wise data
    preszrate = np.nanmean(CellStartRawFR[:,0:int((bincount-5)/steps)],axis=1) # From start of data to 5s before seizure start time
    szrate = np.nanmean(CellStartRawFR[:,int(bincount/steps):int((bincount+szrange)/steps)],axis=1)
    preszsem = ss.sem(preszrate,nan_policy='omit')
    szsem = ss.sem(szrate,nan_policy='omit')

    w,p = ss.ttest_rel(preszrate,szrate,nan_policy='omit')
    

    print(location + ' baseline = ' + 
          str(np.around(np.nanmean(preszrate),decimals=3)) + ', SEM = ' +
          str(np.around(preszsem,decimals=3)))
    print(location + ' seizure = ' + 
          str(np.around(np.nanmean(szrate),decimals=3)) + ', SEM = ' +
          str(np.around(szsem,decimals=3)))
    print(location + ' base v seiz p = ' +
        str(p))
    
    x_jitter1 = pd.Series(np.random.normal(loc=0, scale=jitter, size=preszrate.shape))+offset
    ax.plot(x_jitter1, preszrate, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='r')
    ax.plot([offset-0.2,offset+0.2],[np.mean(preszrate),np.mean(preszrate)],'k-')
    ax.plot([offset-0.1,offset+0.1],[np.mean(preszrate)+ss.sem(preszrate),np.mean(preszrate)+ss.sem(preszrate)],'k-')
    ax.plot([offset-0.1,offset+0.1],[np.mean(preszrate)-ss.sem(preszrate),np.mean(preszrate)-ss.sem(preszrate)],'k-')
    ax.plot([offset,offset],[np.mean(preszrate)-ss.sem(preszrate),np.mean(preszrate)+ss.sem(preszrate)],'k-')
    
    offset+=1
    
    x_jitter2 = pd.Series(np.random.normal(loc=0, scale=jitter, size=szrate.shape))+offset
    ax.plot(x_jitter2, szrate, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='b')
    ax.plot([offset-0.2,offset+0.2],[np.mean(szrate),np.mean(szrate)],'k-')
    ax.plot([offset-0.1,offset+0.1],[np.mean(szrate)+ss.sem(szrate),np.mean(szrate)+ss.sem(szrate)],'k-')
    ax.plot([offset-0.1,offset+0.1],[np.mean(szrate)-ss.sem(szrate),np.mean(szrate)-ss.sem(szrate)],'k-')
    ax.plot([offset,offset],[np.mean(szrate)-ss.sem(szrate),np.mean(szrate)+ss.sem(szrate)],'k-')


    for pattern in patterns:
        GrpFR = TransFR.loc[TransFR.Classes == pattern]
        SzStartRawFR = GrpFR.Start
        SzStartRawFR = np.array(SzStartRawFR.tolist()) # These are just some shenanigans to get values into numpy array
        
        SzEndRawFR = GrpFR.End
        SzEndRawFR = np.array(SzEndRawFR.tolist()) # These are just some shenanigans to get values into numpy array
    
#        yvals = np.nanmean(SzStartRawFR[:,0:int((bincount+szrange)/steps)],axis=0)
#        yerr = ss.sem(SzStartRawFR[:,0:int((bincount+szrange)/steps)],axis=0,nan_policy='omit')
#        plt.figure()
#        plt.plot(xvalson,yvals,color='b')
#        plt.fill_between(xvalson,yvals-yerr,yvals+yerr,color='lightblue')
#        plt.axvline(x=zeroon,color='r');
##        plt.ylim([np.nanmin(yvals-yerr)-2,np.nanmax(yvals+yerr)+2])
#        
#        yvals = np.nanmean(SzEndRawFR[:,-int((bincount+szrange)/steps):],axis=0)
#        yerr = ss.sem(SzEndRawFR[:,-int((bincount+szrange)/steps):],axis=0,nan_policy='omit')
#        plt.plot(xvalsoff,yvals,color='b')
#        plt.fill_between(xvalsoff,yvals-yerr,yvals+yerr,color='lightblue')
#        plt.axvline(x=zerooff,color='r')
#        plt.ylabel('Firing Rate (Hz)')
#        plt.xlabel('Time from Seizure End (s)')
#        plt.title([location + pattern+ 'by seiz'])
#        plt.xticks(xticklocs,xtickvals)
#
        CellStartRawFR = np.empty([1,np.size(xvalson,0)])
        CellEndRawFR = np.empty([1,np.size(xvalson,0)])
        CellStartNormFR = np.empty([1,np.size(xvalson,0)])
        CellEndNormFR = np.empty([1,np.size(xvalson,0)])
    
        for cell in GrpFR.Cell.unique():
            oncellfr = np.nanmean(np.array(SzTransFiringRates.loc[SzTransFiringRates.Cell==cell].Start.tolist())[:,0:int((bincount+szrange)/steps)],axis=0)
#            oncellfr = np.nanmean(np.array(SzTransISIcov.loc[SzTransISIcov.Cell==cell].Start.tolist())[:,0:int((bincount+szrange)/steps)],axis=0)
#            oncellfr[oncellfr==0]= np.nan # Get rid of zeros because they will become infinite when reciprocated
#            oncellfr = 1/(oncellfr) # We define rhythmicity as the reciprocal of ISI coefficient of variation
            CellStartRawFR=np.vstack([CellStartRawFR, oncellfr])
            offcellfr = np.nanmean(np.array(SzTransFiringRates.loc[SzTransFiringRates.Cell==cell].End.tolist())[:,-int((bincount+szrange)/steps):],axis=0)
#            offcellfr = np.nanmean(np.array(SzTransISIcov.loc[SzTransISIcov.Cell==cell].End.tolist())[:,-int((bincount+szrange)/steps):],axis=0)
#            offcellfr[offcellfr==0]= np.nan
#            offcellfr = 1/(offcellfr)
            CellEndRawFR=np.vstack([CellEndRawFR,offcellfr])
            CellStartNormFR=np.vstack([CellStartNormFR, oncellfr/np.nanmax(np.abs(np.append(oncellfr,offcellfr)),axis=0)])
            CellEndNormFR = np.vstack([CellEndNormFR, offcellfr/np.nanmax(np.abs(np.append(oncellfr,offcellfr)),axis=0)])
            
        CellStartRawFR = np.delete(CellStartRawFR, (0), axis=0)    
        CellEndRawFR = np.delete(CellEndRawFR, (0), axis=0)    
        CellStartNormFR = np.delete(CellStartNormFR, (0), axis=0)    
        CellEndNormFR = np.delete(CellEndNormFR, (0), axis=0)    
        
        # This next bit is to calculate isi coefficients of variation in the first second vs entire seizure
#        firstsecrhythm = 1-np.nanmean(CellStartNormFR[:,int(bincount/steps):int((bincount+1)/steps)],axis=1)
#        fullseizrhythm = 1-np.nanmean(CellStartNormFR[:,int(bincount/steps):],axis=1)
#        w,p = ss.wilcoxon(firstsecrhythm,fullseizrhythm)
#        data = pd.melt(pd.DataFrame({"First Second":firstsecrhythm, "Whole Seizure":fullseizrhythm}), var_name = 'Time', value_name = 'Rhythmicity')
        
#        plt.figure()
#        sns.swarmplot(data = data, x = 'Time', y = 'Rhythmicity', hue = 'Time')
#        sns.boxplot(data=data, x='Time', y = 'Rhythmicity')
#        plt.title([location + ' ' + pattern])
#        plt.text(1, 0.05, ['p = '+ str(p)])
#        
#        yvals = np.nanmean(CellStartNormFR[:,0:int((bincount+szrange)/steps)],axis=0)
#        yerr = ss.sem(CellStartNormFR[:,0:int((bincount+szrange)/steps)],axis=0,nan_policy='omit')
#        plt.figure()
#        plt.plot(xvalson,yvals,color='b')
#        plt.fill_between(xvalson,yvals-yerr,yvals+yerr,color='lightblue')
#        plt.axvline(x=zeroon,color='r')
#        
#        yvals = np.nanmean(CellEndNormFR[:,-int((bincount+szrange)/steps):],axis=0)
#        yerr = ss.sem(CellEndNormFR[:,-int((bincount+szrange)/steps):],axis=0,nan_policy='omit')
#        plt.plot(xvalsoff,yvals,color='b')
#        plt.fill_between(xvalsoff,yvals-yerr,yvals+yerr,color='lightblue')
#        plt.axvline(x=zerooff,color='r')
#        plt.ylabel('Firing Rate (Proportion of Cell Max)')
#        plt.xlabel('Time from Seizure Start (s)')
#        plt.title([location+pattern+'normalized by cell'])
#        plt.xticks(xticklocs,xtickvals)
#        plt.ylim([0,1])
#        plt.xlim([xvalson[0],xvalsoff[-1]+1])

    # Testing correlation coefficients between 1 min pre-seizure firing rates and time
#        preYs = np.nanmean(CellStartNormFR[:,int(slopewin[0]/steps):int(slopewin[1]/steps)],axis=0)
#        r,p = ss.pearsonr(xvals[int(slopewin[0]/steps):int(slopewin[1]/steps)],preYs)
#        
#        print(location + ' ' + pattern + ' -80 to -10 R = ' + 
#          str(np.around(r,decimals=3)) + ', p = ' +
#          str(p))
#
#        
#        yvals = np.nanmean(CellStartRawFR[:,0:int((bincount+szrange)/steps)],axis=0)
#        yerr = ss.sem(CellStartRawFR[:,0:int((bincount+szrange)/steps)],axis=0,nan_policy='omit')
#        plt.figure()
#        plt.plot(xvalson,yvals,color='b')
#        plt.fill_between(xvalson,yvals-yerr,yvals+yerr,color='lightblue')
#        plt.axvline(x=zeroon,color='r')
##        plt.ylim([np.nanmin(yvals-yerr)-2,np.nanmax(yvals+yerr)+2])
#        
#        yvals = np.nanmean(CellEndRawFR[:,-int((bincount+szrange)/steps):],axis=0)
#        yerr = ss.sem(CellEndRawFR[:,-int((bincount+szrange)/steps):],axis=0,nan_policy='omit')
#        plt.plot(xvalsoff,yvals,color='b')
#        plt.fill_between(xvalsoff,yvals-yerr,yvals+yerr,color='lightblue')
#        plt.axvline(x=zerooff,color='r')
#        plt.ylabel(' Firing Rate')
#        plt.xlabel('Time from Seizure End (s)')
#        plt.title([location + pattern + 'by cell'])
#        plt.xticks(xticklocs,xtickvals)

        
        # Calculating mean firing rates up to 5 s pre-seizure (transitions only, not state) for cell-wise data
        preszrate = np.nanmean(CellStartRawFR[:,0:int((bincount-5)/steps)],axis=1) # From start of data to 5s before seizure start time
        szrate = np.nanmean(CellStartRawFR[:,int(bincount/steps):int((bincount+szrange)/steps)],axis=1)
        preszsem = ss.sem(preszrate,nan_policy='omit')
        szsem = ss.sem(szrate,nan_policy='omit')
    
        w,p = ss.ttest_rel(preszrate,szrate,nan_policy='omit')
        p = p*4 #Bonferroni correction
        print(location + ' ' + pattern + ' baseline = ' + 
              str(np.around(np.nanmean(preszrate),decimals=3)) + ', SEM = ' +
              str(np.around(preszsem,decimals=3)))
        print(location + ' ' + pattern + ' seizure = ' + 
              str(np.around(np.nanmean(szrate),decimals=3)) + ', SEM = ' +
              str(np.around(szsem,decimals=3)))
        print(location + ' ' + pattern + ' base v seiz p = ' +
            str(p))

        offset+=1

        x_jitter1 = pd.Series(np.random.normal(loc=0, scale=jitter, size=preszrate.shape))+offset
        ax.plot(x_jitter1, preszrate, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='r')
        ax.plot([offset-0.2,offset+0.2],[np.mean(preszrate),np.mean(preszrate)],'k-')
        ax.plot([offset-0.1,offset+0.1],[np.mean(preszrate)+ss.sem(preszrate),np.mean(preszrate)+ss.sem(preszrate)],'k-')
        ax.plot([offset-0.1,offset+0.1],[np.mean(preszrate)-ss.sem(preszrate),np.mean(preszrate)-ss.sem(preszrate)],'k-')
        ax.plot([offset,offset],[np.mean(preszrate)-ss.sem(preszrate),np.mean(preszrate)+ss.sem(preszrate)],'k-')
        
        offset+=1
        
        x_jitter2 = pd.Series(np.random.normal(loc=0, scale=jitter, size=szrate.shape))+offset
        ax.plot(x_jitter2, szrate, 'o', alpha=.40, zorder=1, ms=8, mew=1, color='b')
        ax.plot([offset-0.2,offset+0.2],[np.mean(szrate),np.mean(szrate)],'k-')
        ax.plot([offset-0.1,offset+0.1],[np.mean(szrate)+ss.sem(szrate),np.mean(szrate)+ss.sem(szrate)],'k-')
        ax.plot([offset-0.1,offset+0.1],[np.mean(szrate)-ss.sem(szrate),np.mean(szrate)-ss.sem(szrate)],'k-')
        ax.plot([offset,offset],[np.mean(szrate)-ss.sem(szrate),np.mean(szrate)+ss.sem(szrate)],'k-')
        
    ax.set_xticks(np.arange(0,offset+1)[0::2]+0.5)
    ax.set_xticklabels(np.insert(patterns,0,'All Cells',axis=0))

