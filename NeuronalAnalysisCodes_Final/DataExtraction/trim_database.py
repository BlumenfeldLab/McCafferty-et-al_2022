#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 10:37:16 2020

@author: rjt37
"""

import pandas as pd
import warnings
import numpy as np

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

#%%
def remove_from_dataframe(dataframe,**kwargs):
    """this function takes in a general dataframe returns a
    new dataframe with inputted rows removed."""
    if dataframe.empty:
        raise Exception('Dataframe argument is empty')
    for name, value in kwargs.items():
        dataframe = dataframe[dataframe[name]!=value]
    dataframe = fix_df_index(dataframe)
    return dataframe

#%%
def pull_select_dataframe(dataframe, names):
    """this function takes in a general dataframe of recording data and returns a
    new dataframe that only contains data for the objects named in the list"""
    new_dataframe = pd.DataFrame()
    for name in names:
        this_dataframe = extract_dataframe(dataframe, Name = name)
        new_dataframe = pd.concat([new_dataframe,this_dataframe],)
    new_dataframe = fix_df_index(new_dataframe)
    return new_dataframe   

#%%
def pull_select_recid_dataframe(dataframe, recids):
    """this function takes in a general dataframe of recording data and returns a
    new dataframe that only contains data for the recordings named in the list"""
    new_dataframe = pd.DataFrame()
    for recid in recids:
        this_dataframe = extract_dataframe(dataframe, recording_id = recid)
        new_dataframe = pd.concat([new_dataframe,this_dataframe],)
    new_dataframe = fix_df_index(new_dataframe)
    return new_dataframe

#%%
def pull_select_number_dataframe(dataframe, numbers):
    """this function takes in a general dataframe of recording data and returns a
    new dataframe that only contains data for the recordings named in the list"""
    new_dataframe = pd.DataFrame()
    for number in numbers:
        this_dataframe = extract_dataframe(dataframe, number = number)
        new_dataframe = pd.concat([new_dataframe,this_dataframe],)
    new_dataframe = fix_df_index(new_dataframe)
    return new_dataframe

#%%
def pull_seiz_celltype(dataframe, celltype = 'Cortical'):
    '''useful if you input a seizure dataframe; will output only seizures from 
    cortical recordings or thalamic recordings depending on celltype argument'''
    from itertools import compress
#    dataframe = extract_dataframe(dataframe, Type = 'Seizure')
    new_dataframe = pd.DataFrame()
    rat_names = list(dataframe['Rat'].unique())
    type_log = np.zeros(len(rat_names))
    for i in range(len(rat_names)):
        type_log[i] = rat_names[i].startswith('Sil') #1 for cortical cells
    type_log = type_log == 1
    if celltype == 'Cortical':
        rat_idx = list(compress(range(len(type_log)), type_log))
    elif celltype == 'Thalamic':
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
    '''this function takes in 2 dataframes and spits out the rows of 
    df1 that overlap with rows of df2. Note: if inverse is true, it spits out the 
    rows of df1 that don't overlap with rows of df2'''
    if not inverse:
        overlap = df1[df1[column].isin(df2[column])]
    else:
        overlap = df1[~df1[column].isin(df2[column])]
    return overlap

#%%
def fix_df_index(dataframe):
    '''makes the indices of the dataframe 0 to n in order'''
    dataframe = dataframe.reset_index()
    dataframe = dataframe.drop(columns=['index'])
    return dataframe


