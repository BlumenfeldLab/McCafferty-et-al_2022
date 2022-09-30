# -*- coding: utf-8 -*-
"""
29/04/2019 - Peter Vincent
This python file contains class definitions for the cell and seizure objects
that are used to store data after pre-processing and extraction

"""

import random, string
import pickle as pk
import re

class SWD:
    Type = 'Seizure'
    def __init__(self,data_file='',seizure_number=(),seiz_bounds=(),seiz_label='None'):
        self.data_file = data_file
        self.number = seizure_number  
        self.hash_key = gen_hash()
        self.recording_id = get_recording_id(data_file,self.Type)
        self.Rat = get_rat_name(self.recording_id)
        self.start = seiz_bounds[0]
        self.end = seiz_bounds[1]
        self.label = seiz_label
        self.properties = dict()

        
        
        
    # Enter methods below, if they become usefull
    def get_data(self):
        seiz_data_loc = self.data_file
        seiz_number   = self.number
        with open(seiz_data_loc, 'rb') as f:
            seiz_dict = pk.load(f)
            seiz_data = seiz_dict[seiz_number]
            return seiz_data

    def __del__(self):
        pass
        
        
    
    
class Cell:
    Type = 'Cell'
    def __init__(self,data_file='',channel='',recording_id='',cell_bounds=(),
                 cell_data=(),cell_label=''):
        self.data_file = data_file
        self.number  = channel
        self.hash_key = gen_hash()
        self.recording_id = recording_id
        self.Rat = get_rat_name(self.recording_id)
        self.start = cell_bounds[0]
        self.end = cell_bounds[1]
        self.cell_data = cell_data
        self.label = cell_label
        self.properties = dict()
        
    def __del__(self):
        pass

        
        
def gen_hash():
    # This function assumes the database is stored at
    # /mnt/Data4/GAERS_Data/GAERS_Objects.pkl
    hash_key = ''.join(random.choice(string.ascii_uppercase+
                                         string.ascii_lowercase+
                                         string.digits) for _ in range(16))
    # Check the dataframe to make sure no other files have this hash key
    return hash_key

def get_recording_id(data_file,class_type):
    if class_type == 'Seizure':
        folder_markers = [pos for pos, 
                          char in enumerate(data_file) if char == '/']
        final_marker = folder_markers[-1]
        penultimate_marker = folder_markers[-2] + 1
        file_name = data_file[penultimate_marker:final_marker]
        recording_id = file_name
        return recording_id
    elif class_type == 'Cell':
        pass
    return 0
        

def get_rat_name(recording_id):
    parent_dir = '/mnt/Data4/AnnotateSWD/'
    text_file  = 'AnimalLogFile.txt'
    total_path = parent_dir + text_file
    with open(total_path,'r') as log:
        for line in log:
            if recording_id in line:
                animal = line.replace(recording_id + ' ','')
                if 'Error' in animal:
                    report = 'Error'
                elif 'Sil' in  animal:
                    report = animal
                    break
                else:
                    report = animal
                    break
    report = report.strip()
    return report

def save_object(object_inst,save_dir='/mnt/Data4/GAERS_Data/'):
    name_pre = object_inst.__class__.__name__
    pattern  = re.compile('[\W_]+')
    name_body= pattern.sub('',object_inst.recording_id)
    number   = object_inst.number
    full_name= name_pre + '_' + name_body + str(number) + '.pkl'
    if name_pre == "Cell":
        full_name = full_name.replace('.pkl',('_' + str(object_inst.start)))
        full_name = full_name + '.pkl'
    s = open(save_dir + full_name,'wb')
    pk.dump(object_inst,s)
    s.close()
    
