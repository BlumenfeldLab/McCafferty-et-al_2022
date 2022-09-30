#blu# -*- coding: utf-8 -*-
"""
30/04/2019 - Peter Vincent
This Python file establishes the pandas database that logs and stores all the
information that must be preserved when handling the different cell and seizure
objects.

"""
from os.path import expanduser
home = expanduser('~')

import pandas as pd
import os
import pickle as pk
import class_definitions as cd

# The code commented below is the original single line used to make the 
# dataframe.  It is now commented out since we should have no need to ever 
# run it again.  The dataframe was made on the 02/05/2019 (British convention)
# by Peter Vincent
# GAERS_Objects = pd.DataFrame(columns=['hash_key','recording_id','Type','Rat',
#                                'start','end'])

#---#---#---#---#---#---#

# The folowing code include tools to work with the dataframe

def load_in_dataframe(dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    exist = os.path.isfile(dataframe)
    if exist:
        if '.pkl' in dataframe:
            database = open(dataframe,'rb')
            database = pk.load(database)
            return database
        else:
            raise ValueError('Provided dataframe name : ' + dataframe + '''
                             is not a .pkl file.  Either provide a .pkl file
                             or use a different function!''')
    else:
        raise ValueError('Provided dataframe name: ' + dataframe + ''' does not
                         exist.  Please supply a valid name''')
    
def save_dataframe(dataframe_var, 
                   save_dir = home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    dataframe_var.to_pickle(save_dir)
    
def remove_erroneous_cells(dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl',lookup=home+'/mnt/Data4/GAERS_Data/Cell_Lookup.pkl'):
    database = load_in_dataframe(dataframe)
    with open(lookup,'r') as p:
        matching_dict = pk.load(p)
    
    valid_rec = []
    too_few = []
    no_cells = []
    too_many = []
    for key, value in matching_dict.items():
        small_df = extract_dataframe(database,recording_id=key,Type='Cell')
        num_cells = len(small_df)
        if num_cells == value:
            valid_rec.append(key)
        elif num_cells < value:
            if num_cells == 0:
                no_cells.append(key)
            else:
                too_few.append(key)
            remove_objects('Cell',key,dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl')
            remove_objects('Seizure',key,dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl')
        elif num_cells > value:
            too_many.append(key)
            remove_objects('Cell',key,dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl')
            remove_objects('Seizure',key,dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl')
    
    clean_dataframe()
    return valid_rec,too_few,no_cells,too_many

def clean_dataframe(dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl',
                    dataLocation=home+'/mnt/Data4/GAERS_Data/'):
    database = load_in_dataframe(dataframe)
    database = database.reset_index()
    database = database.drop(columns='index')
    to_delete_name = []
    to_delete_index = []
    num_objects = len(database)
    for cur_object in range(num_objects):
        cur_object_meta = database.iloc[cur_object]
        object_name = cur_object_meta.Name
        if not isinstance(object_name,str):
            to_delete_index.append(cur_object)
            continue
        full_path = dataLocation + object_name
        if not os.path.isfile(full_path):
            to_delete_name.append(object_name)
            to_delete_index.append(cur_object)
    database = database.drop(to_delete_index,axis=0)
    database = database.reset_index()
    database = database.drop(columns='index')
    
    save_dataframe(database,dataframe)
    return database
    
    
def dataframe_pkl_to_csv(dataframe= home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    database = load_in_dataframe(dataframe)
    csv_name = dataframe.replace('.pkl','.csv')
    database.to_csv(csv_name)
    


def add_column(column, dataframe= home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    database = load_in_dataframe(dataframe)
    if type(column) is not list:
        column = [column]
    num_columns = len(column)
    for col in range(num_columns):
        cur_col = column[col]
        if cur_col in database:
            print(cur_col + ' is already in ' + database)
        else:
            database[cur_col] = list()
    database.to_pickle(dataframe)


def update_dataframe(dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl',
                     obj_store=home+'/mnt/Data4/GAERS_Data/',
                     classes=['Cell', 'SWD']):
    obj_store_files = os.listdir(obj_store)
    obj_database = load_in_dataframe(dataframe)
    check_unique_names = list(obj_database.Name)
    obj_dict = obj_database.to_dict()
    new_dict = []
    for class_type in  classes:
        valid_classes = [inst for inst in obj_store_files if class_type in inst]
        for instance in valid_classes:
            cur_dict = {}
            dict_valid = True
            if instance not in check_unique_names:
                with open(obj_store + instance,'rb') as p:
                    cur_inst = pk.load(p)
                for key in obj_dict:
                    if key == "Name":
                        cur_dict[key] = instance
                    else:
                        try:
                            cur_dict[key] = getattr(cur_inst,key)
                        except:
                            dict_valid = False
            if dict_valid:
                new_dict.append(cur_dict)
    
    obj_database = obj_database.append(new_dict,ignore_index=False)
    
    save_dataframe(obj_database,dataframe)
    
    return obj_database


def update_dataframe_sztms(dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects_Old.pkl',
                           indir=home+'/mnt/Data4/AnnotateSWD/'):
    '''
    Updates the SWDs with new sztms. 
    '''
    data = load_in_dataframe(dataframe)
    data = data.loc[:,['recording_id','Type','Rat','start','end','Name','number','Duration','label']]
    new_data = data[data['Type']=='Cell']
    
    rec_ids = list(set(data['recording_id'].to_list()))
    for sess in rec_ids:
        rat = cd.get_rat_name(sess)
        this_indir = indir + sess + '/'
        for f in os.listdir(this_indir):
            if f.endswith('sztms.txt'):
                sztms = os.path.join(this_indir, f)
                break
        
        with open(sztms,'rb') as f:
            times = f.read().split()
        
        try: 
            int(times[0])
        except ValueError:
            times = times[4:]
            
        seiz_num = len(times)/4       
        valid_seiz_counter = 0
        for seiz in range(seiz_num): # numbering starts from 0
            start_ind = seiz*4 + 1;
            end_ind   = seiz*4 + 2;
            start_val = float(times[start_ind])
            end_val   = float(times[end_ind])
            start_val = int(start_val*1000) # Convert seconds into miliseconds
            end_val = int(end_val*1000)     # Convert seconds into miliseconds
            seiz_dur = len(range(start_val, end_val))
            seiz_name = sess.replace('-','')
            seiz_name = seiz_name.replace('_','')
            
            if seiz_dur > 999:
                seiz_name = 'SWD_' + str(seiz_name) + str(valid_seiz_counter) + '.pkl'
                new_data = new_data.append({'recording_id':sess, 
                                            'Type':'Seizure', 
                                            'Rat':rat, 
                                            'start':start_val, 
                                            'end':end_val, 
                                            'Name':seiz_name,
                                            'number':valid_seiz_counter,
                                            'Duration':seiz_dur,
                                            'label':None}, ignore_index=True)
                valid_seiz_counter += 1
            
            
    return new_data
                    
            
    

def clear_dataframe(dataframe = home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    database = load_in_dataframe()
    database.drop(database.index, inplace=True)
    save_dataframe(database,dataframe)
    

def delete_column(column, dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    database = load_in_dataframe(dataframe)
    if type(column) is not list:
        column = [column]
    num_columns = len(column)
    for col in range(num_columns):
        cur_col = column[col]
        if cur_col in database:
            database = database.drop(columns=cur_col)
        else:
            print(cur_col + ' does not exist in ' + dataframe)
    database.to_pickle(dataframe)
    
def add_object(object_inst,dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    database = load_in_dataframe(dataframe)
    headers  = list(database.columns.values)
    pass
    # num_indices = 
    # for header = headers:
        
def remove_duplicates(dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    database = load_in_dataframe(dataframe)
    database = database.reset_index()
    database = database.drop(columns='index')
    # This is based off start and end times, since these are the most 
    # "unique" feature of any of the objects
    duplicate_sets = []
    for instance in range(database.shape[0]):
        cur_instance = database.iloc[instance]
        cur_type = cur_instance.Type
        cur_start= cur_instance.start
        cur_end  = cur_instance.end
        same_type= database.loc[database["Type"] == cur_type]
        same_start=same_type.loc[same_type["start"] == cur_start]
        same_end  = same_start.loc[same_start["end"] == cur_end]
        if len(same_end) > 1:
            duplicate_index = list(same_end.index.values)
            duplicate_sets.append(duplicate_index)
    to_delete = []
    for duplicates in range(len(duplicate_sets)):
        cur_set = duplicate_sets[duplicates]
        cur_set.sort
        set_to_delete = cur_set[:-1]
        to_delete.append(set_to_delete)
    to_delete = [item for sublist in to_delete for item in sublist]
    to_delete_index = list(set(to_delete))
    database = database.drop(to_delete_index,axis=0)
    database = database.reset_index()
    database = database.drop(columns='index')
    
    save_dataframe(database,dataframe)
    
    return database

def remove_objects(obj_type,recording,dataframe=home+'/mnt/Data4/GAERS_Data/GAERS_Objects.pkl'):
    # Funcation removes objects of a certain type ("Seizure" or "Cell")
    # and specified recording ID
    database = load_in_dataframe(dataframe)
    database = database.reset_index()
    database = database.drop(columns='index')
    
    same_type = database.loc[database["Type"] == obj_type]
    same_id = same_type.loc[same_type["recording_id"] == recording]
    
    remove_idx = same_id.index.tolist()
    database = database.drop(remove_idx)
    database = database.reset_index()
    database = database.drop(columns='index')
    
    save_dataframe(database,dataframe)
    
    return database
   
#%% To remove sessions with no seizures and/or no cells from working database
        
def clean_liveDB(to_clean):

    cleaned = to_clean # create new database for output
    sess_ids = to_clean['recording_id'].unique() # find all unique session/recording IDs
    
    for idx, sess_id in enumerate(sess_ids): # Iterate through unique recording IDs
        database_sess = to_clean.loc[to_clean['recording_id'] == sess_id] # Create mini-database for this session
        if not 'Seizure' in database_sess.values or not 'Cell' in database_sess.values:
            # if it has either no seizure or no cell objects
            cleaned = cleaned[cleaned.recording_id != sess_id] # remove that session from the database
    
    return cleaned

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

def fix_df_index(dataframe):
    dataframe = dataframe.reset_index()
    dataframe = dataframe.drop(columns=['index'])
    return dataframe
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        