# -*- coding: utf-8 -*-
"""
26/04/2019 - Peter Vincent
This Python file contains a variety of utilities used to extract data from the 
GAERS project.  Notably, this script contains functions to extract seizure 
data and cell data from the relevant files following other steps in the 
pre-processing
ani
"""
import locate_files as lf
import numpy as np
import struct as st
import pickle
import os 
import class_definitions as ccd
import pandas as pd
import warnings
import build_database as bd
import subprocess
from scipy.io import loadmat

def batch_seizures(target_directory = '/mnt/Data4/AnnotateSWD/',
                   object_save_directory = '/mnt/Data4/GAERS_Data/',
                   datType = 'oct',
                   dataframe_name = '/GAERS_Objects.pkl'):
    # This function is run after the bash extraction.  It proceeds through the 
    # files in "target_directory" and executes the gen_seizure_objects function .
    # on each case and saves the resultant seizure classes in
    # "object_save_directory"
    dataframe = object_save_directory + dataframe_name
    if '//' in dataframe:
        dataframe = dataframe.replace('//','/')
    # Generate folders/files that should be ignored in the target_directory
    database = bd.load_in_dataframe(dataframe)
    database = database.loc[database['Type'] == 'Seizure']
    processed_runs = database[['recording_id']]
    processed_runs = processed_runs.values.tolist()
    processed_runs = [item for sublist in processed_runs for item in sublist]
    files_to_reject= ['AnimalLogFile.txt', 'SWD and Spikes annotation.xlsx']
    target_dirs = os.listdir(target_directory)
    failed_dirs = list()
    successful_dirs = list()
    for target in target_dirs:
        if "Null" in target:
            continue
        if target in processed_runs:
            continue
        if target in files_to_reject:
            continue
        # assuming tfolder shouldn't be skipped, proceed to process it
        fulldir_target = target_directory + target
        # Check that target is a directory, not a file
        if not os.path.isdir(fulldir_target):
            continue
        if os.path.isdir(fulldir_target):
            try:
                gen_seizure_objects(target, datType, save=1, 
                                    dataDir=object_save_directory)
                successful_dirs.append(target)
            except:
                failed_dirs.append(target)
                print("Seizure extraction on " + target + ''' has failed.
                     Try execution of subroutines manually (i.e., not in this
                     batch) to diagnose the problem''')
    
    output_dict = {'Succssful':successful_dirs,
                'Failed':failed_dirs}
    new_dataframe = bd.update_dataframe(dataframe=dataframe,
                                           obj_store=object_save_directory)
    clean = bd.clean_dataframe(dataframe=dataframe,
                               dataLocation=object_save_directory)
    
    return output_dict

def batch_cells(target_directory = '/mnt/Data4/AnnotateSWD/',
                object_save_directory = '/mnt/Data4/GAERS_Data/', 
                dataframe_name = '/GAERS_Objects.pkl',verbose=0,save=1,
                files=['.clu.','.res.'],downsample=30,cutoff=1,
                spkFile='.spk.',max_clusters=15):
    # This function is run after the bash extraction.  It proceeds through the 
    # files in "target_directory" and executes the gen_cell_objects function 
    # on each case and saves the resultant cell classes in
    # "object_save_directory"
    
    # For thalamic cells, use ['.mat'] as the files argument
    
    dataframe = object_save_directory + dataframe_name
    if '//' in dataframe:
        dataframe = dataframe.replace('//','/')
    # Generate a list of files/folders we want to ignore
    database = bd.load_in_dataframe(dataframe)
    database = database.loc[database['Type'] == 'Cell']
    processed_runs = database[['recording_id']]
    processed_runs = processed_runs.values.tolist()
    processed_runs = [item for sublist in processed_runs for item in sublist]
    files_to_reject= 'AnimalLogFile.txt'
    target_dirs = os.listdir(target_directory)
    failed_dirs = list()
    successful_dirs = list()
    for target in target_dirs:
        if "Null" in target:
            continue
        if target in processed_runs:
            continue
        if target in files_to_reject:
            continue
        fulldir_target = target_directory + target
        if os.path.isdir(fulldir_target):
            print(target)
            try:
                gen_cell_objects(target,verbose=verbose,files=files,save=save,
                     dataDir=object_save_directory,downsample=downsample,
                     cutoff=cutoff,refDir =target_directory,spkFile=spkFile,
                     max_clusters=max_clusters)
                print('Extracting cell data...')
                successful_dirs.append(target)
            except:
                failed_dirs.append(target)
                print("Cell extraction on " + target + ''' has failed.
                     Try execution of subroutines manually (i.e., not in this
                     batch) to diagnose the problem''')
    output_dict = {'Successful':successful_dirs,
                'Failed':failed_dirs}
    new_dataframe = bd.update_dataframe(dataframe=dataframe,
                                           obj_store=object_save_directory)
    clean = bd.clean_dataframe(dataframe=dataframe,
                               dataLocation=object_save_directory)
    
    return output_dict
    

def extract_seizures(session,datType='oct',save = 0,seiz_times_id='sztms.txt'):
    # This function takes in a session as argument are parses and looks for 
    # whether the seizure times and dat_type files are available for processing
    # if no argument is supplied for dat_type, this function will default to 
    # look for a oct file.
    
    # Get the paths of the files we're interested in
    targets = lf.locate_bool(session, datType,seiz_times_id=seiz_times_id)
    
    if targets[0] ==0 or targets[1] == 0:
        raise Exception(' Data is missing from ' + session + '''.   Either 
                        seiztimes does not exist, or ''' + datType + ''' does 
                        not exist''')
    seiz_times_loc = targets[2]
    data_loc       = targets[3]
    # Confirm the data_loc doesn't have anything dodgy
    
    channel_num    = len(data_loc)
    save_dir       = seiz_times_loc.replace(seiz_times_id,'')
    
    # Read in the text file containing the seizure times
    with open(seiz_times_loc,'r') as f:
        times = f.read().split()
    
    try: 
        int(times[0])
    except ValueError:
        times = times[4:]

    seiz_num = len(times)/4 # The text file has 4 columns, so we need to jump 
                            # in fours since the python list when it is loaded
                            # in is just flat
    seiz_list = list()
    starts = list()
    ends = list()
    #seiz_times = np.zeros([seiz_num, 2])
    for seiz in range(seiz_num):
        start_ind = seiz*4 + 1;
        end_ind   = seiz*4 + 2;
        start_val = float(times[start_ind])
        end_val   = float(times[end_ind])
        start_val = int(start_val*1000) # Convert seconds into miliseconds
        end_val = int(end_val*1000)     # Convert seconds into miliseconds
        seiz_dur = len(range(start_val, end_val))
        if seiz_dur > 999:
            starts.append(start_val)
            ends.append(end_val)
            #seiz_times[seiz, 0] = start_val
            #seiz_times[seiz, 1] = end_val
            seiz_data = np.zeros([channel_num, seiz_dur])
            chan_num = 0
            for cur_chan in  data_loc:
                with open(cur_chan,'rb') as b:
                    samp_num = 0
                    val = start_val
                    bin_val = val * 2 # Find the start of the seizure
                    b.seek(bin_val,0)
                    for val in range(start_val,end_val):
                        value = st.unpack('h', b.read(2)) # Read in the data
                        value = int(value[0])             # The pointer moves
                        if len(session) < 10:
                            value *= 1000
                        else:
                            value *= 0.195
                        seiz_data[chan_num, samp_num] = value
                        samp_num += 1
                    chan_num += 1
            
                    
            seiz_list.append(seiz_data)
    
    seiz_times = np.transpose(np.array([starts,ends])).astype('float64')    
    output_dict = {'Times': seiz_times, 'Data': seiz_list}
    save_q = bool(save)
    if save_q:
        s = open(save_dir + datType + "_run_seizures.pkl",'wb')
        pickle.dump(output_dict, s)
        s.close()
        

    return {'Times': seiz_times, 'Data': seiz_list}

def gen_seizure_list(session, datType='oct', save = 1,
                     seiz_times_id='sztms.txt'):
    # This function extracts and saves seizures in an appropriate structure for
    # use in the ts-fresh machine package, saved in a .pkl file named
    # seizure_list
    targets = lf.locate_bool(session, datType, seiz_times_id='sztms.txt')
    if targets[0] ==0 or targets[1] == 0:
        raise Exception(' Data is missing from ' + session + '''.   Either 
                        seiztimes does not exist, or ''' + datType + ''' does 
                        not exist''')
    seiz_times_loc = targets[2]
    save_dir       = seiz_times_loc.replace(seiz_times_id,'')
    search_file    = save_dir + datType + '_run_seizures.pkl'
    exists = os.path.isfile(search_file)
    save_q = bool(save)
    if exists:
        with open(search_file, 'rb') as f:
            seiz_dict = pickle.load(f)
        
    else:
        seiz_dict = extract_seizures(session,datType,save=save)
    seiz_list = seiz_dict['Data']
    if save_q:
        s = open(save_dir + datType + '_seizure_list.pkl','wb')
        pickle.dump(seiz_list,s)
        s.close()
    
    return seiz_list

def gen_seizure_objects(session, datType='oct', save=1, 
                        dataDir='/mnt/Data4/GAERS_Data/',
                        seiz_times_id='sztms.txt',
                        seiz_label_id='1_szlabels.txt'):
    # This function looks for a file in the given seizure directory.  If the 
    # seizure file exists, it loads it into memory, after which it proceeds to
    # create an object with that information.  The object is given a unique 
    # filename and saved to the data directory.  
    
    # Get the information for a given session's seizures
    time_mat, seiz_location, seiz_labels = extract_seizure_times_labels(session,0,seiz_times_id,seiz_label_id)
    search_dir = seiz_location.replace(seiz_times_id,'')
    search_file = search_dir + datType + '_seizure_list.pkl'
    exist = os.path.isfile(search_file)
    gen_seizure_list(session, datType, 1)

    num_seiz = len(time_mat)
    save_q = bool(save)
    for seiz in range(num_seiz):
        seiz_start = time_mat[seiz, 0]
        seiz_end   = time_mat[seiz, 1]
        seiz_start = (seiz_start)
        seiz_end   = (seiz_end)
        seiz_tuple = (seiz_start, seiz_end)
        cur_seiz_obj = ccd.SWD(search_file, seiz, seiz_tuple, seiz_labels[seiz])
        if save_q:
            ccd.save_object(cur_seiz_obj)
    
    return 0
    

def extract_seizure_times_labels(session,save = 0,seiz_times_id='sztms.txt',seiz_label_id='4_szlabels.txt'):
    # A section from the above code on extract_seizures, which generates a 
    # matrix of seizure times
    # Check for seizure times file
    time_targets = lf.locate_bool(session,'null',seiz_times_id='sztms.txt')
    if time_targets[0] == 0:
        raise Exception('Seizure times not yet extracted')
    
    # Check for seizure labels file
    label_targets = lf.locate_bool(session,'null',seiz_times_id=seiz_label_id)
    label_q = bool(label_targets[0])
    
    # Load in seizure times
    seiz_times_loc = time_targets[2]
    save_dir       = seiz_times_loc.replace(seiz_times_id,'')
    f = open(seiz_times_loc)
    times = f.read().split()
    try: 
        int(times[0])
    except ValueError:
        times = times[4:]
    seiz_num = len(times)/4
    
    # Load in seizure labels
    if label_q:
        seiz_labels_loc = label_targets[2]
        f = open(seiz_labels_loc)
        label_list = f.read().split()
        label_seiz_num = len(label_list)
        if seiz_num != label_seiz_num:
            warnings.warn('Different number of seizure times and labels')
            label_q = False
    
    #seiz_times = np.zeros([seiz_num, 2])
    starts = list()
    ends = list()
    labels = list()
    for seiz in range(seiz_num):
        start_ind = seiz*4 + 1;
        end_ind   = seiz*4 + 2;
        start_val = float(times[start_ind])
        end_val   = float(times[end_ind])
        start_val = int(start_val*1000) # Convert seconds into miliseconds
        end_val = int(end_val*1000)     
        seiz_dur = len(range(start_val, end_val))
        if seiz_dur > 999:
            starts.append(start_val)
            ends.append(end_val)
            if label_q:
                labels.append(label_list[seiz])
            else:
                labels.append('None')
        #seiz_times[seiz, 0] = start_val
        #seiz_times[seiz, 1] = end_val
    save_q = bool(save)
    if save_q:
        s = open(save_dir + 'seiztimes.txt','wb')
        pickle.dump(seiz_times,s)
        s.close
        
    seiz_times = np.transpose(np.array([starts,ends])).astype('float64') 
    return seiz_times, seiz_times_loc, labels

def get_orig_data_from_session(session,refDir='/mnt/Data4/AnnotateSWD/',
                               origData=('/mnt/Data4/GAERS_EnsembleUnits/',
                                         '/mnt/Data2/GAERS_EnsembleUnits/')):
                        # As data moves to other locations, add more paths to
                        # the origData to ensure this script continues to work
    subprocess.call("/mnt/Data4/GAERS_Codes/DataExtraction/logRuns")
    underscores = [pos for pos, char in enumerate(session) if char == '_']
    animal_file = refDir + "AnimalLogFile.txt"
    if os.path.isfile(animal_file):
        with open(animal_file,'r') as t:
            session_to_ani = t.read().split()
    else:
        raise Exception("logRuns file in " + refDir + " doesn't exist")
    
    sessions = len(session_to_ani)
    for sess in range(sessions):
        cur_comp = session_to_ani[sess]
        if session in cur_comp:
            ani = session_to_ani[sess + 1]
            break
    session_name = session[0:underscores[1]]
    experiment_num   = session[underscores[1]+1]
    recording_num    = session[underscores[2]+1]
    for orig_loc_opt in origData:
        orig_data_path = (orig_loc_opt+ani+'/'+session_name+'/experiment' +
                     str(experiment_num) + '/recording' + str(recording_num) +
                     '/')
        if os.path.exists(orig_data_path):
            break
        
    
    return orig_data_path, ani
    
def get_data_path_cian(session,refDir='/mnt/Data4/AnnotateSWD/',
                       origData=('/mnt/Data4/GAERS_EnsembleUnits/',
                                 '/mnt/Data2/GAERS_EnsembleUnits/')):
    #Version of get_orig_data_from_session from above for Cian's thalamic data
    subprocess.call("/mnt/Data4/GAERS_Codes/DataExtraction/logRuns")
    animal_file = refDir + "AnimalLogFile.txt"
    if os.path.isfile(animal_file):
        with open(animal_file,'r') as t:
            session_to_ani = t.read().split()
    else:
        raise Exception("logRuns file in " + refDir + " doesn't exist")
        
    sessions = len(session_to_ani)
    for sess in range(sessions):
        cur_comp = session_to_ani[sess]
        if session in cur_comp:
            ani = session_to_ani[sess + 1]
            break
    for orig_loc_opt in origData:
        orig_data_path = (orig_loc_opt+ani+'/'+session+'/')
        if os.path.exists(orig_data_path):
            break
    return orig_data_path, ani

def gen_cell_objects(session,verbose=0,files=['.clu.','.res.'],save=1,
                     dataDir='/mnt/Data4/GAERS_Data/',downsample=30,cutoff=1,
                     refDir ='/mnt/Data4/AnnotateSWD/',spkFile = '.spk.',
                     max_clusters=15):
    # This function looks for cluster files in the current directory,
    # defaulting to .clu. and .res. files, extracting the relevant data, 
    # downsampling and creating a dataframe with the data of spikes
    # in each channel in each cluster, through a series of sub-routines.  This
    # function then seperates the data into individual objects and saves these
    # in the default location, or otherwise specified location
    
    # First we translate the session into the actual data path
    recording_id = session
    try:
        session, ani = get_orig_data_from_session(session)
    except:
        session, ani = get_data_path_cian(session)
    
    # Get unit type labels
    if "Sil" in ani:
        label = 'Cortical'
    else:
        label = 'Thalamic'
    
    TSD = extract_clusters(session,verbose,files,downsample,max_clusters)
    
    valid_spikes = TSD[TSD.Cluster_Num > cutoff]
    channels = TSD.Chan_Num.unique()
    for chan in channels:
        chan_data = valid_spikes[valid_spikes.Chan_Num == chan]
        cluster_vals = chan_data.Cluster_Num.unique()
        data_file = lf.locate_clusters(session,spkFile+str(chan))
        for clus in cluster_vals:
            channel_cell = chan_data[chan_data.Cluster_Num == clus]
            if len(channel_cell) == 0:
                continue
            start_time = channel_cell.Spk_Time.iloc[0]
            end_time   = channel_cell.Spk_Time.iloc[-1]
            cell_tuple = (start_time, end_time)
            cur_cell_obj = ccd.Cell(data_file,chan,recording_id,cell_tuple,
                                    channel_cell,label)
            ccd.save_object(cur_cell_obj)
    
    return 0

        
        

def extract_clusters(session,verbose = 0,files=['.clu.','.res.'],
                     downsample = 30,max_clusters=15):
    # This function cycles through the specified files and pairs them together.
    # This function is currently explicitly designed to pair together .clu and
    # .res files.  Further options should be made
    paths = lf.locate_clusters(session,files)
    if not bool(paths):
        raise Exception('Session specified is not valid - no files found')
        
    if len(files) > 1:
        for len_test in range(len(files)-1):
            if len(paths[files[len_test]]) != len(paths[files[len_test + 1]]):
                # Identify which paths are not good and then go from there            
                warnings.warn('Number of channels in desired files ' + 
                                   'is not equal.  Check the inputs')
    elif '.mat' in files:
        spike_data = extract_mat(paths,verbose)
        return spike_data
    else:
        print('''Only one type of file provided.  
              Numpy array of this data will be returned 'as is', if
              possible...''')
        data_dict = {}
        chan_dirs = paths[files[0]]
        file_type = files[0]
        num_chan = len(chan_dirs)
        array_sto= [None] * num_chan
        count = 0
        for chan_dir in chan_dirs:
            cur_chan_array = np.loadtxt(chan_dir)
            chan_id    = (chan_dir.split(file_type)[1])
            try:
                chan_number= int(chan_id)
                chan_number = chan_number - 1
            except:
                chan_id = chan_id[-1]
                chan_number = count
                count += 1
            array_sto[chan_number] = cur_chan_array
            data_dict[file_type + 'chan_num' + chan_id] = cur_chan_array
        data_dict[file_type] = array_sto
        return data_dict

    if '.clu.' in files and '.res.' in files:
        spike_data = extract_clu_res(paths,downsample,verbose,max_clusters)
        return spike_data
        

def extract_mat(paths,verbose):
    # This function does the actual heavy work of extracting the clu and res 
    # data from the relevant text files.  A new version of this function would
    # need to be made for different file types.
    
    spike_info = []
    mat_dirs = paths['.mat']
    overall_clusters = 0
    overall_spikes   = 0
    for mat in mat_dirs:
        # Get file info
        filename=(mat.split('/')[-1])
        nameparts = filename.split('_')
        try:
            chan_number = int(nameparts[1])
        except:
            continue
        try:
            cluster_number = int(nameparts[2].split('.')[0])
        except:
            continue

        #chan_id     = (clu.split('.clu.')[1])
        mat_file = loadmat(mat,matlab_compatible=True)
        #Get key number
        for idx in range(len(mat_file.keys())):
            if "spk" in mat_file.keys()[idx]:
                key = mat_file.keys()[idx]
        spike_data = mat_file[key]
        spike_data = np.squeeze(spike_data)
        #for cluster in range(spike_data.shape[0]):
        cluster_data = spike_data[0]
        for n in range(len(cluster_data)):
            spk_time = int(cluster_data[n])
            spike_info.append({"Chan_Num":chan_number,
                               "Cluster_Num":cluster_number,
                               "Spk_Time":spk_time,
                               "Spk_Num":n})
       
        if verbose:
            print('\n')
            print('Processing of')
            print(mat)
            print(clu)
            print('Completed')
            print(str(num_clusters) + ' clusters extracted with ' + 
                  str(spike_num) + ' spikes identified.')
        
    cellDF = pd.DataFrame(spike_info)
    if verbose:
        print('\n')
        print('Processing complete')
        print('A total of ' + str(overall_clusters) + ''' clusters were 
              extracted with a total of ''' +str(overall_spikes) + 
              ' spikes counted')
    
    return cellDF


def extract_clu_res(paths,downsample,verbose,max_clusters=15):
    # This function does the actual heavy work of extracting the clu and res 
    # data from the relevant text files.  A new version of this function would
    # need to be made for different file types.
    
    spike_info = []
    intArr = np.vectorize(np.int8)

    clu_dirs = paths['.clu.']
    overall_clusters = 0
    overall_spikes   = 0
    for clu in clu_dirs:
        
        cur_chan_array = intArr(np.loadtxt(clu))
        sorted_chan_array = np.sort(cur_chan_array)
        first_element = sorted_chan_array[0]
        if first_element != 0:
            warnings.warn(clu + ' has no trash group, and therefore has not '
                          + 'been clustered.  It is being skipped')
            continue
        
        res = clu.replace('.clu.','.res.')
        num_clusters = cur_chan_array[0]
         
        if num_clusters > max_clusters:
            warnings.warn(clu + ' has more than ' + str(max_clusters) + 
                          'clusters.  ')
        
            
        spike_iterator = np.where(cur_chan_array != 0); 
        spike_iterator = list(spike_iterator[0]); del spike_iterator[0]
        overall_clusters = overall_clusters + num_clusters
        spike_num = len(spike_iterator)
        overall_spikes = overall_spikes + spike_num
        chan_id     = (clu.split('.clu.')[1])
        chan_number = int(chan_id)
        with open(res) as res_file:
            lines = res_file.readlines()
            for pos_spk in spike_iterator:
                clus_num = cur_chan_array[pos_spk]
                time_index = pos_spk-1
                spk_time  = lines[time_index]
                spk_time  = int(round(int(spk_time)/downsample))
                spk_index = pos_spk
                spike_info.append({"Chan_Num":chan_number,
                                       "Cluster_Num":clus_num,
                                       "Spk_Time":spk_time,
                                       "Spk_Num":spk_index})         
        if verbose:
            print('\n')
            print('Processing of')
            print(res)
            print(clu)
            print('Completed')
            print(str(num_clusters) + ' clusters extracted with ' + 
                  str(spike_num) + ' spikes identified.')
        
    cellDF = pd.DataFrame(spike_info)
    if verbose:
        print('\n')
        print('Processing complete')
        print('A total of ' + str(overall_clusters) + ''' clusters were 
              extracted with a total of ''' +str(overall_spikes) + 
              ' spikes counted')
    
    return cellDF