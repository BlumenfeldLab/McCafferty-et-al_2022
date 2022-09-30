#This function will take a folder name as the argument and output the location of:
	#(1) Location of any .tet files in the folder
	#(2) Location of any seiztimes.txt files in the folder
#The output will be a file called location_folderName.txt
# Renee Tung, Jacob Prince, Peter Vincent

import os
#import numpy as np
import glob
#from importlib import reload
def find_files(parentFolder,regex):
    file_list = []
    for (root,dirs,files) in os.walk(parentFolder):
        for name in files:
            #print(name)
            if 'autosave' in name:
                continue
            if regex in name:
                file_list.append(os.path.join(root, name))
    return file_list
            
def listFiles(fullPath, text_file): #lists name and location for .tet and seiztimes.txt files
	# find *.tet files
	#os.chdir(fullPath)
    print("listing files for path:")
    print(fullPath)
    print("****************")
    dirname = fullPath
    #print(dirname)
    tetFiles = glob.glob(dirname + '*.tet') #list of names that match
    if tetFiles: #if there's a .tet file..
        text_file.write("*.tet files: \n") #section header
        for file in tetFiles: #for each .tet file..
            text_file.write("%s \n" % tetFiles(file)) #tet file name
            text_file.write("%s \n" % fullPath) #tet file location
        else:
            print("no tet files found")

		# find seiztimes.txt files
    seizFiles = glob.glob(fullPath + 'seiztimes.txt') #list of names that match
    if seizFiles: #if there's a seiztimes.txt file..
        text_file.write("seiztimes.txt files: \n") #section header
        for file in seizFiles: #for each seiztimes.txt file..
            text_file.write("%s \n" % fullPath) #file location

def locate(folderName): #folderName will be a string, when calling use locate('folderName')
    fil_loc = '/mnt/Data4/AnnotateSWD/' 
    root_path = os.path.join(fil_loc, folderName)
    output_path = '/mnt/Data4/GAERS_Codes/'
	#text file that output will be written into
    text_file_fullpath = os.path.join(output_path, "Location_" + folderName + ".txt")
    text_file = open(text_file_fullpath, "w")
    print("successfully opened text file at path:")
    print(text_file_fullpath)
    text_file.write("Folder: %s \n" % folderName) #list folder name at the top

	#navigate to folder
	#os.chdir() #moves to specified folder
    #cwd = fil_loc + folderName #gets current working directory name
    #print(cwd)
	# inFolder = os.listdir(cwd) #lists files and subdirectories in the current directory
    #listFiles(root_path, text_file) #lists the files in the original folder
    
    for (root,dirs,files) in os.walk(root_path): #for each item in the current directory
        for name in dirs:
            print(os.path.join(root, name))
        for name in files:
            print(os.path.join(root, name))
            if 'autosave' in name:
                continue
            if name == 'seiztimes.txt':
                print("found a seiztimes file")
                text_file.write(os.path.join(root, name) + '\n')
            if '.tet' in name:
                print("found a tet file")
                text_file.write(os.path.join(root, name) + '\n')
                
            
        #fullPath = os.path.join(root_path,"/",root) #make fullPath name of the full path
        #listFiles(fullPath, text_file) #for each fullPath, do listFiles function

        #if os.path.isdir(fullPath): #if this entry is a subdirectory
        #    listFiles(fullPath, text_file) #list tet's or seiztimes in subdirectory

	#close output text file
    text_file.close()

    return

def locate_bool(folderName,datType,fil_loc = '/mnt/Data4/AnnotateSWD/',
                seiz_times_id='sztms.txt'):
    # This function searchs for seiztimes files and another specified data 
    # format
    root_path = os.path.join(fil_loc, folderName) # puts together the path from main dir & tgt folder
    seiz_exist = 0; dat_exist = 0 # at the start no seizures or data loaded yet
    seiz_file = []; dat_file = [] # create empty placeholders
    for (root,dirs,files) in os.walk(root_path): # walk thru this folder finding all folders & files within
        # then run through each file in each directory in the root folder
        for name in files: # for each individual file
            if 'autosave' in name: # ignore autosaves
                continue
            if '.pkl' in name: # ignore pickles
                continue
            if seiz_times_id in name:         #name == 'seiztimes.txt':
                seiz_exist = 1
                seiz_file  = os.path.join(root, name)
            if datType in name:
                dat_exist = 1
                dat_file.append(os.path.join(root, name))
    res_tuple = (seiz_exist, dat_exist, seiz_file, dat_file)
    return res_tuple

def locate_clusters(session, extensions):
    # This function searches for .clu# .res# and .spk# in the speficied 
    # directory
    locations = {}
    for ext_type in extensions:
        name_list = list()
        for (root,dirs,files) in os.walk(session):
            for name in files:
                if ext_type in name:
                    full_name = os.path.join(root,name)
                    if 'autosave' in full_name:
                        continue
                    if 'lfp' in full_name:
                        continue
                    name_list.append(full_name)
        locations[ext_type] = name_list
    return locations
                
                
