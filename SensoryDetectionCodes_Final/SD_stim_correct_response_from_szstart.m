 %% Overview

%End result is:

%A) "ictal_stim": By column:
%1. animal
%2. session
%3. sz onset to stim
%4. if lick is within 10s of stim (1 yes 0 no) & during sz

clear

%% Loading data, setting variables, creating empty cell arrays to be filled in

%Enter here the IDs of the animals to be analyzed
ANIMALS_TO_ANALYZE = {'SD12\','SD14\','SD15\','SD17\','SD17_halfs\','SD18\','SD18_halfs\','SD20\','SD21\','SD21_halfs\','SD22\','SD22_halfs\','SD23\','SD23_halfs\','SD24\','SD25_halfs\','SD26_halfs\','SD27_halfs\','SD28_halfs\','SD30_halfs\','SD31_halfs\','SD32_halfs\'};

% 10s stimulus:{'SD12\','SD14\','SD15\','SD17\','SD18\','SD20\','SD21\','SD22\','SD23\','SD24\'};
% half-sec stimulus:{'SD17_halfs\','SD18_half-s\','SD21_halfs\','SD22_halfs\','SD23_halfs\','SD25_half\','SD26_halfs\','SD27_halfs\','SD28_halfs\','SD30_halfs\','SD31_halfs\','SD32_halfs\'};
% ALL: {'SD12\','SD14\','SD15\','SD17\','SD17_halfs\','SD18\','SD18_halfs\','SD20\','SD21\','SD21_halfs\','SD22\','SD22_halfs\','SD23\','SD23_halfs\','SD24\','SD25_half\','SD26_halfs\','SD27_halfs\','SD28_halfs\','SD30_halfs\','SD31_halfs\','SD32_halfs\'};
% Train: change location to Training, then {'SD25\','SD26\','SD27\','SD28\','SD30\','SD31\','SD32\'};

% Enter here the location of the files to be analyzed
fil_loc = 'W:\GAERS_SeizureSeverity\Behavior\SensoryDetection\RawData\PreProcessed\';

%Setting empty cell arrays to be filled in later. These variables are important final values.
ictal_stim= []; %Creates an empty cell array. See above for each column.
idx = 0; %Variable for indexing into output variable ictal_stim
MaxSampleTime = 7200; %total sample time in s
Fs = 1000; %sampling frequency
timeframe = 10; %time frame that rat must lick within in s
idx2 = 0; %Variable for indexing into variable StartCalibIssue
StartCalibIssue = []; %If there's a calibration issue between Spike2 and MedPC, animal/session will be recorded here

%% Import and callibrate all data for all animals and all sessions.
% For each animal

for i_animal = 1:size(ANIMALS_TO_ANALYZE,2) %Creates a loop for each animal
    % Note: '*' is a wildcard placeholder. These lines point to all directories
    % with the prefixes and suffixes specified, around the *.
    SzTms = dir([fil_loc,(ANIMALS_TO_ANALYZE{1,i_animal}),'*_sztms.txt']);%creates a directory of sztms data for each animal
    behav = dir([fil_loc,(ANIMALS_TO_ANALYZE{1,i_animal}),'*_behav.mat']);%creates a directory of bhv data for each animal
    allchans = dir([fil_loc,(ANIMALS_TO_ANALYZE{1,i_animal}),'*_allchans.mat']);%creates a directory of allchans data for each animal
    
    % For each session
    for j_session = 1:size(SzTms,1) % Creates a loop for each session
        %% CHECK TO MAKE SURE HAVE 3 FILES FOR EACH SESSION (BEHAV, SZTMS, ALLCHANS)
        FindDate = cell(1,4);
        FindDate{1,1} = strsplit(allchans(j_session).name,'_'); %Find date on eeg data
        FindDate{1,2} = strsplit(SzTms(j_session).name,'_'); %Find date on SzTms data
        FindDate{1,3} = strsplit(behav(j_session).name,'_'); %Find date on behav data
        
        DateCheck1 = strcmp(FindDate{1,1}{1,1},FindDate{1,1}{1,1}); %check to ensure dates on files are the same
        DateCheck2 = strcmp(FindDate{1,1}{1,1},FindDate{1,2}{1,1});%check to ensure dates on files are the same
        DateCheck3 = strcmp(FindDate{1,1}{1,1},FindDate{1,3}{1,1});%check to ensure dates on files are the same
        
        % Check if all dates for a given session are the same. If not,error.
        if DateCheck1+DateCheck2+DateCheck3 < 3 %if dates not the same, stop running script and display warning message
            f = warndlg('Dates do not match.', 'Warning Dialog');
            disp('Discontinue processing');
            drawnow     % Necessary to print the message
            waitfor(f);
        end
        
        %%Load all EEG data into loadAll, name, and build calibration parameters.
        
        loadALL=load([fil_loc,(ANIMALS_TO_ANALYZE{1,i_animal}),'\',allchans(j_session).name]); %load allchans (eeg) data for that session
        varnames = fieldnames(loadALL);  % checks what the variable names are
        
        substructure = getfield(loadALL,char(varnames{3,1})); %pull out Spike2 start time data for that recording. If there is an error here, it's likely that there is no artifact removed channel (needs 3 channels in allchans file: raw eeg, artifact removed, and start time)
        Spike2_start = substructure.times * 1000; %call the Spike2 start time the variable Spike2_start
       
        %% Load all behavioral data into LoadMedPC, name, and build calibration parameters.
        
        LoadMedPC = load([fil_loc,(ANIMALS_TO_ANALYZE{1,i_animal}),'\',behav(j_session).name]); %load MedPC (lick) data for that session
        % 1st column: time, 2nd column: behavior
        % (1=lick,13=start,12=stim)
        Time_event = LoadMedPC.CorrectedMedPC; %puts the variable containing behavioral data into Time_event
        MedPC_start = Time_event(find((Time_event(:,2) == 13),1),1); %finds the MedPC start time (1st column where 2nd column is 13)
        
        
        %%CALIBRATE ALL DATA
        
        %Check if start time in MedPC and Spike2 align
        if abs(Spike2_start - MedPC_start) < 10 %if MedPC & Spike2 start times are within 10ms
            
            %From now on, the "start time" used for calibration will be Spike2_start 
            
            % Calibrate the behavior time with start time so everything starts at t = 0.
            % To index properly, add 1 to make the start 1 instead of 0
            Time_event(:,1) = Time_event(:,1)-Spike2_start+1; %calibrates behavior times using start time
            
            %Determine the end time of the behavioral session: either after 50 rewards or at 7200s
            Rew_time = round(Time_event(Time_event(:,2) == 15,:)); %creates variable with reward times (rew is 15 on MedPC)
            Rew_num = size(Rew_time,1); %counts the number of total rewards in the session
            if Rew_num == 50 %if there were 50 rewards in the session
                Bhv_end = Rew_time(50,1) + 1*Fs; %behavioral data collection ends 1 second after the 50th reward
            else
                Bhv_end = MaxSampleTime * Fs; %if there haven't been 50 rewards, collection ends after 7200s
            end
            
            %%LOAD AND CALIBRATE SEIZURE TIMES DATA
           
            % Load all seizures into allseiz.
            allseiz = load([fil_loc,(ANIMALS_TO_ANALYZE{1,i_animal}),'\',SzTms(j_session).name]);%load SzTms for that session and call it allseiz
            
            if isempty(allseiz) == 0 %if allseiz is not empty (aka there are seizures in the session), continue...
                % Converts sztms from the 1s unit output of sztms to 1ms units
                sztimes = allseiz(:,2:4).*Fs;
                % Calibrate the seizure times/adjust seizure times to match when the task started
                sztimes(:,1:2) = sztimes(:,1:2)-(Spike2_start)+1; %adjust seizure times to match the start; add 1 because indexing
                sztimes(any(sztimes<0,2),:) = []; % remove any seizures that were before the task start (ie have negative times)
                sztimes(any(sztimes(:,2)>Bhv_end,2),:) = []; %remove any seizure after behavior end
                sztimes = round(sztimes); % ms resolution is sufficient, and sz times will be used as indices
                
                %%CREATE STIMLOG, LICKLOG, and SEIZLOG(for all times, will fill in a 1 for a stimulus, lick, or seizure)
                
                clear substructure;
                substructure = getfield(loadALL,char(varnames{1,1}));
                allEEG = substructure.values; %create allEEG vector with EEG values
                if size(allEEG,1) - round(Spike2_start) >= Bhv_end %if Spike2 session goes beyond behavioral data collection time
                    allEEG = allEEG(round(Spike2_start):round(Spike2_start) + Bhv_end - 1); %takes EEG values from behavior start to behavior end
                elseif size(allEEG,1) - round(Spike2_start) < Bhv_end %if behavioral data goes beyond Spike2 recording session
                    allEEG = allEEG(round(Spike2_start):end);
                    sztimes(any(sztimes(:,2)>size(allEEG,1),2),:) = []; %remove any seizure after behavior end
                    Time_event(any(Time_event(:,1)>size(allEEG,1),2),:) = []; %removes behavioral data after spike2 recording
                end
                
                szlog = zeros(size(allEEG));% creates array of zeroes for all eeg times, fills in 1 during seizures
                for r = 1:size(sztimes,1)
                    szlog(sztimes(r,1):sztimes(r,2),1)=1; %creates array of zeroes for all eeg times, fills in 1 during seizures
                end
                
                stimlog = zeros(size(allEEG));
                licklog = zeros(size(allEEG));
                
                if sum(Time_event(Time_event(:,2)==1)) > 0 % if this file used the ID "1" for licks
                    Stim_times = Time_event(Time_event(:,2) == 12,:);
                    Stim_times = round(Stim_times);
                    stimlog(Stim_times(:,1),1) = 1; %fills in 1 for stimulus onset times
                    
                    Licks_time = Time_event(Time_event(:,2) == 1,:);
                    Licks_time = round(Licks_time);
                    licklog(Licks_time(:,1),1) = 1; %fills in 1 for licking times
                                        
                else % if this file did not use the ID "1" for licks
                    f = warndlg('1 not used for licks', 'MedPC behav data error');
                    disp('Consider omitting this session or rat');
                    waitfor(f);
                end
                
                %%CALCULATIONS - START FILLING OUT VARIABLES FOR EACH STIMULI:
                
                %For each stimulus
                for r_times = 1:size(Stim_times,1) %goes through each stimulus
                    if szlog(Stim_times(r_times,1))==1 %if stim was in a seizure
                        idx = idx + 1;
                        ictal_stim(idx,1) = i_animal; %add the animal number to the first column
                        ictal_stim(idx,2) = j_session; %add the session number to the second column
                        
                        szstartend = diff(szlog); %changed the seizure starts to 1 and seizure ends to -1
                        szstart_idx = find(szstartend == 1); %finding the index of where szs start
                        szend_idx = find(szstartend == -1); %finding the index of where szs end
                        
                        szon_to_stim = Stim_times(r_times,1) - szstart_idx; %stimulus - all indices
                        szoff_to_stim = Stim_times(r_times,1) - szend_idx; %stimulus - all indices
                        
                        nearest_start = min(szon_to_stim(szon_to_stim > 0)); %finds the nearest sz start
                        nearest_end = max(szoff_to_stim(szoff_to_stim < 0)); %finds the nearest sz end
                        
                        ictal_stim(idx,3) = nearest_start; %add time from sz start to stim to third column
                        
                        if any(licklog(Stim_times(r_times,1):end)~=0) %looking at licks that occur from stim time to end of entire recording
                            %logical: if there is a lick within timeframe and before the end of the seizure
                            ictal_stim(idx,4) = (find(licklog(Stim_times(r_times,1):end),1) <= timeframe*Fs)& (find(licklog(Stim_times(r_times,1):end),1) < (-nearest_end));
                        else
                            %no more licks after stim time until the end of the session
                            ictal_stim(idx,4) = 0;
                        end
                    end
                end
            end
        else %make a variable that can record which animal/session has start calib issues btwn MedPC & Spike2
            StartCalibIssue(idx2,1) = i_animal;
            
            
            StartCalibIssue(idx2,2) = j_session;
            idx2 = idx2 + 1;
        end
    end
end
clear FindDate; clear DateCheck*;
