function [ Seizures] = StateLickRates( FilLoc, Epoch, WinSize, FsEEG )
%FreeLickSzProperties - returns varies properties of seizures and
%baseline periods from the Spontaneous Licking behavioral paradigm
%   This function analyzes every appropriate* seizure from each session and
%   each animal in the Spontaneous Licking behavioral paradigm, and
%   extracts relevant seizure parameters including duration, lick rate
%   before/during/after seizure, vRMS amplitude of EEG before/during/after
%   seizure, and spectrogram of EEG before/during/after seizure (both vRMS
%   and spectrogram as time courses with downsampled resolution).

%   Input Arguments:
%   FilLoc is the full path of the directory where all files are located
%        e.g. FilLoc = '\\MWMJ03EEUL\GAERS_headfix_SeizureSeverityProject\Behavior\Data\Experiments\FreeLicking\';
%   Epoch is the length (in seconds) of time to be analyzed before and after the seizure
%   onset
%   WinSize is the length (in seconds) of the window containing each data
%   point (where relevant)
%   PreLickRate [min, max] is the allowable range of pre-seizure lick rates
%   BaseLine is the period of time in (s) used as a preseiz control for
%   LICK RATE and for EEG AMPLITUDE - tends to be a long period to establish arousal level
%   FsEEG is the EEG sampling rate in Hz

% Output Arguments:
%   First, a single seizure structure, with dimensions 1 x (no. of seizures). For every
%   seizure there will be a field identifying its animal of origin, session
%   no. within that animal, and a field for each of the desired properties.
%   Second, a "start control" structure, for randomly-selected control
%   periods matching the lick rate trajectories of pre-seizure periods, having
%   the same parameters as the seizure structure.
%   Third, an "end control" structure, for randomly-selected baseline
%   periods matching the lick rate trajectories of end-of-seizure periods,
%   having the same parameters as the seizure structure.
%   The final outputs contain the frequencies and times used for
%   spectrogram generation, and the input argument parameters passed.

% Sample Call:
% Seizures = StateLickRates('\\MWMJ03EEUL\E\GAERS_SeizureSeverity\Behavior\SpontaneousLicking\RawData\PreProcessed\',1200,50,1000)

ANIMALS_TO_ANALYZE = {'FL2\','FL4\','FL5\','FL6\','FL7\','FL16\',...
    'FL19\','FL20\','FL22\','FL23\','FL24\','FL25\','FL26\',...
    'FL27\','FL29\', 'FL30\', 'FL31\', 'FL32\', 'FL33\', 'FL35\', 'FL36\',...
    'FL37\', 'FL38\', 'FL39\', 'FL40\', 'FL41\', 'FL42\'};

% ANIMALS_TO_ANALYZE = {'SD12/','SD14/','SD15/','SD17/','SD17_halfs/','SD18/','SD18_halfs/','SD20/','SD21/','SD21_halfs/','SD22/','SD22_halfs/','SD23/','SD23_halfs/','SD24/','SD25_halfs/','SD26_halfs/','SD27_halfs/','SD28_halfs/','SD30_halfs/','SD31_halfs/','SD32_halfs/'};

% The maximum duration of the behavioral session in seconds
MaxBhvTime = 7200;


% This next line creates a counter that increments with every included seizure
SzCount = 0;

% And this one converts our Epoch units from seconds to number of windows
EpochWins = Epoch/WinSize;
% and also from seconds into number of samples
EpochSamps = Epoch*FsEEG;
% and from window size into number of samples
WinSamps = round(WinSize*FsEEG);

params.tapers=[2,3];
params.fpass=[0,100];
params.Fs = FsEEG;
movingwin=[WinSize*2,WinSize];

for s_animal = 1:size(ANIMALS_TO_ANALYZE,2) % go through each animal's folder
    disp(ANIMALS_TO_ANALYZE{s_animal});
    %% STEP 1: BUILD DIRECTORIES OF DATA FILES
    cd([FilLoc,(ANIMALS_TO_ANALYZE{1,s_animal})]) % go to the animal's directory
    seizures = dir([FilLoc,(ANIMALS_TO_ANALYZE{1,s_animal}),'*_sztms.txt']); % and seizure time files etc
    behav = dir([FilLoc,(ANIMALS_TO_ANALYZE{1,s_animal}),'*_behav.mat']); % MEDPC behavioral files, times adjusted to be relative to start of Spike2 recording session
    allchans = dir([FilLoc,(ANIMALS_TO_ANALYZE{1,s_animal}),'*_allchans.mat']); % spike2 files
    
    %for x_session = 1:8
    for x_session = 1:size(allchans,1) % iterates through each recording in the animal folder
        
        %% STEP 2: CHECK DATA FILES ARE PRESENT AND DATES MATCH
        FindDate = cell(1,4);
        FindDate{1,1} = strsplit(allchans(x_session).name,'_'); %Find date on eeg data
        FindDate{1,2} = strsplit(seizures(x_session).name,'_'); %Find date on seizures data
        FindDate{1,3} = strsplit(behav(x_session).name,'_'); %Find date on behav data
        
        DateCheck1 = strcmp(FindDate{1,1}{1,1},FindDate{1,1}{1,1}); %check to ensure dates on files are the same
        DateCheck2 = strcmp(FindDate{1,1}{1,1},FindDate{1,2}{1,1});%check to ensure dates on files are the same
        DateCheck3 = strcmp(FindDate{1,1}{1,1},FindDate{1,3}{1,1});%check to ensure dates on files are the same
        
        % Check if all dates for a given session are the same. If not,error.
        if DateCheck1+DateCheck2+DateCheck3 < 3 %if dates not the same, stop running script and display warning message
            f = warndlg('Dates do not match.', 'Warning Dialog');
            error('Discontinue processing');
            drawnow     % Necessary to print the message
            waitfor(f);
        end
        
        %% STEP 3: LOAD DATA FILES
        
        %3a: load EEG
        loadALL=load([FilLoc,(ANIMALS_TO_ANALYZE{1,s_animal}),allchans(x_session).name]); % loads all variables within the specified filename (has to be done this way because sometimes variable names break convention while filenames never do)
        varnames = fieldnames(loadALL); % checks what the variable (ie Spike2 channel) names are
        eegchanname = getfield(loadALL,char(varnames{1,1})); % pulls out the first subfield from the loaded file, which should be raw EEG
        allEEG = nan(size(eegchanname.values)); % creates an allEEG nan vector the size of the values field of this structure
        FreqRef = eegchanname.times;
        clear eegchanname;
        
        AReegchanname = getfield(loadALL,char(varnames{2,1})); % pulls out the second subfield, which should be artifact-removed EEG
        allEEG(round((AReegchanname.times-FreqRef(1,1))*FsEEG)+1) = AReegchanname.values; % sets the elements of the allEEG nan vector that correspond to existing values of this artifact removed EEG
        clear AReegchanname;
        
        %3b: load seizure times
        allseiz = load([FilLoc,(ANIMALS_TO_ANALYZE{1,s_animal}),seizures(x_session).name]); % seizure times can be loaded directly
        if isempty(allseiz) == 0 % making sure there were some seizures in this session, if not there's no point wasting processing time
            sztms = allseiz(:,2:4)*FsEEG; % Converts from the 1s unit output of sztms to 1ms units
            
            %3c: load behavioral data
            LoadMedPC = load([FilLoc,(ANIMALS_TO_ANALYZE{1,s_animal}),'\',behav(x_session).name]); %load MedPC (lick) data for that session
            % 1st column: time, 2nd column: behavior
            % (1=lick,13=start,12=stim)
            TimeEvent = LoadMedPC.CorrectedMedPC; %puts the variable containing behavioral data into Time_event
            
            %% STEP 4: CHECK SYNCHRONIZATION OF SPIKE2 & MEDPC DATA
            %4a: Behavior start, as recorded in Spike2. Note: Animals before FL33 have 3rd field as start time, but afterwards have 4th field as start time
            
            StrtTTLchanname = getfield(loadALL,char(varnames{end,1})); % third field of allchans should be start time
            
            
            BhvStrtSpk2 = StrtTTLchanname.times*FsEEG;
            
            if isempty(BhvStrtSpk2) == 1 %if there are no calibration times
                f = warndlg('No calibration info between Spike2 and Med-PC', 'Calibration Error');
                disp('Consider omitting this session');
                waitfor(f);
            elseif size(BhvStrtSpk2,1) > 1
                f = warndlg('Multiple start times detected', 'Calibration Error');
                disp('Consider omitting this session');
                waitfor(f);
            end
            
            %4b: Behavior start, as recorded in MedPC and adjusted
            BhvStrtMPC = TimeEvent(find((TimeEvent(:,2) == 13),1),1); %ID 13 in Time_event represents start signal
            
            if abs(BhvStrtMPC-BhvStrtSpk2) > 5 % if the synchronization is out by more than 10ms at session start
                f = warndlg('Synchronization failed.', 'Warning Dialog');
                disp('Discontinue processing');
                drawnow     % Necessary to print the message
                waitfor(f);
            end
            
            %4c: Adjust all times to be relative to this behavior start
            TimeEvent(:,1) = TimeEvent(:,1)- (BhvStrtSpk2 - 1); % express behavior event times relative to (behavior start-1ms)
            sztms(:,1:2) = sztms(:,1:2) - (BhvStrtSpk2 - 1); % express seizure start and end times relative to (behavior start-1ms)
            
            
            %% STEP 5: EXCLUDE PORTIONS OF SESSION THAT ARE MISSING EITHER EEG OR BEHAVIORAL DATA
            %Note all times are now relative to start of behavior session
            %5a: find behavior end
            RewTime = round(TimeEvent(TimeEvent(:,2) == 15,:)); %creates variable with reward times (rew is 15 on MedPC)
            RewNum = size(RewTime,1); %counts the number of total rewards in the session
            if RewNum == 50 %if there were 50 rewards in the session
                BhvEnd = RewTime(50,1) + 1*FsEEG; %behavioral data collection ends 1 second after the 50th reward
            elseif RewNum == 100
                BhvEnd = RewTime(100,1) + 1*FsEEG;
            else
                BhvEnd = MaxBhvTime * FsEEG; %if there haven't been 50 rewards, collection ends after 7200s
            end
            %%
            
            %5b: exclude EEG-derived info that occurred before or after
            %behavioral info
            allEEG = allEEG(BhvStrtSpk2:min(size(allEEG,1),BhvStrtSpk2+BhvEnd)); % remove EEG values from before or after behavior
            sztms(any(sztms<0,2),:) = []; % remove any seizures that were before the behavior start (ie have negative times)
            sztms(any(sztms>BhvEnd,2),:) = []; % remove any seizures that were after the end of the behavior
            if sum(any(sztms>size(allEEG,1),2)) > 0 % if there are any seizures supposedly occurring after the end of the EEG
                f = warndlg('Seizures Detected After EEG','Fatal Error'); % then we have a serious problem
                disp('Check session attribution of EEG, seizure, behavior files'); % the EEG, sztms, and bhv files don't match
                waitfor(f);
            end
            
            
            %5c: exclude behavioral info that occurred after EEG (before is not possible)
            TimeEvent(TimeEvent(:,1)>size(allEEG,1)) = []; % remove any behavioral events for which we have no EEG data
            
            %% STEP 6: CREATE LOGICALS FOR BINARY DATA TYPES
            
            %%
            % We now create "logicals" for the presence of physiological or
            % behavioral events (SWD, lick) at each sample of our session
            
            szlog = zeros(size(allEEG));
            for r_seizure = 1:size(sztms,1)
                szlog(sztms(r_seizure,1):sztms(r_seizure,2),1)=1; %creates array of zeroes for all eeg times, fills in 1 during seizures
            end
            
            licklog = zeros(size(allEEG));
            
            if sum(TimeEvent(TimeEvent(:,2)==1)) > 0 % if this file used the ID "1" for licks
                LicksOnly = TimeEvent(TimeEvent(:,2) == 1,:);
                LicksOnly = round(LicksOnly);
                licklog(LicksOnly(:,1),1) = 1; %fills in 1 for licking samples
            else
                LicksOnly = TimeEvent(TimeEvent(:,2) == 2,:); % or if the file used the ID "2" for licks
                LicksOnly = round(LicksOnly);
                licklog(LicksOnly(:,1),1) = 1;
            end
            
            %% STEP 7: ADD NaNs TO FACILITATE EPOCH SELECTION
            % we need to check if the seizure is
            % too close to the start or end of the session to
            % select a full desired epoch. If there are, we add on
            % NaN values TO ALL OUR DATA TYPES!
            
            for r_seizure = 1:size(sztms,1) % go through each seizure
                if sztms(r_seizure,1)-EpochSamps < 1 % if our seizure is closer to the start of the session than the size of our desired epoch
                    szlog = [NaN(EpochSamps,1);szlog]; % add NaNs on to the start to make sure it's long enough
                    licklog = [NaN(EpochSamps,1);licklog];
                    sztms = sztms+EpochSamps;
                    allEEG = [NaN(EpochSamps,1);allEEG];
                elseif sztms(r_seizure,2)+EpochSamps > size(szlog,1) % if it's not too close to the start, it could be too close to the end
                    szlog = [szlog;NaN(EpochSamps,1)];
                    licklog = [licklog;NaN(EpochSamps,1)];
                    allEEG = [allEEG;NaN(EpochSamps,1)];
                end
            end
            
            % Now double check that all our logicals are the same length
            % and still the same length as the EEG signal, if not we have
            % problems
            if size(licklog,1) ~= size(szlog,1) || size(licklog,1) ~= size(allEEG,1) || size(szlog,1) ~= size(allEEG,1)
                f = warndlg('Seizure/Behavior/EEG Sizes Do Not Match');
                disp('Check Session Attribution and Processing')
                waitfor(f)
            end
            
            % Before we go through seizure by seizure, stick an extra row
            % beforehand treating session start time as the "effective end
            % time" of seizure 0
            
            SzTms = [NaN,1,NaN;sztms;size(licklog,1),NaN,NaN];
            
            %
            
            %% STEP 8: EXTRACT/CALCULATE DESIRED PARAMETERS
            
            for r_seizure = 2:size(SzTms,1)-1 % go through seizures, starting with 2nd as 1st is actually just an "end time"
                SzCount = SzCount+1; % add one to the seizure counter
                
                % next we filter our lick logs to contain only licks
                % from this sz period, all else being nan
                
                thisszeeg = allEEG;thisszeeg(szlog==1) = nan;thisszeeg(SzTms(r_seizure,1):SzTms(r_seizure,2)) = allEEG(SzTms(r_seizure,1):SzTms(r_seizure,2));
                thisszlicks = licklog;thisszlicks(szlog==1) = nan;thisszlicks(SzTms(r_seizure,1):SzTms(r_seizure,2)) = licklog(SzTms(r_seizure,1):SzTms(r_seizure,2));

                [s,t,f] = mtspecgramc(thisszeeg(max(SzTms(r_seizure,1)-EpochSamps,1):SzTms(r_seizure,1)-1),movingwin,params); % output the frequencies and PSD of the signal, with a 0.5s window and no overlap at 1000 frequencies at 1000 Hz
                figure;imagesc([t,t+t],f,abs(10*log10(s)));set(gca,'YDir','Normal');title('Normalized starts seizures with licks');caxis([-32,-8]);

%                 thispreszlicklog = nan(size(licklog));thispreszlicklog(SzTms(r_seizure-1,2):SzTms(r_seizure,1)-1) = licklog(SzTms(r_seizure-1,2):SzTms(r_seizure,1)-1);
%                 thisszlicklog = nan(size(licklog));thisszlicklog(SzTms(r_seizure,1):SzTms(r_seizure,2)) = licklog(SzTms(r_seizure,1):SzTms(r_seizure,2));
%                 thispstszlicklog = nan(size(licklog));thispstszlicklog(SzTms(r_seizure,2):SzTms(r_seizure+1,1)) = licklog(SzTms(r_seizure,2):SzTms(r_seizure+1,1));
%                 
%                 % how many extra seconds of raw EEG and licking data do we want to add to the front and end of outputted epochs?
%                 % added by jacob prince 3/4/19
%                 epochPaddingInSec = 10;
%                 
                % also output raw lick log data for sz period and +/- a few seconds
%                 Seizures(SzCount).preszlicklog = thispreszlicklog(SzTms(r_seizure,1)-(epochPaddingInSec*FsEEG):SzTms(r_seizure,1));
%                 Seizures(SzCount).postszlicklog = thisszlicklog(SzTms(r_seizure,2):SzTms(r_seizure,2)+(epochPaddingInSec*FsEEG));
%                 Seizures(SzCount).szlicklog = thispstszlicklog(SzTms(r_seizure,1):SzTms(r_seizure,2));
                
                %% FIRST OUTPUT: ANIMAL NAME
                Seizures(SzCount).animal = ANIMALS_TO_ANALYZE{1,s_animal};
                
                %% SECOND OUTPUT: SESSION NO.
                Seizures(SzCount).session = x_session;
                
                %% THIRD OUTPUT: SEIZURE START LICKS IN BINS
                startlicks = thisszlicks;startlicks(SzTms(r_seizure,2)+1:end,:) = nan;
                % the next line sets the pre-seizure licks to be all
                % entries from the baseline licklog from 1 epoch
                % before sz start to 1 sample b/f sz start, reshaping
                % to find mean (ie sum divided by count) in each window
                Seizures(SzCount).LicksStart(1,1:EpochWins*2) = nansum(reshape(startlicks(max(SzTms(r_seizure,1)-EpochSamps,1):SzTms(r_seizure,1)+EpochSamps-1),WinSamps,EpochSamps*2/WinSamps),1)/WinSize;
                % and then the same for the start of seizure licks
%                 Seizures(SzCount).LicksStart(1,EpochWins+1:EpochWins*2) = nansum(reshape(thisszlicklog(SzTms(r_seizure,1):SzTms(r_seizure,1)+EpochSamps-1),WinSamps,EpochSamps/WinSamps),1)/WinSize;
                
                %% FOURTH OUTPUT: SEIZURE END LICKS IN BINS
                % next for end of seizure licks
                endlicks = thisszlicks;endlicks(1:SzTms(r_seizure,1)-1,:) = nan;
                Seizures(SzCount).LicksEnd(1,1:EpochWins*2) = nansum(reshape(endlicks(SzTms(r_seizure,2)-EpochSamps:SzTms(r_seizure,2)+EpochSamps-1),WinSamps,EpochSamps*2/WinSamps),1)/WinSize;
                % and finally for post-seizure licks
%                 Seizures(SzCount).LicksEnd(1,EpochWins+1:EpochWins*2) = nansum(reshape(thispstszlicklog(SzTms(r_seizure,2):SzTms(r_seizure,2)+EpochSamps-1),WinSamps,EpochSamps/WinSamps),1)/WinSize;
                
                %% NINTH OUTPUT: SEIZURE DURATION
                Seizures(SzCount).Duration = SzTms(r_seizure,3);
                
            end
        end
    end
end

end