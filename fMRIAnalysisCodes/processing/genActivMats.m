% 21/11/2018 Peter Vincent

% This script is part of the GAERS fMRI project.  It is desgined to build
% 3D matrices of the seizure activity and the baseline activity, combining
% all the animals, for a given set of images.  These can then be used in
% other analyses, since these matrices are saved to the parentDir

% We need to cycle through and generate a 3d matrices of all seizure
% activity and all non-seizure activity (away from the seizures by 20
% seconds)
function [validSeizRuns, nSzTotal] = genActivMats(targetDir,excludeRuns,includeSzs,blBySz)
% This function will make activity matrices that can then be used to
% generate simple statistic maps or percent change maps in later analyses.
% The targetDir argument specifies the prefix of the files we want to be
% analysing.  The rejAni is a vector argument to specify the animal number
% to be rejected.  If we wish to include all animals, then this argument is
% left empty <[]>. excludeRuns is a boolean variable indicating whether
% data need to be excluded by run. 
% blBySz: true or false; compute baseline by run or seizure.

%% set parameters
VOIname = 'Template_IMG_GM';
metric = 'first derivative';
szBLlen = 5;
blRejPercent = 60; % the minimum baseline a seizure must have to be included
smooth = true; % whether to use smoothed data or not 

%% I/O
addpath('/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_/NIFTI_20110921');
prefix = targetDir;
parentDir = '/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_';

% load excluded runs
if excludeRuns
    excludeRunInds = loadExcludedRuns();
end

% load szIncluded
if includeSzs
    load([parentDir '/' prefix '_PercentChange_10_10/All_Animals/szIncluded.mat'])
end

prefix = targetDir;
parentDir = '/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_';
runTimesFile = fullfile(targetDir,'RunFileInfo.mat');
load(runTimesFile);
saveDir = fullfile(parentDir,[prefix '_Activity']);
if ~exist(saveDir,'dir')
    mkdir(saveDir)
end

folders = dir(targetDir); folders = folders([folders.isdir]);
folders(1:2) = [];
maxT = 300;
runNum = length(folders);
sizeVec = [64,32,12,runNum];
if contains(prefix,'US')
    sizeVec = [256,128,12,runNum];
end
if blBySz
    seizTemplate = {};
    baseTemplate = {};
else
    seizTemplate = zeros(sizeVec);
    baseTemplate = zeros(sizeVec);
end

validSeizRuns= zeros(1,runNum);
seizTemplateT= zeros([sizeVec(1:3) maxT]); % 4-D matrix
seizNormT    = zeros(1,maxT);
seizNorm = 0;
baseNorm = 0;
rejectParam = 20;
noSeizDelete = zeros(1,runNum);
nSzTotal = 0;
for ii = 1:runNum
    curRun = ['runID_' num2str(ii)];
    eventFile = ['SeizureTimes_' prefix filesep curRun filesep 'regressors.mat'];
    if includeSzs
        thisOnsetIncluded = szIncluded{ii,1};
        thisOffsetIncluded = szIncluded{ii,2};
        thisIncludedSzs = intersect(thisOnsetIncluded,thisOffsetIncluded);
    end
    if smooth
        timeFile  = [targetDir filesep curRun filesep 'sTimeImg.mat'];
    else
        timeFile  = [targetDir filesep curRun filesep 'TimeImg.mat'];
    end
    if ~exist(timeFile,'file')
        continue
    end
    load(eventFile); 
    
    % Make the event times;
    seizNum = size(regressStr,1);
    if seizNum == 0 || (excludeRuns && excludeRunInds(ii)) || (includeSzs && isempty(thisIncludedSzs))
        noSeizDelete(ii) = 1;
        disp(['Skipping run number ' num2str(ii)])
        continue
    end
    
    load(timeFile);
    % reject frames
    validTimes = runTimes{1,ii};
    [rejInds,regressStr] = rejFramesVOI(TimeImg,validTimes,VOIname,metric,regressStr);
    TimeImg(:,:,:,rejInds) = [];
    validTimes(rejInds) = [];
    
    seizVec = zeros(length(validTimes),1);
    baseVec = ones(length(validTimes),1);
    
    for seiz = 1:seizNum
        szStart = regressStr(seiz,1);
        szEnd = regressStr(seiz,3);

        if not(includeSzs && sum(thisIncludedSzs==seiz)==0)
            curSeiz = (validTimes >= szStart) & (validTimes <= szEnd);
            curSeizAbs = validTimes(curSeiz);
            if isempty(curSeizAbs)
                continue
            end
            
            if blBySz
                thisBLinds = (validTimes>=szStart-szBLlen) & (validTimes<szStart);
                if sum(thisBLinds) < blRejPercent*0.01*szBLlen
                    continue
                end
                baseTemplate{end+1} = mean(TimeImg(:,:,:,thisBLinds),4);
                seizTemplate{end+1} = mean(TimeImg(:,:,:,curSeiz),4);
            end
            
            seizL = curSeizAbs - curSeizAbs(1) + 1;
            if sum(isnan(TimeImg(:,:,:,curSeiz)),'all') == 0
                seizTemplateT(:,:,:,seizL) = seizTemplateT(:,:,:,seizL) + TimeImg(:,:,:,curSeiz);
                seizNormT(seizL) = seizNormT(seizL) + ones(1,length(seizL));
            end
            seizVec(curSeiz) = 1;
            nSzTotal = nSzTotal + 1;
        end

        curExclude = (validTimes >= regressStr(seiz,1)-rejectParam) & (validTimes <= regressStr(seiz,3)+rejectParam);
        baseVec(curExclude) = 0;
    end
    
    seizVec = logical(seizVec);
    baseVec = logical(baseVec);
    
    if not(blBySz)
        seizActiv = mean(TimeImg(:,:,:,seizVec),4); % mean across all frames during seizure
        baseActiv = mean(TimeImg(:,:,:,baseVec),4); % mean across all baseline
        seizTemplate(:,:,:,ii) = seizActiv;
        seizNorm = seizNorm + 1;
        validSeizRuns(ii) = 1;
        baseTemplate(:,:,:,ii) = baseActiv;
        baseNorm = baseNorm + 1;
    end

    disp(['Processed ' num2str(ii)]);
end

% only keep times with at least five seizures
rejectTimes = seizNormT < 5;
seizNormT(rejectTimes) = [];
seizTemplateT(:,:,:,rejectTimes) = [];
for time = 1:length(seizNormT)
    seizTemplateT(:,:,:,time) = seizTemplateT(:,:,:,time) ./ seizNormT(time);
end
if excludeRuns
    seizSaveName = fullfile(saveDir,[prefix '_SeizActiv_' num2str(rejectParam) '_ExcludeRuns.mat']);
    baseSaveName = fullfile(saveDir,[prefix '_BaseActiv_' num2str(rejectParam) '_ExcludeRuns.mat']);
    seizTSaveName = fullfile(saveDir,[prefix '_SeizActivT_' num2str(rejectParam) '_ExcludeRuns.mat']);
else
    seizSaveName = fullfile(saveDir,[prefix '_SeizActiv_' num2str(rejectParam) '_Exclude.mat']);
    baseSaveName = fullfile(saveDir,[prefix '_BaseActiv_' num2str(rejectParam) '_Exclude.mat']);
    seizTSaveName = fullfile(saveDir,[prefix '_SeizActivT_' num2str(rejectParam) '_Exclude.mat']);
end

if blBySz
    seizTemplate = cat(4,seizTemplate{:});
    baseTemplate = cat(4,baseTemplate{:});
else
    validSeizRuns = logical(validSeizRuns);
    seizTemplate(:,:,:,not(validSeizRuns)) = [];
    baseTemplate(:,:,:,not(validSeizRuns)) = [];
end

save(baseSaveName,'baseTemplate','-v7.3');
save(seizSaveName,'seizTemplate','-v7.3');
save(seizTSaveName,'seizTemplateT','-v7.3');
end
