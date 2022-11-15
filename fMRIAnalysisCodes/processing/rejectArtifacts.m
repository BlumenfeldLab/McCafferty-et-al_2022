% 18/07/2019 Peter Vincent
% 18/06/2021 Jingjing Li - to reject both chewing and movement artifacts

function rejectArtifacts(target,prefix,removeFirst10)

if nargin < 3
    removeFirst10 = true;
end

addpath('/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_/NIFTI_20110921');
parentDir = '/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_/';
metaPath  = fullfile(parentDir,'Meta');
chewingPath=fullfile(parentDir,'Chewing');

targetRuns = dir(fullfile(parentDir,target,'runID_*'));
outputDir = fullfile(parentDir,[prefix target]);
if exist(outputDir,'dir')
    warning('The output already exists');
else
    mkdir(outputDir);
end
for run = 1:length(targetRuns)
    curRun = targetRuns(run).name;
    outputFolder = fullfile(outputDir,curRun);
    mkdir(outputFolder);
    curNii = dir(fullfile(targetRuns(run).folder,targetRuns(run).name,'*.nii'));
    metaData = fullfile(metaPath,curRun);
    chewingData=fullfile(chewingPath,curRun);
    curRunArtifacts = parse_chewing_artifacts_hpc(metaData,chewingData);
    goodImages  = find(~curRunArtifacts); % indices of images without artifacts
    
    for nifti = 1:length(curNii)
        curImg = curNii(nifti).name;
        niiNum = curImg(end-8:end-4);
        niiNum = str2double(niiNum);
        isGood = find(goodImages == niiNum,1);
        source = fullfile(curNii(nifti).folder,curNii(nifti).name);
        targetNii = fullfile(outputFolder,[prefix curNii(nifti).name]);
        if (removeFirst10 && nifti<11) || isempty(isGood) % if this nii contains an artifact, replace all values with nan
            thisNii = load_nii([curNii(nifti).folder '/' curImg]);
            thisNii.img(:,:,:) = nan;
            save_nii(thisNii,targetNii);
        end
        copyfile(source,targetNii) % copy all good images to the target folder
    end
end
updateRunTimes(target,[prefix target]);

% create TimeImg
genTimeImg([prefix target],false)

%% remove movement artifacts
% load runTimes
load([parentDir prefix target '/RunFileInfo.mat'])

for runInd=1:194
    thisRunPath = [parentDir prefix target '/runID_' num2str(runInd) '/'];
    % load TimeImg and regressStr
    load([thisRunPath 'TimeImg.mat'])
    load([parentDir 'SeizureTimes_d/runID_' num2str(runInd) '/regressors.mat'])
    
    % identify movement artifacts
    thisFrames = runTimes{1,runInd};
    [rejInds,regressStr] = rejFramesVOI(TimeImg,thisFrames,'Template_IMG_GM','first derivative',regressStr);
    
    % remove movement artifacts by setting .nii files to nan
    curNii = dir(fullfile(thisRunPath,'*.nii'));
    rejInds = find(rejInds);
    for rejInd=1:length(rejInds)
        thisRejInd = rejInds(rejInd);
        thisRejIndStr = num2str(thisRejInd,'%05.f'); % 1 -> 00001
        thisNiiPath = dir(fullfile(thisRunPath,['*' thisRejIndStr '.nii']));
        thisNiiPath = fullfile(thisNiiPath(1).folder,thisNiiPath(1).name);
        thisNii = load_nii(thisNiiPath);
        thisNii.img(:,:,:) = nan;
        save_nii(thisNii,thisNiiPath);
    end
end

genTimeImg([prefix target],false)
end

        