% 22/03/2019 Peter Vincent

% This function cycles through a folder and generates a TimeImg for each
% run, which will then be used in later analyses

function genTimeImg(targetDir,smooth)
% The targetDir argument is the folder for which we want to make the
% TimeImg variables/files for each run.
% smooth - true if using smoothed data, false otherwise
addpath('/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_/NIFTI_20110921')
runList = dir(targetDir);
runList(1:2) = []; runList = runList([runList.isdir]);
if exist([targetDir '/' 'RunFileInfo.mat'],'file')
    load([targetDir '/' 'RunFileInfo.mat'],'runTimes')
else
    updateRunTimes('u',targetDir);
    load([targetDir '/' 'RunFileInfo.mat'],'runTimes')
end
for run = 1:length(runList)
    curRun = runList(run).name;
    runNum = str2double(curRun(strfind(curRun,'_')+1:end));
    if ~contains(curRun,'runID')
        continue
    end
    if smooth
%         niiFile = dir([fullfile(targetDir,curRun) '/ssnr*.nii']); 
        niiFile = dir([fullfile(targetDir,curRun) '/s' targetDir '*.nii']); 
    else
        niiFile = dir([fullfile(targetDir,curRun) '/' targetDir '*.nii']); 
    end
    testNii = load_nii(fullfile(niiFile(1).folder,niiFile(1).name));
    niiDim  = size(testNii.img);
    TimeImg = zeros([niiDim length(niiFile)]);
    curNiiTimes = runTimes{1,runNum};
    for nii = 1:length(niiFile)
        curNii = niiFile(nii).name;
        niiNum = curNii(strfind(curNii,'.nii')-5:strfind(curNii,'.nii')-1);
        niiNum = str2double(niiNum);
        niiLoc = curNiiTimes == niiNum;
        niiImg = load_nii(fullfile(niiFile(nii).folder,curNii));
        TimeImg(:,:,:,niiLoc) = niiImg.img;
    end
    if smooth
        save(fullfile(targetDir,curRun,'sTimeImg.mat'),'TimeImg');
    else
        save(fullfile(targetDir,curRun,'TimeImg.mat'),'TimeImg');
    end
end