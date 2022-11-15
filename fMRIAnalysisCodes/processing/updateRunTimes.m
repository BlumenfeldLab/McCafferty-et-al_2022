% 22/03/2019 Peter Vincent

% This function takes a RunFileInfo.mat file from a source directory and
% updates it to reflect the conditions of the target directory.

function updateRunTimes(sourceFolder,targetFolder)
% The source folder is the location of the directory that contains the
% original runTimes information, containing the meta data for each runID_
% sub-directory.  This function updates this file with the runTimes in the
% targetFolder, before saving this variable in the targetFolder with the
% same name, without editing the meta-data

load(fullfile(sourceFolder,'RunFileInfo.mat'),'runTimes');
runFolds = dir(targetFolder);
if exist(fullfile(targetFolder,'RunFileInfo.mat'),'file')
    warning('The runTimes file already exists in the target directory');
end
runFolds(1:2) = []; runFolds = runFolds([runFolds.isdir]);
for runDir  = 1:length(runFolds)
    curRun  = (runFolds(runDir).name);
    runNum  = curRun(strfind(curRun,'_')+1:end);
    runNum  = str2double(runNum);
    fileList= dir([targetFolder '/' curRun '/' targetFolder '*.nii']);
    timeArray = zeros(length(fileList),1);
    if length(timeArray) > 600
        disp(['Error in ' fullfile(targetFolder,curRun) ' --  too many '...
            ' files. Check this.  Algorithm will continue, blanking '...
            'this run'])
        runTimes{1,runNum} = [];
        continue
    end
    for nii = 1:length(fileList)
        curNii = fileList(nii).name;
        niiNum = curNii(strfind(curNii,'.nii')-5:strfind(curNii,'.nii')-1);
        timeArray(nii,1) = str2double(niiNum);
    end
    timeArray = sort(timeArray);
    runTimes{1,runNum} = timeArray;
end
save([targetFolder '/' 'RunFileInfo.mat'],'runTimes');
end
    