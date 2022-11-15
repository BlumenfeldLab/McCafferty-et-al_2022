% 27/02/2019 Peter Vincent

% This script performs timecourse regression on a voxel by voxel basis
% acorss a 4D set of volumes
function TimeRegress(target_dir,output_dir,whole)
% The target_dir argument here lists the directory in which this function
% searches.  This function works on .nii files.  The output_dir gives the
% destination of the final files.  If whole is set to 0, then the
% regression is performed on each individual voxel.  If whole is set to 1,
% then it is performed on the whole brain
parent = '/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_';
cd(parent);
addpath('/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_/NIFTI_20110921');
runs = dir(target_dir); runs(1:2) = []; runDirs = [runs.isdir];
runs = runs(runDirs);
if ~exist([parent '/' output_dir],'dir')
    mkdir([parent '/' output_dir]);
else
    warning('The specified output directory already exists.  This program will continue');
end
cd(target_dir)
templateScanRun = fullfile(parent,'u','RunFileInfo.mat');
load(templateScanRun)

for runDir = 1:length(runs)
    curRunDir = ['runID_' num2str(runDir)];
    mkdir([parent filesep output_dir filesep curRunDir]);
    cd(curRunDir);
    niiFiles = dir('*.nii');
    templateFile = load_nii(niiFiles(1).name);
    imgSize = size(templateFile.img);
    scanDur = length(runTimes{1,runDir});
    if length(niiFiles) > scanDur
        error(['In ' curRunDur ' there are more .nii files than expected'])
    end
    timeVolume = nan([imgSize scanDur]);
    niftNum = niiFiles(1).name;
    niftNum = str2double(niftNum(end-8:end-4));
    timeVolume(:,:,:,niftNum) = templateFile.img;
    hdrStore = cell(1,scanDur);
    hdrInfo = templateFile; hdrInfo = rmfield(hdrInfo,'img');
    hdrStore{niftNum} = hdrInfo;
    for volume = 2:length(niiFiles)
        curVol = load_nii(niiFiles(volume).name);
        hdrInfo= curVol; hdrInfo = rmfield(hdrInfo,'img');
        niftNum= niiFiles(volume).name;
        niftNum = str2double(niftNum(end-8:end-4));
        hdrStore{niftNum} = hdrInfo;
        timeVolume(:,:,:,niftNum) = curVol.img;
    end
    if whole == 0
        for xdim = 1:imgSize(1)
            for ydim = 1:imgSize(2)
                for zdim = 1:imgSize(3)
                    curTC= timeVolume(xdim,ydim,zdim,:);
                    curTC= squeeze(curTC)';
                    fit  = polyfit(1:scanDur,curTC,3);
                    curve= polyval(fit,1:scanDur);
                    meanVal = ones(1,scanDur) .* mean(curve);
                    curTC = curTC - curve + meanVal;
                    timeVolume(xdim,ydim,zdim,:) = curTC;
                end
            end
        end
    else
        regressPoints = nan(1,scanDur);
        for tp = 1:scanDur
            regressPoints(tp) = nanmean(nanmean(nanmean(timeVolume(:,:,:,tp))));
        end
        nonNan = find(~isnan(regressPoints));
        fit  = polyfit(nonNan,regressPoints(nonNan),3);
        curve= polyval(fit,1:scanDur);
        meanVal = ones(1,scanDur) .* mean(curve);
        for xdim = 1:imgSize(1)
            for ydim = 1:imgSize(2)
                for zdim = 1:imgSize(3)
                    curTC = squeeze(timeVolume(xdim,ydim,zdim,:));
                    curTC = curTC' - curve + meanVal;
                    timeVolume(xdim,ydim,zdim,:) = curTC;
                end
            end
        end
    end
    for volume = nonNan
        curVol = int16(timeVolume(:,:,:,volume));
        curNii = hdrStore{volume}; curNii.img = curVol;
        curNii.hdr.dime.glmax = max(max(max(curVol)));
        curNii.hdr.dime.glmin = min(min(min(curVol)));
        for niiNames = 1:length(niiFiles)
            posName  = niiFiles(niiNames).name;
            posNumber=str2double(posName(end-8:end-4));
            if posNumber == volume
                niiName = posName;
                niiFiles(niiNames) = [];
                break
            end 
        end
        fileLoc = fullfile(parent,output_dir,curRunDir,['trG' niiName]);
        save_nii(curNii,fileLoc);
    end
    cd([parent filesep target_dir])
end
cd(parent);

updateRunTimes(target_dir,output_dir)
        
        
    