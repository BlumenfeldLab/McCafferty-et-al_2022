%Compute_SNR
%% Computes SNR on functional images
% Originally created by Josh Ryu 170124
% Modified by Cian McCafferty, 7/13/2017
% Modified by Zach Kratochvil
% Modified by Peter Vincent 03/05/2019
function [baselineFrames] = compute_SNR_Baseline(targetDir,preReject,postReject,snr_threshold,Map_or_Mask)
addpath('NIFTI_20110921')
% Map of Mask: Map = 1, Mask = 0
%% Directories
template_img = load_nii('Template_IMG.nii');
home_dir = '/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_';
input_folder = targetDir;
countNANsAsZeros = false;
if Map_or_Mask == 1
    output_folder = [targetDir '_SNR_Maps_Baseline_' num2str(preReject) '_'...
        num2str(postReject) '_' num2str(snr_threshold)];
else
    output_folder = [targetDir '_SNR_Mask_Baseline_' num2str(preReject) '_'...
        num2str(postReject) '_' num2str(snr_threshold)];
end
regressor_dirs= ['SeizureTimes_' targetDir]; 
runs_dir = dir(fullfile(home_dir,input_folder,'runID_*'));
if ~exist(output_folder,'dir')
    mkdir(output_folder)
end
% Load a test file
run_num = 1;
load(fullfile(runs_dir(run_num).folder, ['runID_' num2str(run_num)], 'TimeImg.mat'));
imgDirs = size(TimeImg); imgDirs = imgDirs(1:3);
snrTotalImg = zeros(imgDirs);
totalNs = zeros(imgDirs);
% Load a template nii for the header
curNii = template_img;
load(fullfile(input_folder,'RunFileInfo.mat'))
baselineFrames = zeros(length(runs_dir),1);
for run_num = 1:length(runs_dir)
    curTime = runTimes{1,run_num};
    % load data
    load(fullfile(runs_dir(run_num).folder, ['runID_' num2str(run_num)], 'TimeImg.mat'));

    if ~isequal(size(TimeImg(:,:,:,1)),imgDirs)
        warning(['Run ' num2str(run_num) ' has unexpected 3d functional nii dimensions.']);
    end
    %%%  Based off the Seizure time matrices, 
    load(fullfile(regressor_dirs,['runID_' num2str(run_num)], 'regressors.mat'));
    seizNum = size(regressStr,1);
    rejectArray = zeros(1,length(curTime));
    for seiz = 1:seizNum
        curSeizStart = (regressStr(seiz,1)-regressStr(seiz,4))-preReject;
        curSeizEnd   = (regressStr(seiz,3)+regressStr(seiz,5))+postReject;
        for rej = curSeizStart : curSeizEnd
            location = curTime == rej;
            rejectArray(location) = 1;
        end
    end
    rejectArray = logical(rejectArray);
    baselineFrames(run_num) = length(rejectArray)-sum(rejectArray);
    TimeImg(:,:,:,rejectArray) = [];         
    % calculate voxel-wise SNRtempalte_hdr = template_img.hdr;
    meanVoxels = mean(TimeImg,4); % calculates mean value of each voxel in 4th dimension (across all time points)
    stdVoxels  = std(TimeImg,0,4); % calculates std of each voxel in 4th dimension (across all time points)
    snrVoxels  = 20*log10(meanVoxels./stdVoxels); % SNR is here defined as 20*log10(mean/std)
    snrVoxels(isinf(snrVoxels)) = nan; % Remove any infinite values
    snrVoxels(imag(snrVoxels) ~= 0) = nan;
    if countNANsAsZeros
        totalNs = totalNs + 1;
    else
        totalNs = totalNs + ~isnan(snrVoxels);
    end
    snrVoxels(isnan(snrVoxels)) = 0; % Remove any nan values
    snrTotalImg = snrTotalImg + snrVoxels;
    % save output
    dirName = fullfile(home_dir,output_folder,['runID_' num2str(run_num)]);
    if ~exist(dirName,'dir')
        mkdir(dirName);
    end
    save(fullfile(home_dir,output_folder,['runID_' num2str(run_num)],'SNR.mat'),'snrVoxels');
    if Map_or_Mask ~= 1
        snrVoxels = snrVoxels > snr_threshold;
    end
    curNii.img = snrVoxels;
    curNii.fileprefix = ['runID_' num2str(run_num) 'SNR'];
    curNii.original   = curNii.hdr;
    curNii.hdr.dime.dim(2:4) = imgDirs;
    curNii.dime.glmax = max(max(max(snrVoxels)));
    curNii.dime.glmin = min(min(min(snrVoxels)));
    save_nii(curNii,fullfile(home_dir,output_folder,['runID_' num2str(run_num)],'SNR.nii'));
end
snrAverageRun = snrTotalImg ./ totalNs;
if Map_or_Mask ~=1
    snrAverageRun = snrAverageRun > snr_threshold;
end
save(fullfile(home_dir,output_folder,'SNR_Average.mat'),'snrAverageRun','totalNs')
curNii.img = snrAverageRun;
curNii.fileprefix = ['overall_SNR'];
curNii.original   = template_img.hdr;
curNii.dime.glmax = max(max(max(snrAverageRun)));
curNii.dime.glmin = min(min(min(snrAverageRun)));
save_nii(curNii,fullfile(home_dir,output_folder,'SNR.nii'));

templateStruct = template_img.img;
saveDir = fullfile(output_folder,'AverageImages');
if ~exist(saveDir,'dir')
    mkdir(saveDir)
end
snrMask = snrAverageRun > snr_threshold;

structSize = size(templateStruct); structSize = structSize(1:2);
snrSize    = size(snrAverageRun);  snrSize    = snrSize(1:2);
ratio      = structSize ./ snrSize; ratio = unique(ratio);
if length(ratio) > 1
    warning('Sizes of SNR maps and structural template are incompatible')
else
    snrAverageRun = imresize(snrAverageRun,ratio);
end
snrMask = snrAverageRun > snr_threshold;
for slice = 1:12
    structImg= templateStruct(:,:,slice);
    snrSlice = snrAverageRun(:,:,slice);
    imgOverlay(structImg,snrSlice,0.3,1);
    saveas(gcf,fullfile(saveDir,['slice_' num2str(slice) '.tif']));
    close
    maskedSlice = snrMask(:,:,slice);
    imgOverlay(structImg,maskedSlice,0.3,1);
    saveas(gcf,fullfile(saveDir,['masked_thresh_' num2str(snr_threshold) 'slice_' num2str(slice) '.tif']));
    close
end
