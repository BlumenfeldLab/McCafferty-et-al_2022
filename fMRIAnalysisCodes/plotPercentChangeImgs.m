%% Compute Stats & plot percent change images on the structural template 
% Jingjing Li - 03/03/2021, Xinyuan Zheng - 09/08/2022

clear; close all; clc

% data folder
target = 'mtsnr-snrmratrGd';

%% set constants
VOIname = 'Template_IMG_GM';
metric = 'first derivative';
plottingThreshold = [0.2; 2]; % the plotting threshold for percent change maps
nSecSzBl = 5; % the number of seconds of baseline to take per seizure
nSecSzBloffset = 3; % the end of baseline is 3s before seizure start: edited 080822

excludeDataPercent = 0; % the min percentage of non-artifact containing data
BLRejPercent = 100; % the max percentage change during bl above which to reject a seizure 
szAnalysisTime = 10; % the number of seconds to analyze during seizure
szRejPercent = 10; % the max percentage change during seizure above which to reject a seizure 
COMRejThreshold = 0.5; % the max amount of COM change above which to reject a seizure
preszLen = 10;
postszLen = 10;

%% I/O
addpath(genpath('F:\xinyuan\GAERS\fmri\glmCodes\spm12_radiological'));
addpath('F:\xinyuan\GAERS\fmri\glmCodes\NIFTI_20110921');
homedir =  'F:\xinyuan\GAERS\fmri\glmCodes\';

outdir = [homedir target '_PercentChange\'];
runPlotDir = [outdir 'Plots\All_Animals\AR\'];
if not(exist(outdir,'dir'))
    mkdir(outdir);
end
if not(exist(runPlotDir,'dir'))
    mkdir(runPlotDir);
end

% load run info
load([homedir target '\RunFileInfo.mat']); % load runTimes
% load mask
Template_Img = load_nii('Template_IMG.nii');
Template_ROI = load_nii('Template_ROI.nii');
removeMask = logical(abs(1-double(Template_ROI.img))); % voxels to be removed
SNR_ROI = load_nii('Template_ROISNR2.nii');
SNRmask = SNR_ROI.img;
GMmask = load_nii([homedir '\VOI\Template_IMG_GM.nii']);
GMmask = GMmask.img;

%% produce maps for each animal
% keep track of seizures included for onset and offset
szIncluded = cell(size(runTimes,2),2);

% exclude by runs
runInds = [1:size(runTimes,2)]; % get indices of all runs
excludeRunInds = loadExcludedRuns();
runTimes(:,excludeRunInds) = [];
runInds(excludeRunInds) = [];

% load animal info
runAnis = str2double(cellfun(@(x) x(7:end),runTimes(2,:),'UniformOutput',false))'; 
uniqueAnis = unique(runAnis);

% loop thru each animal and extract data
for aniInd=1:length(uniqueAnis)
    thisAni = uniqueAnis(aniInd);
    disp(['Processing fMRI_t' num2str(thisAni)])

    % initialize variables
    onsetData = {};
    offsetData = {};
    szMeanData = {};

    % get runs of this animal
    aniRunInds = runInds(runAnis==thisAni);

    % loop thru each run to extract data
    for runInd=1:length(aniRunInds)
        thisRun = aniRunInds(runInd);
        thisRunID = ['runID_' num2str(thisRun)];

        % load seizure times 
        thisReg = struct2cell(load([homedir 'SeizureTimes_d\' thisRunID '\regressors.mat']));
        thisReg = thisReg{1};
        % skip if no seizures
        if isempty(thisReg) 
            continue
        end

        % load TimeImg and frames
        load([homedir target '\' thisRunID '\sTimeImg.mat'])
        thisFrames = runTimes{1,runInds==thisRun};
        % create a vector indicating whether a frame is sz (1) or nonsz (0) or garbage (nan)
        thisIndicator = zeros(length(thisFrames),1);

        % reject frames by VOI timecourses
%             rejInds = rejFramesVOI(TimeImg,thisFrames,VOIname,metric,thisReg,[runPlotDir thisRunID '.tif']);
        rejInds = squeeze(sum(TimeImg,[1 2 3])) == 0;
        TimeImg(:,:,:,rejInds) = [];
        thisFrames(rejInds) = [];
        
        D0 = TimeImg;
        for szInd=1:size(thisReg,1)
            szStart = thisReg(szInd,1);
            szEnd = thisReg(szInd,3);
            thisIndicator(thisFrames(thisFrames>=szStart & thisFrames<=szEnd)) = 1;
            D0(:,:,:,thisFrames>=szStart & thisFrames<=szEnd) = NaN;
        end
        D0 = nanmean(D0,4);
        thisIndicator(rejInds) = nan;

        % keep track of included seizures
        thisSzIncludedOn = [];
        thisSzIncludedOff = [];

        % extract onset and offset data
        for szInd=1:size(thisReg,1)
            szStart = thisReg(szInd,1);
            szEnd = thisReg(szInd,3);

            % exclude seizure if its periictal window overlaps with
            % another seizure's
            if szInd==1 && szStart-thisFrames(1)<5
                continue
            end
            if szInd<size(thisReg,1)
                nextSzStart = thisReg(szInd+1,1);
                if nextSzStart-szEnd <= 5
                    continue
                end
            elseif thisFrames(end)-szEnd < 5
                continue
            end

            % recompute D0 if baseline is computed by seizure
            % exclude seizure if not enough baseline
            if sum(thisFrames>=szStart-nSecSzBl-nSecSzBloffset & thisFrames<szStart-nSecSzBloffset) < nSecSzBl*60*0.01
                continue
            elseif sum(thisIndicator(thisFrames(thisFrames>=szStart-nSecSzBl-nSecSzBloffset & thisFrames<szStart-nSecSzBloffset)),'omitnan') > 0
                continue
            end
            % exclude seizures with no data
            if sum(thisFrames>=szStart & thisFrames<=szEnd) < 1
                continue
            end
            % exclude seizure if too much baseline variance in VOI
            thisVOI = load_nii([homedir 'VOI\' VOIname '.nii']);
            thisVOI = imresize(logical(thisVOI.img),0.25);
            D0 = TimeImg(:,:,:,thisFrames>=szStart-nSecSzBl-nSecSzBloffset & thisFrames<szStart-nSecSzBloffset);
            thisVOItc = zeros(1,size(D0,4));
            for tsInd=1:size(D0,4)
                temp = D0(:,:,:,tsInd);
                thisVOItc(tsInd) = nanmean(temp(thisVOI));
            end
            if (nanmax(thisVOItc)-nanmin(thisVOItc))/nanmin(thisVOItc)>0.01*BLRejPercent
                continue
            end
            D0 = nanmean(D0,4);

            % exclude seizure if it contains too many artifcats
            if sum(thisFrames>=szStart & thisFrames<=szEnd) < excludeDataPercent*0.01*(szEnd-szStart+1)
                continue
            end

            thisSz = TimeImg(:,:,:,thisFrames>=szStart & thisFrames<=szEnd);
            thisSzTimes = thisFrames(thisFrames>=szStart & thisFrames<=szEnd) - szStart + 1;

            % onset aligned
            thisOnsetData = nan(64,32,12,preszLen+szAnalysisTime);
            for tsInd=1:preszLen
                thisPreszInd = szStart-preszLen-1+tsInd; % absolute time
                if thisPreszInd < 1
                    continue
                end
                if thisIndicator(thisPreszInd)==0 % nonseizure
                    thisOnsetData(:,:,:,tsInd) = TimeImg(:,:,:,thisFrames==thisPreszInd);
                end
            end
            for tsInd=1:szAnalysisTime
                if sum(thisSzTimes==tsInd)==0
                    continue
                end
                thisOnsetData(:,:,:,preszLen+tsInd) = thisSz(:,:,:,thisSzTimes==tsInd);
            end
            
            % offset aligned
            thisOffsetData = nan(64,32,12,postszLen+szAnalysisTime);
            for tsInd=1:postszLen
                thisPostszInd = szEnd+tsInd; % absolute time
                if thisPostszInd > thisFrames(end)
                    continue
                end
                if thisIndicator(thisPostszInd)==0 % nonseizure
                    thisOffsetData(:,:,:,szAnalysisTime+tsInd) = TimeImg(:,:,:,thisFrames==thisPostszInd);
                end
            end
            for tsInd=1:szAnalysisTime
                szEnd = thisSzTimes(end);
                if sum(thisSzTimes==szEnd-tsInd+1)==0
                    continue
                end
                thisOffsetData(:,:,:,szAnalysisTime-tsInd+1) = thisSz(:,:,:,thisSzTimes==szEnd-tsInd+1);
            end

            if decideInclusionSz(thisOnsetData,VOIname,szRejPercent,COMRejThreshold) && decideInclusionSz(thisOffsetData,VOIname,szRejPercent,COMRejThreshold)
%             if decideInclusionSz(cat(4,thisOnsetData,thisOffsetData),VOIname,szRejPercent,COMRejThreshold) && decideInclusionSz(thisOffsetData,VOIname,szRejPercent,COMRejThreshold)
                for tsInd=1:preszLen+szAnalysisTime
                    thisOnsetData(:,:,:,tsInd) = (thisOnsetData(:,:,:,tsInd)-D0)./D0*100;
                end
                onsetData{end+1} = thisOnsetData;
                thisSzIncludedOn = [thisSzIncludedOn szInd];
                
                for tsInd=1:postszLen+szAnalysisTime
                    thisOffsetData(:,:,:,tsInd) = (thisOffsetData(:,:,:,tsInd)-D0)./D0*100;
                end
                offsetData{end+1} = thisOffsetData;
                thisSzIncludedOff = [thisSzIncludedOff szInd];
                
                thisSzMean = (mean(thisSz,4,'omitnan')-D0)./D0*100;
                szMeanData{end+1} = thisSzMean;
            end
        end

        szIncluded{thisRun,1} = thisSzIncludedOn;
        szIncluded{thisRun,2} = thisSzIncludedOff;
    end

    % save maps
    savedir = [outdir 'fMRI_t' num2str(uniqueAnis(aniInd)) '\'];
    if not(exist(savedir,'dir'))
        mkdir(savedir);
    end
    save([savedir 'onsetData.mat'],'onsetData')
    save([savedir 'offsetData.mat'],'offsetData')
    save([savedir 'szMeanData.mat'],'szMeanData')
    clear onsetData
    clear offsetData
    clear szMeanData
end
savedir = [outdir 'All_Animals\'];
if not(exist(savedir,'dir'))
    mkdir(savedir);
end
save([savedir 'szIncluded.mat'],'szIncluded')


%% plot percent change maps for each animal 
% load presaved data
aniFolders = dir(fullfile(outdir,'fMRI_t*'));
aniNames = extractfield(aniFolders,'name');

for aniInd=1:length(aniNames)
    thisAni = aniNames{aniInd};
%     thisAni = 'fMRI_t10';
disp(['Running for Animal ' thisAni '...'])
% init matrices to hold group averaged data
onsetDataAll = {};
offsetDataAll = {};
szMeanDataAll = {};
    
load([outdir thisAni '\onsetData.mat']) % load onsetData
load([outdir thisAni '\offsetData.mat']) % load postszData
load([outdir thisAni '\szMeanData.mat']) % load szMeanData

% average pre and post sz data
if not(isempty(onsetData))
    temp = {onsetDataAll, onsetData'};
    onsetDataAll = cat(1,temp{:});
end
if not(isempty(offsetData))
    temp = {offsetDataAll, offsetData'};
    offsetDataAll = cat(1,temp{:});
end
if not(isempty(szMeanData))
    temp = {szMeanDataAll, szMeanData'};
    szMeanDataAll = cat(1,temp{:});
end

onsetDataMean = imresize(nanmean(cat(5,onsetDataAll{:}),5),4); % 256 128 12 22
offsetDataMean = imresize(nanmean(cat(5,offsetDataAll{:}),5),4); % 256 128 12 22
for szInd=1:length(szMeanDataAll)
    temp = szMeanDataAll{szInd};
    temp(temp==Inf) = nan;
    szMeanDataAll{szInd} = temp;
end
szMeanDataMean = imresize(nanmean(cat(4,szMeanDataAll{:}),4),4); % 256 128 12

% mask data
for timeInd=1:size(onsetDataMean,4)
    thisData = onsetDataMean(:,:,:,timeInd);
    if not(length(size(thisData))==length(size(removeMask)))
        disp('here')
    end
    thisData(removeMask) = NaN;
    onsetDataMean(:,:,:,timeInd) = thisData;
end
for timeInd=1:size(offsetDataMean,4)
    thisData = offsetDataMean(:,:,:,timeInd);
    thisData(removeMask) = NaN;
    offsetDataMean(:,:,:,timeInd) = thisData;
end
szMeanDataMean(removeMask) = NaN;

savedir = [outdir thisAni '\']; 
if not(exist(savedir,'dir'))
    mkdir(savedir);
end

nii = load_nii('Template_IMG_WholeBrain_truncated.nii');
nii.hdr.dime.datatype = 16;
nii.hdr.dime.bitpix = int16(32);
nii.hdr.dime.roi_scale = 1;
for tsInd=1:preszLen+szAnalysisTime
    thisData = onsetDataMean(:,:,:,tsInd);
    thisData(not(logical(SNRmask))) = nan;
    thisData(not(logical(GMmask))) = nan;
    thisDataPos = thisData;
    thisDataPos(thisDataPos<0.01) = NaN;
    thisDataPos = thisDataPos(41:216,:,:);
    thisDataNeg = thisData;
    thisDataNeg(thisDataNeg>-0.01) = NaN;
    thisDataNeg = thisDataNeg .* (-1);
    thisDataNeg = thisDataNeg(41:216,:,:);
    nii.img = thisDataPos;
    nii.hdr.dime.cal_max = max(nii.img,[],'all');
    save_nii(nii,[savedir 'onset_pos_' num2str(tsInd) '.nii'])
    nii.img = thisDataNeg;
    nii.hdr.dime.cal_max = max(nii.img,[],'all');
    save_nii(nii,[savedir 'onset_neg_' num2str(tsInd) '.nii'])
end
for tsInd=1:size(offsetDataMean,4)
    thisData = offsetDataMean(:,:,:,tsInd);
    thisData(not(logical(SNRmask))) = nan;
    thisData(not(logical(GMmask))) = nan;
    thisDataPos = thisData;
    thisDataPos(thisDataPos<0.01) = NaN;
    thisDataPos = thisDataPos(41:216,:,:);
    thisDataNeg = thisData;
    thisDataNeg(thisDataNeg>-0.01) = NaN;
    thisDataNeg = thisDataNeg .* (-1);
    thisDataNeg = thisDataNeg(41:216,:,:);
    nii.img = thisDataPos;
    nii.hdr.dime.cal_max = max(nii.img,[],'all');
    save_nii(nii,[savedir 'offset_pos_' num2str(tsInd) '.nii'])
    nii.img = thisDataNeg;
    nii.hdr.dime.cal_max = max(nii.img,[],'all');
    save_nii(nii,[savedir 'offset_neg_' num2str(tsInd) '.nii'])
end

thisData = szMeanDataMean;
thisData(not(logical(SNRmask))) = nan;
thisData(not(logical(GMmask))) = nan;
thisDataPos = thisData;
thisDataPos(thisDataPos<0.01) = NaN;
thisDataPos = thisDataPos(41:216,:,:);
thisDataNeg = thisData;
thisDataNeg(thisDataNeg>-0.01) = NaN;
thisDataNeg = thisDataNeg .* (-1);
thisDataNeg = thisDataNeg(41:216,:,:);
nii.img = thisDataPos;
nii.hdr.dime.cal_max = max(nii.img,[],'all');
save_nii(nii,[savedir 'szmean_pos.nii'])
nii.img = thisDataNeg;
nii.hdr.dime.cal_max = max(nii.img,[],'all');
save_nii(nii,[savedir 'szmean_neg.nii'])

save([savedir 'onsetDataAll.mat'],'onsetDataAll','-v7.3')
save([savedir 'offsetDataAll.mat'],'offsetDataAll','-v7.3')

%% plot for group
%Setup SPM Model

global model;
model.xacross = 'auto';
model.itype{1} = 'Structural';
model.itype{2} = 'Blobs - Positive';
model.itype{3} = 'Blobs - Negative';
model.imgns{1} = 'Img 1 (Structural)';
model.imgns{2} = 'Img 2 (Blobs - Positive)';
model.imgns{3} = 'Img 3 (Blobs - Negative)';
model.range(:,1) = [0 1];
model.range(:,2) = plottingThreshold; 
model.range(:,3) = plottingThreshold; 
model.transform = 'axial'; %'axial','coronal','sagittal'
model.axialslice = [2:9];
filedir = [outdir thisAni '\']; 
savedir = [outdir thisAni '\Plots\' ]; 
if not(exist(savedir,'dir'))
    mkdir(savedir);
end

for tsInd=1:preszLen+szAnalysisTime
%     model.imgs{1, 1} = '/gpfs/ysm/project/blumenfeld/prv4/GAERS_fMRI/PV_Autumn_2018_/Template_IMG.nii';
    model.imgs{1, 1} = [homedir 'Template_IMG_WholeBrain_truncated.nii'];
    model.imgs{2, 1} = fullfile([filedir 'onset_pos_' num2str(tsInd) '.nii']);
    model.imgs{3, 1} = fullfile([filedir 'onset_neg_' num2str(tsInd) '.nii']);
    display_slices_bai_2;
    set(gcf,'PaperPositionMode','auto');
    print('-dtiff', [savedir 'onset_' num2str(tsInd)]);
    close all;
end
for tsInd=1:postszLen+szAnalysisTime
%     model.imgs{1, 1} = '/gpfs/ysm/project/blumenfeld/prv4/GAERS_fMRI/PV_Autumn_2018_/Template_IMG.nii';
    model.imgs{1, 1} = [homedir 'Template_IMG_WholeBrain_truncated.nii'];
    model.imgs{2, 1} = fullfile([filedir 'offset_pos_' num2str(tsInd) '.nii']);
    model.imgs{3, 1} = fullfile([filedir 'offset_neg_' num2str(tsInd) '.nii']);
    display_slices_bai_2;
    set(gcf,'PaperPositionMode','auto');
    print('-dtiff', [savedir 'offset_' num2str(tsInd)]);
    close all;
end

model.imgs{1, 1} = [homedir 'Template_IMG_WholeBrain_truncated.nii'];
model.imgs{2, 1} = fullfile([filedir 'szmean_pos.nii']);
model.imgs{3, 1} = fullfile([filedir 'szmean_neg.nii']);
display_slices_bai_2;
set(gcf,'PaperPositionMode','auto');
print('-dtiff', [savedir 'szmean']);
close all;



%% Plot VOI timecourse and extract stats for each animal 
resdir = fullfile(outdir,thisAni,'Res');
if not(exist(resdir,'dir'))
    mkdir(resdir);
end

VOIlist = {'VB','SomatoCtx','CPU_upper','CPU_lower'};

for VOIInd=1:length(VOIlist)
[VOIMeanOn, VOIMeanOff, VOIAllBlMean, VOIAllSzMean] = plotROITC(thisAni, ...
    fullfile(resdir,[VOIlist{VOIInd} '.tif']), VOIlist{VOIInd},nSecSzBl,nSecSzBloffset);

Stats.Animal = thisAni;
Stats.VOI = VOIlist{VOIInd};
Stats.MeanOn_plot = VOIMeanOn;
Stats.MeanOff_plot = VOIMeanOff;
Stats.BlMean = VOIAllBlMean;
Stats.SzMean = VOIAllSzMean;
save(fullfile(resdir,[VOIlist{VOIInd} '_stats']),'Stats')

end

end



%% Get final statistics and VOI timecourse plots for all animals  
addpath(genpath('F:\xinyuan\GAERS\fmri\glmCodes\spm12_radiological'));
addpath('F:\xinyuan\GAERS\fmri\glmCodes\NIFTI_20110921');
target = 'mtsnr-snrmratrGd';
homedir =  'F:\xinyuan\GAERS\fmri\glmCodes\';
outdir = [homedir target '_PercentChange\'];

aniFolders = dir(fullfile(outdir,'fMRI_t*'));
aniNames = extractfield(aniFolders,'name');

VOIlist = {'VB','SomatoCtx','CPU_upper','CPU_lower'};
for VOIInd=1:length(VOIlist)
    thisVOI = VOIlist{VOIInd};
    disp(['Running for VOI ', thisVOI, ' ...']);
    
    MeanOn_plot = zeros(length(aniNames),22);
    MeanOff_plot = zeros(length(aniNames),22);
    BlMeanAll = zeros(length(aniNames),1);
    SzMeanAll = zeros(length(aniNames),1);
    for aniInd=1:length(aniNames)
        thisAni = aniNames{aniInd};
        VOIstats = load(fullfile(outdir,thisAni,'Res', [thisVOI '_stats.mat']));
        
        BlMeanAll(aniInd) = VOIstats.Stats.BlMean;
        SzMeanAll(aniInd) = VOIstats.Stats.SzMean; % 22
        MeanOn_plot(aniInd,:) = VOIstats.Stats.MeanOn_plot; % animal number x time point 18x22
        MeanOff_plot(aniInd,:) = VOIstats.Stats.MeanOff_plot;
    end
    
    nSzOn = size(MeanOn_plot,1);
    nSzOff = size(MeanOff_plot,1);
    
    [~,VOIonP,VOIonCI,~] = ttest(BlMeanAll,SzMeanAll);
    disp([thisVOI, ' pairwise t-test p-value: ', num2str(VOIonP)]);
    disp([thisVOI, ' pairwise t-test CI: ', num2str(VOIonCI')]);
    disp([thisVOI, ' seizure mean: ', num2str(mean(SzMeanAll))]);
    disp([thisVOI, ' seizure SEM: ', num2str(std(SzMeanAll) / sqrt(sum(~isnan(SzMeanAll))))]);
    
    StatsRes.(VOIlist{VOIInd}).p = VOIonP;
    StatsRes.(VOIlist{VOIInd}).CI = VOIonCI;
    StatsRes.(VOIlist{VOIInd}).szMean = mean(SzMeanAll);
    StatsRes.(VOIlist{VOIInd}).szSEM = std(SzMeanAll) / sqrt(sum(~isnan(SzMeanAll)));
    
    VOIMeanOn = mean(MeanOn_plot,1,'omitnan'); % average across seizure
    VOISEMOn = std(MeanOn_plot,1,'omitnan') ./ sqrt(sum(not(isnan(MeanOn_plot)),1));
    VOIMeanOff = mean(MeanOff_plot,1,'omitnan'); % average across seizure
    VOISEMOff = std(MeanOff_plot,1,'omitnan') ./ sqrt(sum(not(isnan(MeanOff_plot)),1));
    
    szAnalysisTime = 12;
    
    figure('Position',[10 10 900 600])
    onsetx = -10:(szAnalysisTime-1);
    offsetx = (1-szAnalysisTime):10;
    
    ax1 = subplot(1,2,1);
    plot(ax1,onsetx,VOIMeanOn, 'Color', [0 0.4470 0.7410 1],'LineWidth',1.3); hold on
    plot(ax1,onsetx,VOIMeanOn-VOISEMOn,'Color',[0.4660 0.6740 0.1880 0.7]); hold on
    plot(ax1,onsetx,VOIMeanOn+VOISEMOn,'Color',[0.4660 0.6740 0.1880 0.7])
    yline(ax1,0,'k--'); xline(ax1,0,'k--');
    title(ax1,[char(join(strsplit(thisVOI,'_'))) ' (n=' num2str(nSzOn) ')']);    
    
    ax2 = subplot(1,2,2);
    plot(ax2,offsetx,VOIMeanOff, 'Color', [0 0.4470 0.7410 1],'LineWidth',1.3);hold on
    plot(ax2,offsetx,VOIMeanOff-VOISEMOff,'Color',[0.4660 0.6740 0.1880 0.7]);hold on
    plot(ax2,offsetx,VOIMeanOff+VOISEMOff,'Color',[0.4660 0.6740 0.1880 0.7])
    yline(ax2,0,'k--');xline(ax2,0,'k--');
    title(ax2,[char(join(strsplit(thisVOI,'_'))) ' (n=' num2str(nSzOff) ')']);
    
    ylabel(ax1,'BOLD Percent Change');
    xlabel(ax1,'Seconds to Seizure Onset');
    xlabel(ax2,'Seconds to Seizure Offset');
    xlim(ax1,[-10,15]);xlim(ax2,[-15,10]);
    ylim(ax1,[-2.5,2]);ylim(ax2,[-2.5,2]);
    
    saveResdir = fullfile(outdir,'\All_Animals\Res');
    if not(exist(saveResdir,'dir'))
        mkdir(saveResdir);
    end
    saveas(gcf,fullfile(saveResdir,[VOIlist{VOIInd} '.eps']));
    saveas(gcf,fullfile(saveResdir,[VOIlist{VOIInd} '.tif']));
    close all;
end
save(fullfile(saveResdir, 'StatsRes.mat'),'StatsRes');

