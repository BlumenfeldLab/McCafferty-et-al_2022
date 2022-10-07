function [VOIMeanOn, VOIMeanOff, VOIAllBlMean, VOIAllSzMean] = plotROITC_090822(thisAni,savepath,VOIname,nSecSzBl,nSecSzBloffset)
% Plot ROI timecourse
% Xinyuan Zheng - 09/08/2022, Jingjing Li
% 
% Args:
%   - thisAni: animal ID
%   - savepath: the output directory
%   - VOIname: name of VOI
%   - nSecSzBl: baseline duration in sec
%   - nSecSzBloffset: time between ictal start and baseline end in sec

%% I/O
% add NIFTI tools path. Jimmy Shen (2022).
addpath('F:\xinyuan\GAERS\fmri\glmCodes\NIFTI_20110921');
homedir = 'F:\xinyuan\GAERS\fmri\glmCodes\'; % code folder
target = 'mtsnr-snrmratrGd'; % data folder

szAnalysisTime = 12;
szTtestTime = 10;
preszLen = 10;

load(fullfile(homedir,[target '_PercentChange'], thisAni, 'onsetDataAll.mat'))
load(fullfile(homedir,[target '_PercentChange'], thisAni, 'offsetDataAll.mat'))
nSzOn = length(onsetDataAll);
nSzOff = length(offsetDataAll);

% VOI template
VOIMask = load_nii(['F:\xinyuan\GAERS\fmri\glmCodes\VOI\Cian_VOI\' VOIname '.nii']);
VOIMask = logical(VOIMask.img);
VOIMask = repmat(VOIMask,1,1,1,preszLen+szAnalysisTime);

%% calculate TCs
onsetData = cellfun(@(data) applyMask(data,VOIMask),onsetDataAll,'UniformOutput',false);
VOITCon = cellfun(@(x) reshape(mean(x,1:3,'omitnan'),1,[]),onsetData,'UniformOutput',false); 
VOITCon = vertcat(VOITCon{:});
VOITCon(VOITCon==Inf) = nan; % seizure number x time points
VOIMeanOn = mean(VOITCon,1,'omitnan'); % average across seizure
VOISEMOn = std(VOITCon,1,'omitnan') ./ sqrt(sum(not(isnan(VOITCon)),1));

%% Stats for BOLD time course analysis

% get baseline mean - take average across time
VOIblmean = mean(VOITCon(:, preszLen-nSecSzBl-nSecSzBloffset+1 : preszLen-nSecSzBloffset), 2, 'omitnan'); % bl: -8s to -3s before szOn
VOIblmean = VOIblmean(sum(isnan(VOITCon(:,preszLen-nSecSzBl-nSecSzBloffset+1 : preszLen-nSecSzBloffset)),2)<5); % if NaN exists average over the non-NaN values
VOIblmean(find(sum(isnan(VOITCon(:,preszLen-nSecSzBl-nSecSzBloffset+1 : preszLen-nSecSzBloffset)),2)==5))=NaN; % if all NaNs returns NaN 
% get sz mean - take average across time
VOIszonmean = mean(VOITCon(:,10+1:10+szTtestTime), 2, 'omitnan'); % sz: 10s after szOn
VOIszonmean = VOIszonmean(sum(isnan(VOITCon(:,10+1:10+szTtestTime)),2)<szTtestTime); % if NaN exists average over the non-NaN values
VOIszonmean(find(sum(isnan(VOITCon(:,10+1:10+szTtestTime)),2)==szTtestTime))=NaN; % if all NaNs returns NaN 

% take average across seizures
VOIAllBlMean = mean(VOIblmean, 1);
VOIAllSzMean = mean(VOIszonmean, 1);

%%

clear VOITCon
clear onsetData
clear onsetDataAll

offsetData = cellfun(@(data) applyMask(data,VOIMask),offsetDataAll,'UniformOutput',false);
VOITCoff = cellfun(@(x) reshape(mean(x,1:3,'omitnan'),1,[]),offsetData,'UniformOutput',false);
VOITCoff = vertcat(VOITCoff{:});
VOITCoff(VOITCoff==Inf) = nan;
VOIMeanOff = mean(VOITCoff,1,'omitnan');
VOISEMOff = std(VOITCoff,1,'omitnan') ./ sqrt(sum(not(isnan(VOITCoff)),1));

clear VOITCoff 
clear offsetData
clear offsetDataAll

%% plot
figure('Position',[10 10 900 600])
onsetx = -10:(szAnalysisTime-1);
offsetx = (1-szAnalysisTime):10;
ax1 = subplot(1,2,1);
plot(ax1,onsetx,VOIMeanOn)
hold on
plot(ax1,onsetx,VOIMeanOn-VOISEMOn,'Color',[0, 0, 1, 0.3])
hold on
plot(ax1,onsetx,VOIMeanOn+VOISEMOn,'Color',[0, 0, 1, 0.3])
yline(ax1,0,'k--')
xline(ax1,0,'k--')
title(ax1,[VOIname ' (n=' num2str(nSzOn) ')'])
ax2 = subplot(1,2,2);
plot(ax2,offsetx,VOIMeanOff)
hold on
plot(ax2,offsetx,VOIMeanOff-VOISEMOff,'Color',[0, 0, 1, 0.3])
hold on
plot(ax2,offsetx,VOIMeanOff+VOISEMOff,'Color',[0, 0, 1, 0.3])
yline(ax2,0,'k--')
xline(ax2,0,'k--')
title(ax2,[VOIname ' (n=' num2str(nSzOff) ')'])
ylabel(ax1,'BOLD Percent Change')
xlabel(ax1,'Seconds to Seizure Onset')
xlabel(ax2,'Seconds to Seizure Offset')
xlim(ax1,[-10,15])
xlim(ax2,[-15,10])
ylim(ax1,[-2.5,1.5])
ylim(ax2,[-2.5,1.5])

saveas(gcf,savepath)

close all
end