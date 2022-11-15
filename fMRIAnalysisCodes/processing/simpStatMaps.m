% 26/04/2019 Peter Vincent
% 04/26/2021 rewritten by Jingjing Li 

function simpStatMaps(target,rejTime,excludeRuns,maskStruc,snrMask)
% The target directory gives the parent folder that this function will look
% in, once it is appended with 'activity'.  the rejTime argument specifies
% the rejection number this function will look for, and the rejAni
% specifies the rejected animal this function will look for, if the
% argument is supplied
if nargin < 6
    snrMask = 0;
end
addpath('F:\xinyuan\GAERS\fmri\glmCodes\NIFTI_20110921');
prefix = target;
reject = rejTime;
% load seizure and baseline templates, taking rejAni into account
if excludeRuns
    seizTemplate = load(fullfile([prefix '_Activity'],[prefix '_SeizActiv_' num2str(reject) '_ExcludeRuns.mat'])); seizTemplate = seizTemplate.seizTemplate;
    baseTemplate = load(fullfile([prefix '_Activity'],[prefix '_BaseActiv_' num2str(reject) '_ExcludeRuns.mat'])); baseTemplate = baseTemplate.baseTemplate;
else
    seizTemplate = load(fullfile([prefix '_Activity'],[prefix '_SeizActiv_' num2str(reject) '_Exclude.mat'])); seizTemplate = seizTemplate.seizTemplate;
    baseTemplate = load(fullfile([prefix '_Activity'],[prefix '_BaseActiv_' num2str(reject) '_Exclude.mat'])); baseTemplate = baseTemplate.baseTemplate;
end

Template_Img = load_nii('Template_IMG.nii');
Template_ROI = load_nii('Template_ROI.nii');
SNR_ROI = load_nii('Template_ROISNR2.nii');
SNRmask = SNR_ROI.img;
GM_ROI = load_nii('F:\xinyuan\GAERS\fmri\glmCodes\VOI\Template_IMG_GM.nii');
GMmask = GM_ROI.img;
dirName = [prefix '_SimpleStatMaps'];
if excludeRuns
    dirName = [dirName '_ExcludeRuns'];
end 
if ~exist(dirName,'dir')
    mkdir(dirName);
end
mask = Template_ROI.img;
removeMask = logical(abs(1-double(mask))); % voxels to be removed
maskSize = size(removeMask);
% Resize functional images if neccessary
seizSliceSize = size(seizTemplate);
baseSliceSize = size(baseTemplate);
if baseSliceSize ~= seizSliceSize
    error('Baseline data and seizure data are somehow different sizes')
end
ratio = unique(seizSliceSize(1:2) ./ maskSize(1:2));
if length(ratio) > 1
    error('No fixed resizing for mask and images');
end
seizTemplate = imresize(seizTemplate,1/ratio);
baseTemplate = imresize(baseTemplate,1/ratio);

seizSize = size(seizTemplate);
baseSize = size(baseTemplate);

if size(seizSize,2) < 4
    seizNum = 1;
else
    seizNum = seizSize(4);
end
if size(baseSize,2) < 4
    baseNum = 1;
else
    baseNum = baseSize(4);
end
% Now we mask the data
if seizNum ~= baseNum
    error('Different amount of data for seizure and baseline data')
end
seizMat = nan(size(seizTemplate));
baseMat = nan(size(baseTemplate));

Template_Img = double(Template_Img.img);
for session = 1:seizNum
    seizSlice = seizTemplate(:,:,:,session);
    seizSlice(removeMask) = nan;
    baseSlice = baseTemplate(:,:,:,session);
    baseSlice(removeMask) = nan;
    seizMat(:,:,:,session) = seizSlice;
    baseMat(:,:,:,session) = baseSlice;
end
% And mask the template
if maskStruc
    Template_Img(removeMask) = 0;
end
percentChange = (seizTemplate ./ baseTemplate)-1;

% corrected by Jingjing 2/18/2021
% percentChange(removeMask) = nan;
meanPer    = nanmean(percentChange,4);
meanPer(removeMask) = nan;
if ~exist('insert','var')
    save(fullfile(dirName,[ num2str(reject) '_Exclude_PChangeMatrix.mat']),'meanPer');
else
    save(fullfile(dirName,[ num2str(reject) '_Exclude_' insert '_PChangeMatrix.mat']),'meanPer');
end
difTensor  = seizMat - baseMat;
meanDif    = nanmean(difTensor,4);
stdTensor  = nanstd(difTensor,0,4);
matSize = size(meanDif);
obsCount = zeros(matSize);
for x = 1:matSize(1)
    for y = 1:matSize(2)
        for z = 1:matSize(3)
            obsCount(x,y,z) = sum(~isnan(difTensor(x,y,z,:)));
        end
    end
end
steTensor = stdTensor ./ sqrt(obsCount);
tstat = meanDif ./ steTensor;
nanMatrix = isnan(tstat);
tstat(nanMatrix) = 0;
tstat(~logical(GMmask)) = nan;
mask(~logical(GMmask)) = 0;
if snrMask
    tstat(~logical(SNRmask)) = nan;
    mask(~logical(SNRmask)) = 0;
end
alphaVal        = 0.05;
if ~exist('insert','var')
    save(fullfile(dirName,[ num2str(reject) '_Exclude_StatMatrix.mat']),'tstat');
else
    save(fullfile(dirName,[ num2str(reject) '_Exclude_' insert '_StatMatrix.mat']),'tstat');
end
nonCorrUpThresh  = tinv(1-alphaVal/2,seizNum);
% nonCorrDownThresh= -1 * tinv(1-alphaVal/2,seizNum);
bonfCorr = zeros(size(mask,3),1);
for ii = 1:size(mask,3)
    bonfCorr(ii) = sum(mask(:,:,ii),'all');
end
% compute BH correction p threshold
pvals = tcdf(tstat,seizNum); % obtain uncorrected p vals
pvals(pvals>0.5) = (1-pvals(pvals>0.5)).*2;
pvals(pvals<=0.5) = pvals(pvals<=0.5).*2;
pvals_order = sort(pvals(not(isnan(pvals)))); % sort non-nan pvals in increasing order
V = length(pvals_order);
pthreshBH = 0;
for i=V:-1:1
    if pvals_order(i) <= (i/V)*alphaVal
        pthreshBH = pvals_order(i);
        break;
    end
end

correction = sum(bonfCorr);
BonCorrUpThresh  = tinv(1-alphaVal/(correction*2),seizNum);
% BonCorrDownThresh= -1 * tinv(1-alphaVal/(correction*2),seizNum);
if ~exist('insert','var')
    nonCorrTitle = [num2str(reject) '_Excluded_Noncorrected'];
    bonCorrTitle = [num2str(reject) '_Excluded_Bonf_Corrected'];
    bhCorrTitle  = [num2str(reject) '_Excluded_BH_Corrected'];
    steTitle     = [num2str(reject) '_Excluded_SdErr_Map'];
else
    nonCorrTitle = [num2str(reject) '_Excluded_Noncorrected_simple_map_Exclude_' insert];
    bonCorrTitle = [num2str(reject) '_Excluded_Bonf_Corrected_simple_map_Exclude_' insert];
    bhCorrTitle  = [num2str(reject) '_Excluded_BH_Corrected_simple_map_slice_' insert];
    steTitle     = [num2str(reject) '_Excluded_SdErr_Map_Exclude_' insert];
end

% plot non-thresholded t maps
plotTmaps(target,tstat,['Non_thresh_' num2str(reject) '_paired'])
    
% plot non-corrected t maps
tstatNonCorr = tstat;
tstatNonCorr(abs(tstatNonCorr)<nonCorrUpThresh) = nan;
plotTmaps(target,tstatNonCorr,nonCorrTitle)

% plot bonferroni corrected t maps
tstatBonf = tstat;
tstatBonf(abs(tstatBonf)<BonCorrUpThresh) = nan;
plotTmaps(target,tstatBonf,bonCorrTitle)

% plot BH corrected t maps
tstatBH = tstat;
tstatBH(pvals>pthreshBH) = nan;
plotTmaps(target,tstatBH,bhCorrTitle)