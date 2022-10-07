function plotTmapsSPM(indir,outdir,niiNamePos,niiNameNeg,figname)
% Jingjing Li - 4/14/2021
% Plots the t-map of seizure vs baseline for slices 2-11 using SPM
%
% Args:
%   - niiName: the name of the nii file containing the t stats
%   - outdir: the output directory
%   - figname: name of figure

%% I/O
addpath(genpath('F:\xinyuan\GAERS\fmri\glmCodes\spm12'));
addpath('F:\xinyuan\GAERS\fmri\glmCodes\NIFTI_20110921');
homedir =  'F:\xinyuan\GAERS\fmri\glmCodes\';
templateName = 'Template_IMG_WholeBrain_truncated.nii';

% load mask
% GMmask = load_nii([homedir 'Template_ROI.nii']);
GMmask = load_nii([homedir 'VOI\Template_IMG_GM.nii']);
GMmask = logical(GMmask.img);

% load tstat
tstatPos = load_nii([indir niiNamePos]);
tstatPos = tstatPos.img;
thisMin = min(tstatPos,[],'all','omitnan');
tstatPos(isnan(tstatPos)) = 0;
tstatPos = imresize(tstatPos,4);
tstatPos(tstatPos<thisMin/4) = 0;
tstatPos(tstatPos<=0) = nan;
tstatPos(not(GMmask)) = nan;
tstatPos = tstatPos(41:216,:,:);

tstatNeg = load_nii([indir niiNameNeg]);
tstatNeg = tstatNeg.img;
thisMin = min(tstatNeg(),[],'all','omitnan');
tstatNeg(isnan(tstatNeg)) = 0;
tstatNeg = imresize(tstatNeg,4);
tstatNeg(tstatNeg<thisMin/4) = 0;
tstatNeg(tstatNeg<=0) = nan;
tstatNeg(not(GMmask)) = nan;
tstatNeg = tstatNeg(41:216,:,:);

% save .nii images
nii = load_nii([homedir templateName]);
nii.hdr.dime.datatype = 16;
nii.hdr.dime.bitpix = int16(32);
nii.hdr.dime.roi_scale = 1;
nii.img = tstatPos;
nii.hdr.dime.cal_max = max(nii.img,[],'all');
save_nii(nii,[indir figname '_pos.nii'])
nii.img = tstatNeg;
nii.hdr.dime.cal_max = max(nii.img,[],'all');
save_nii(nii,[indir figname '_neg.nii'])

% Xinyuan Zheng - 10/19/2021
% voxel counts 
% for i_slide = 1:12
%     disp(['slice',num2str(i_slide), ' # of pos voxel:',num2str(sum(~isnan(tstatPos(:,:,i_slide)),'all'))])
%     disp(['slice',num2str(i_slide), ' # of neg voxel:',num2str(sum(~isnan(tstatNeg(:,:,i_slide)),'all'))])
% end

%% plot
global model;
model.xacross = 4;
model.itype{1} = 'Structural';
model.itype{2} = 'Blobs - Positive';
model.itype{3} = 'Blobs - Negative';
model.imgns{1} = 'Img 1 (Structural)';
model.imgns{2} = 'Img 2 (Blobs - Positive)';
model.imgns{3} = 'Img 3 (Blobs - Negative)';
model.range(:,1) = [0 1];
model.range(:,2) = [0 10]; 
model.range(:,3) = [0 10]; 
model.transform = 'axial'; %'axial','coronal','sagittal'
model.axialslice = 0:11;

model.imgs{1, 1} = [homedir templateName];
model.imgs{2, 1} = fullfile([indir figname '_pos.nii']);
model.imgs{3, 1} = fullfile([indir figname '_neg.nii']);
display_slices_bai_2;
set(gcf,'PaperPositionMode','auto');
print('-depsc', [outdir figname]);
print('-dtiff', [outdir figname]);
close all;

end

