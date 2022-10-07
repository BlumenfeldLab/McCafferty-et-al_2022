function plotTmap(indir,outdir,niiName,figname)
% Plots the t-map of seizure vs baseline for slices 2-11 using SPM
% Jingjing Li - 4/14/2021
%
% Args:
%   - niiName: the name of the nii file containing the t stats
%   - outdir: the output directory
%   - figname: name of figure

%% I/O
addpath(genpath('F:\xinyuan\GAERS\fMRI\glmCodes\spm12'));
addpath('F:\xinyuan\GAERS\fMRI\glmCodes\NIFTI_20110921');
homedir =  'F:\xinyuan\GAERS\fMRI\glmCodes\';
templateName = 'Template_IMG.nii';

% load mask
GMmask = load_nii([homedir 'VOI\Template_IMG_GM.nii']);
GMmask = logical(GMmask.img);

% load tstat
tstat = load_nii([indir niiName]);
tstat = tstat.img;
tstat(isnan(tstat)) = 0;
tstat = imresize(tstat,4);
tstat(not(GMmask)) = nan;
tstatPos = tstat;
tstatPos(tstatPos<=0) = nan;
tstatNeg = tstat .* -1;
tstatNeg(tstatNeg<=0) = nan;

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
model.axialslice = 1:10;

model.imgs{1, 1} = [homedir templateName];
model.imgs{2, 1} = fullfile([indir figname '_pos.nii']);
model.imgs{3, 1} = fullfile([indir figname '_neg.nii']);
display_slices_bai_2;
set(gcf,'PaperPositionMode','auto');
print('-depsc', [outdir figname]);
print('-dtiff', [outdir figname]);
close all;

end

