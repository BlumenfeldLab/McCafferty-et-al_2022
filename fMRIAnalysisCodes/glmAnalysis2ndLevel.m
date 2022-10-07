%% Second level t-maps GLM analysis 
% Jingjing Li, Xinyuan Zheng - 09/10/2022

%% I/O
% add SPM path
addpath(genpath('F:\xinyuan\GAERS\fmri\glmCodes\spm12'))

homedir = 'F:\xinyuan\GAERS\fmri\glmCodes\';
target = 'mtsnr-snrmratrGd';
jobfile = {'F:\xinyuan\GAERS\fmri\glmCodes\glmAnalysis2ndLevel_job.m'};

%%
% get animals
anis = dir(fullfile([homedir target '_SPM\'],'fMRI_t*'));
anis = anis(2:end);
nanis = size(anis,1);

conImgs = cell(nanis,1);
for aniInd=1:nanis
    thisConImg = [anis(aniInd).folder '\' anis(aniInd).name '\con_0001.nii,1'];
    conImgs{aniInd} = thisConImg;
end

% List of open inputs
% Factorial design specification: Scans - cfg_files

nrun = 1; % enter the number of runs here

jobs = repmat(jobfile, 1, nrun);
inputs = cell(1, nrun);
for crun = 1:nrun
    inputs{1, crun} = conImgs; % Factorial design specification: Scans - cfg_files
end
spm('defaults', 'FMRI');
spm_jobman('run', jobs, inputs{:});
spm('Quit')
cd(homedir)

%% plot
inoutdir = [homedir target '_SPM\'];
plotTmapsSPM(inoutdir,inoutdir,'spmT_0001_FDR.nii','spmT_0002_FDR.nii','t-maps_FDR')
% plotTmapsSPM(inoutdir,inoutdir,'spmT_0001_FWE.nii','spmT_0002_FWE.nii','t-maps_FWE')

