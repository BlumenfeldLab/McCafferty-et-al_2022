%% First level fixed-effect t-maps based on SPM GLM analysis 
% Jingjing Li, Xinyuan Zheng - 09/10/2022

%% add the path of SPM12 to MATLAB
addpath(genpath('F:\xinyuan\GAERS\fmri\glmCodes\spm12'))

%% set parameters
VOIname = 'Template_IMG_GM';
metric = 'first derivative';
preszLen = 10;
postszLen = 10;

% generate hrf bf
bf.UNITS = 'scans';
bf.dt = 0.0625;
bf.name = 'hrf';
bf.Volterra = 1;
bf.T = 16;
bf.T0 = 8;
bf.order = 1;
bf = spm_get_bf(bf); % SCALE HRF TO HAVE MAX AMPLITUDE OF 1

%% I/O
% add the path of nifti tools by Jimmy Shen
addpath('F:\xinyuan\GAERS\fmri\glmCodes\NIFTI_20110921');
% Jimmy Shen (2022). Tools for NIfTI and ANALYZE image 
% (https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image),
% MATLAB Central File Exchange. Retrieved October 7, 2022.

homedir = 'F:\xinyuan\GAERS\fmri\glmCodes\';
target = 'mtsnr-snrmratrGd';

% load szIncluded
load([homedir target '_PercentChange\All_Animals\szIncluded.mat'])

% load runTimes
load([homedir target '\RunFileInfo.mat'])
runInds = 1:size(runTimes,2);
excludeRunInds = loadExcludedRuns();
runTimes(:,excludeRunInds) = [];
runInds(excludeRunInds) = [];
runAnis = str2double(cellfun(@(x) x(7:end),runTimes(2,:),'UniformOutput',false))'; % ani info by run
uniqueAnis = unique(runAnis);

nrun = length(uniqueAnis); % number of runs = number of animals

%%
for aniInd=1:nrun
    matlabbatch = {};
    thisAni = uniqueAnis(aniInd);
    savedir = [homedir target '_SPM\fMRI_t' num2str(thisAni)];
    if exist(savedir,'dir')
        rmdir(savedir,'s');
    end 
    mkdir(savedir);
    if not(exist(savedir,'dir'))
        mkdir(savedir);
    end
    
    % init spm job
    matlabbatch{1}.spm.stats.fmri_spec.dir = cellstr(savedir);
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'scans';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = 1;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = 16;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = 8;
    
    % get runs of this animal
    aniRunInds = runInds(runAnis==thisAni);
    niisNameAni = {};
    szStartsAni = [];
    szDurAni = [];
    blStartsAni = [];
    blDurAni = [];
    timeInc = 0; % the amount of time increments due to previous scans
    regressorAniSz = [];
    regressorAniBl = [];
    regressorSessSz = [];
    nSecsSess = [];
    
    batchRun = 0; 
    niisNameSess = [];
    
    for runInd=1:length(aniRunInds)
        thisRunInd = aniRunInds(runInd); % 1-194
        thisSess = runTimes{3,runInds==thisRunInd};
        runID = ['runID_' num2str(thisRunInd)];
        thisFrames = runTimes{1,runInds==thisRunInd};
        thisRegressorSz = zeros(thisFrames(end),1);
        thisRegressorBl = ones(thisFrames(end),1);
        
        % get included szs in this run
        thisOnsetIncluded = szIncluded{thisRunInd,1};
        thisOffsetIncluded = szIncluded{thisRunInd,2};
        thisIncludedSzs = intersect(thisOnsetIncluded,thisOffsetIncluded);
        
        % load sz and bl start times and durations
        load([homedir 'SeizureTimes_d\' runID '\regressors.mat'])
        if size(regressStr,1)==0 || isempty(thisIncludedSzs)
            if runInd == length(aniRunInds) || not(strcmp(thisSess,runTimes{3,runInds==aniRunInds(runInd+1)}))
                batchRun = batchRun + 1;
                nSecsSess = [nSecsSess; length(regressorSessSz)];
                niisNameSess = [];
                regressorSessSz = [];
            end
            
            disp(['Skipping ' runID])
            continue
        end
        
        % load names of input niis
        niis = struct2cell(dir(fullfile(homedir,target,runID,['s' target '*.nii'])));
        niisName = niis(1,:)';
        niisName = cellfun(@(x) strcat(homedir,target,'\',runID,'\',x,',1'),niisName,'UniformOutput',false);
        nScans = length(niisName);
        
        szStarts = regressStr(:,1);
        szEnds = regressStr(:,3);
        szDur = szEnds - szStarts + 1;
        blStarts = szEnds + 1;
        blEnds = szStarts - 1;
        if min(szStarts) > 1
            blStarts = [1; blStarts];
        end
        if max(szEnds) < nScans
            blEnds = [blEnds; nScans];
        end
        blStarts(blStarts<1 | blStarts>nScans) = [];
        blEnds(blEnds<1 | blEnds>nScans) = [];
        blDur = blEnds - blStarts;
        
        % load TimeImg
        load([homedir target '\' runID '\sTimeImg.mat'])
        
        % reject frames 
        thisValidFrames = ones(length(thisFrames),1);
        rejInds = squeeze(sum(TimeImg,[1 2 3])) == 0;
        
        for szInd=1:size(regressStr,1)
            szStart = regressStr(szInd,1);
            szEnd = regressStr(szInd,3);
            
            thisRegressorSz(szStart:szEnd) = 1;
            thisRegressorBl(szStart:szEnd) = 0;
            
%             skip if this seizure is not included
            if sum(thisIncludedSzs==szInd)==0
                thisValidFrames(thisFrames>=szStart & thisFrames<=szEnd) = 0;
            end
        end
        
        [SPM] = setupSPM(bf,nScans,szStarts,szDur,blStarts,blDur);
        U = spm_get_ons(SPM,1);
        
        % convolve boxcar with hrf
        [X,Xn,Fc] = spm_Volterra(U, SPM.xBF.bf, SPM.xBF.Volterra);
        k = nScans;
        X = X((0:(k - 1))*16 + 8 + 32,:);
        
        for i = 1:length(Fc)
            if i<= numel(U) && ... % for Volterra kernels
                    (~isfield(U(i),'orth') || U(i).orth)
                p = ones(size(Fc(i).i));
            else
                p = Fc(i).p;
            end
            for j = 1:max(p)
                X(:,Fc(i).i(p==j)) = spm_orth(X(:,Fc(i).i(p==j)));
            end
        end
        
        % take valid time points of regressor
        thisValidFrames(rejInds) = 0;
        thisRegressorSz = X(:,1);
        thisRegressorBl = X(:,2);
        thisValidFrames = logical(thisValidFrames);
        thisRegressorSz = thisRegressorSz(thisValidFrames);
        thisRegressorBl = thisRegressorBl(thisValidFrames);
        regressorAniSz = [regressorAniSz; thisRegressorSz];
        regressorAniBl = [regressorAniBl; thisRegressorBl];
        regressorSessSz = [regressorSessSz; thisRegressorSz];
        
        % add run info to animal data
        szStartsAni = [szStartsAni; szStarts + timeInc];
        szDurAni = [szDurAni; szDur];
        blStartsAni = [blStartsAni; blStarts + timeInc];
        blDurAni = [blDurAni; blDur];
        
        niisName = niisName(thisValidFrames);
        niisNameAni = [niisNameAni; niisName];
        niisNameSess = [niisNameSess; niisName];
        
        timeInc = timeInc + nScans;
        
        if runInd == length(aniRunInds) || not(strcmp(thisSess,runTimes{3,runInds==aniRunInds(runInd+1)}))
            batchRun = batchRun + 1;
            nSecsSess = [nSecsSess; length(regressorSessSz)];
            niisNameSess = [];
            regressorSessSz = [];
        end
    end
    
    nSess = length(nSecsSess);
    sessLen = sum(nSecsSess);
    regressorsConstSess = zeros(sessLen,nSess);
    sessStart = 1;
    for sessInd=1:nSess
        sessEnd = sessStart+nSecsSess(sessInd)-1;
        regressorsConstSess(sessStart:sessEnd,sessInd) = 1;
        sessStart = sessEnd + 1;
    end
    
    
    matlabbatch{1}.spm.stats.fmri_spec.sess.scans = niisNameAni;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess.regress(1).name = 'sz*hrf';
    matlabbatch{1}.spm.stats.fmri_spec.sess.regress(1).val = regressorAniSz;
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = 128;    
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.none = true;
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {''};
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';
    matlabbatch{2}.spm.stats.fmri_est.spmmat = cellstr([savedir '\SPM.mat']);
    matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
    matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    matlabbatch{3}.spm.stats.con.spmmat = cellstr([savedir '\SPM.mat']);
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'sz*hrf';
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 0];
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.name = '-sz*hrf';
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.weights = [-1 0];
    matlabbatch{3}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.delete = 1;
    
    spm_jobman('run',matlabbatch)
end

%% plot
for aniInd=1:nrun
    thisAni = uniqueAnis(aniInd);
    indir = [homedir target '_SPM\fMRI_t' num2str(thisAni) '\'];
    savedir = [homedir target '_SPM\Plots\fMRI_t' num2str(thisAni) '\'];
    if not(exist(savedir,'dir'))
        mkdir(savedir);
    end
    plotTmap(indir,savedir,'spmT_0001.nii','t-maps_Nonthresholded')
end