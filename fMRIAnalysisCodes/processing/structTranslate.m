% 09/07/2019 Peter Vincent

function structTranslate(mode,templateFile,prefix,functFolder,structFolder,altStructFolder)
% This function is built to allow us to perform manual translation of
% structural images such that they all appear in the same part of the
% image, but without doing any interpolation

if nargin == 5
    isSingleView = true;
elseif nargin == 6
    isSingleView = false;
else
    error('Incorrect number of arguments.');
end

parentDir = '/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_/';
structDir = fullfile(parentDir,structFolder);
addpath('/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_/NIFTI_20110921');

functDir  = fullfile(parentDir,functFolder);
numFuncRuns= dir(fullfile(functDir,'runID_*'));
funcRuns = zeros(1,length(numFuncRuns));
if isempty(numFuncRuns); warning(['Could not locate any functional images at ' functDir]); end
for func = 1:length(numFuncRuns)
    curFunc = numFuncRuns(func).name;
    curFunc = strrep(curFunc,'runID_','');
    funcRuns(func) = str2double(curFunc);
end
funcRuns = sort(funcRuns);

templateLoc = dir(fullfile(parentDir,templateFile));
    if isempty(templateLoc)
        error(['Could not locate template file at ' fullfile(parentDir,templateFile)]); 
    end
    templateImg = load_nii(fullfile(templateLoc.folder,templateLoc.name));
    templateImg = templateImg.img;
    sizeImg = size(templateImg);

if strncmp(mode,'both',5) || strncmp(mode,'reg',4)
    
    if ~isSingleView
        altStructDir = fullfile(parentDir,altStructFolder);
    end
    mainSlice = 5;
    numStructs = dir(fullfile(structDir,'runID_*'));
    structRuns = zeros(1,length(numStructs));
    if isempty(numStructs); error(['Could not locate any structural images at ' structDir]); end
    for struc = 1:length(numStructs)
        curStruc = numStructs(struc).name;
        curStruc = strrep(curStruc,'runID_','');
        structRuns(struc) = str2double(curStruc);
    end
    structRuns = sort(structRuns);
    newStructLoc = fullfile(parentDir,[prefix structFolder]);
    if ~exist(newStructLoc,'dir')
        mkdir(newStructLoc)
    end
    funcMovement = zeros(length(numFuncRuns),2);
    figure('color','w');
    if isSingleView
        ax1 = axes();
        ax2 = nan();
    else
        ax1 = subplot(1,2,1);
        ax2 = subplot(1,2,2);
    end
    for struc = 1:length(numStructs)

        curStructRun = structRuns(struc);
        curStructName = ['runID_' num2str(curStructRun)];
        try
        newStructFolder = fullfile(newStructLoc,curStructName);
        mkdir(newStructFolder);
        if strcmp(curStructName,templateFile)
            copyfile(fullfile(templateLoc.folder,templateLoc.name),fullfile(newStructFolder,templateLoc.name));
            continue
        end
        if struc == length(numStructs)
            structRange = funcRuns(end);
        else
            structRange = structRuns(struc+1)-1;
        end
        funcRange = curStructRun:structRange;
        curStructImg = dir(fullfile(structDir,curStructName,'*nii'));
        if isempty(curStructImg); error(['Could not locate structural file at ' fullfile(structDir,curStructName)]); end
        curStructImg = load_nii(fullfile(curStructImg.folder,curStructImg.name));
        curStructHdr = curStructImg.hdr;
        curStructImg = curStructImg.img;
        imshowpair(templateImg(:,:,mainSlice), curStructImg(:,:,mainSlice),'Parent',ax1);
        addTitles(ax1,ax2,curStructName,isSingleView);
        if ~isSingleView
            curAltStructName = ['runID_' num2str(curStructRun)];
            curAltStructImg = dir(fullfile(altStructDir,curAltStructName,'*nii'));
            if isempty(curStructImg); error(['Could not locate structural file at ' fullfile(altStructDir,curAltStructName)]); end
            curAltStructImg = load_nii(fullfile(curAltStructImg.folder,curAltStructImg.name));
            curAltStructImg = curAltStructImg.img;
            imshowpair(templateImg(:,:,mainSlice),curAltStructImg(:,:,mainSlice),'Parent',ax2);
            addTitles(ax1,ax2,curStructName,isSingleView);
        end
        %%%% Now we register
        fprintf('To finish moving, press enter \n');
        fprintf(['To move in the x direction, enter x, then the number of '...
            'pixels to move by.  Similarly for y \n'])
        curMov = input('Shall we dance?   ','s');
        newStruct = curStructImg;
        movVector = zeros(1,2);
        while ~strncmp(curMov,'n',2)
            if strcmp(curMov,'x')
                try
                    curMov = input('How much are we moving today?   ');
                    numMov = curMov;
                catch 
                    fprintf('Input number, with only "-" in front if necessary \n');
                    curMov = input('How much are we moving today?    ');
                    numMov = curMov;
                end
                movVector(1,2) = movVector(1,2) + numMov;
                if numMov > 0
                    newStruct(:,numMov+1:end,:) = newStruct(:,1:end-numMov,:);
                    newStruct(:,1:numMov,:) = 0;
                elseif numMov < 0
                    numMov = abs(numMov);
                    newStruct(:,1:end-numMov,:) = newStruct(:,numMov+1:end,:);
                    newStruct(:,end-numMov+1:end,:) = 0;
                else
                    curMov = input("Didn't fancy it that time?  Let's go again.   ",'s');
                    continue
                end
                imshowpair(templateImg(:,:,mainSlice), newStruct(:,:,mainSlice),'Parent',ax1);
                addTitles(ax1,ax2,curStructName,isSingleView);
                curMov = input('Nice!  Shall we go again?    ','s');
            elseif strcmp(curMov,'y')
                try
                    curMov = input('How much are we moving today?   ');
                    % curMov = input('How much are we moving today?   ');oldFunc
                    numMov = curMov;
                catch 
                    fprintf('Input number, with only "-" in front if necessary \n');
                    curMov = input('We be shuffling today?    ');
                    numMov = curMov;
                end
                movVector(1,1) = movVector(1,1) + numMov;
                if numMov > 0
                    newStruct(numMov+1:end,:,:) = newStruct(1:end-numMov,:,:);
                    newStruct(1:numMov,:,:) = 0;
                elseif numMov < 0
                    numMov = abs(numMov);
                    newStruct(1:end-numMov,:,:) = newStruct(numMov+1:end,:,:);
                    newStruct(end-numMov+1:end,:,:) = 0;
                else
                    curMov = input('Sometimes beauty is still, but not this time.  Try again   ','s');
                    continue
                end
                imshowpair(templateImg(:,:,mainSlice), newStruct(:,:,mainSlice),'Parent',ax1);
                addTitles(ax1,ax2,curStructName,isSingleView);
                curMov = input('That was great!  Lets have another crack at it    ','s');
            elseif strcmp(curMov,'q')
                if 1 < mainSlice
                    mainSlice = mainSlice - 1;
                    imshowpair(templateImg(:,:,mainSlice), curStructImg(:,:,mainSlice),'Parent',ax1);
                    if ~isSingleView
                        imshowpair(templateImg(:,:,mainSlice), curAltStructImg(:,:,mainSlice),'Parent',ax2);
                    end
                    addTitles(ax1,ax2,curStructName,isSingleView);
                    curMov = input('Shall we dance?   ','s');
                    continue;
                else
                    curMov = input('No more slices in that direction. Shall we dance?   ','s');
                    continue;
                end
            elseif strcmp(curMov,'w')
                if mainSlice < size(templateImg,3) - 1
                    mainSlice = mainSlice + 1;
                    imshowpair(templateImg(:,:,mainSlice), curStructImg(:,:,mainSlice),'Parent',ax1);
                    if ~isSingleView
                        imshowpair(templateImg(:,:,mainSlice), curAltStructImg(:,:,mainSlice),'Parent',ax2);
                    end
                    addTitles(ax1,ax2,curStructName,isSingleView);
                    curMov = input('Shall we dance?   ','s');
                    continue;
                else
                    curMov = input('No more slices in that direction. Shall we dance?   ','s');
                    continue;
                end
            else
                fprintf('\n\n Fool of a Took!  The direction must be entered before the number \n\n');
                curMov = input('Lets dance   ','s');
            end
            fprintf([' \n So far, we have moved by x = ' num2str(movVector(1,2)) ...
                ' and by y = ' num2str(movVector(1,1)) ' \n']);
        end
        modStruct.img = newStruct;
        modStruct.hdr = curStructHdr;
        modStructName = strrep(curStructName,'.nii','_man_reg.nii');
        modPath = fullfile(newStructFolder,modStructName);
        numCurFuncs = length(funcRange);
        funcTrans  = repmat(movVector,numCurFuncs,1);
        funcMovement(funcRange,:) = funcTrans;
        save_nii(modStruct,modPath);
        catch e
            warning([curStructName ' has failed at line ' num2str(e.stack.line) '.\n'...
                'Error msg: ' e.message '\nMoving to the next image.']);
        end
    end
    saveTranslation = fullfile(newStructLoc,'StructuralTranslation.mat');
    save(saveTranslation,'funcMovement');
elseif ~strncmp(mode,'apply',6)
    error('Invalid ''mode'' parameter. Takes values ''both'', ''reg'', or ''apply''.');
end
    
%%% Now the structurals are done, edit the functionals
if strncmp(mode,'apply',6)
    saveTranslation = fullfile(parentDir,structFolder,'StructuralTranslation.mat');
    load(saveTranslation,'funcMovement');
end
if strncmp(mode,'both',5) || strncmp(mode,'apply',6)
    newFuncLoc   = fullfile(parentDir,[prefix functFolder]);
    if ~exist(newFuncLoc,'dir')
        mkdir(newFuncLoc)
    else
        warning(['New functional location: ' newFuncLoc ' , already exists']);
    end
    
    for run = 1:length(numFuncRuns)
        curRunNum = funcRuns(run);
        curRun = ['runID_' num2str(curRunNum)];
        funcFold = fullfile(parentDir,functFolder,curRun);
        numNii = dir(fullfile(funcFold,'*nii'));
        saveDir = fullfile(newFuncLoc,curRun);
        mkdir(saveDir);
        for nii = 1:length(numNii)
            curNii = load_nii(fullfile(numNii(nii).folder,numNii(nii).name));
            curNiiName = numNii(nii).name;
            curHdr = curNii.hdr;
            curImg = curNii.img;
            curImgSize = size(curImg);
            ratio = curImgSize(1:2) ./ sizeImg(1:2);
            ratio = unique(ratio);
            if length(ratio) > 1
                error('Functional and structural image sizes are not compatible')
            end
            curMov = round(funcMovement(run,:) .* ratio);
            editImg = zeros(curImgSize);
            if curMov(1,1) > 0
                editImg(curMov(1,1)+1:end,:,:) = curImg(1:end-curMov(1,1),:,:);
            elseif curMov(1,1) < 0
                movVal = abs(curMov(1,1));
                editImg(1:end-movVal,:,:) = curImg(movVal+1:end,:,:);
            else
                editImg = curImg;
            end
            if curMov(1,2) > 0
                editImg(:,curMov(1,2)+1:end,:) = editImg(:,1:end-curMov(1,2),:);
                editImg(:,1:curMov(1,2),:) = 0;
            elseif curMov(1,2) < 0
                movVal = abs(curMov(1,2));
                editImg(:,1:end-movVal,:) = editImg(:,movVal+1:end,:);
                editImg(:,end-movVal:end,:) = 0;
            end
            curFuncFile.hdr = curHdr;
            curFuncFile.img = editImg;
            saveFuncName = [prefix curNiiName];
            saveFuncPath = fullfile(saveDir,saveFuncName);
            save_nii(curFuncFile,saveFuncPath);
        end
    end
end
end

function addTitles(ax1,ax2,curStructName,isSingleView)
    title(ax1, [curStructName ' raw'],'Interpreter','none');
    if ~isSingleView
        title(ax2, 'registered', 'Interpreter','none');
    end
end