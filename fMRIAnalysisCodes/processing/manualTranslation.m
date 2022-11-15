function manualTranslation(target)
% Jingjing Li - 4/1/2021
%
% This function allows the user to interactively adjust the mean functional image
% to fit the template mask.
% Args:
%   - target(str): the name of the folder to take data from; e.g., 'snr-snrmrtrGad'

%% set constants
homedir = '/gpfs/ysm/project/blumenfeld/prv4/GAERS_fMRI/PV_Autumn_2018_/';
indir = [homedir target '/'];
prefix = 'mt'; 
outdir = [homedir prefix target '/'];
if not(exist(outdir,'dir'))
    mkdir(outdir);
end

%% I/O
% load or create manual translation parameters 
if isfile([outdir 'mtParams.mat'])
    load([outdir 'mtParams.mat'])
else
    mtParams = zeros(194,3); 
end
% create figure
figure('Position',[0 0 1200 900])

%% interactively decide translation parameters
for runInd=1:194
    thisIndir = [indir 'runID_' num2str(runInd) '/'];
    thisOutdir = [outdir 'runID_' num2str(runInd) '/'];
    % skip if translated image has already been saved
    if isfile([thisOutdir 'TimeImg.mat'])
        continue
    end
    if not(exist(thisOutdir,'dir'))
        mkdir(thisOutdir);
    end
    
    % load TimeImg
    load([thisIndir 'TimeImg.mat'])
    
    % interactively plot 
    while true
        % plot original map
        plotContourMap(TimeImg);
        
        % get direction input
        mtDir = input('Which direction do you want to move (x, y or z)? n for next.\n','s');
        if not(strcmp(mtDir,'x') || strcmp(mtDir,'y') || strcmp(mtDir,'z') || strcmp(mtDir,'n'))
            disp('Invalid direction. Try again.')
            continue
        elseif strcmp(mtDir,'n')
            break
        end
        
        % get the number of units to move 
        mtUnits = input('How many integer units do you want to move?\n');
        try 
            mtUnits = int16(mtUnits);
        catch
            disp('Invalid number. Try again.')
            continue
        end
        if mtUnits == 0
            continue
        end
        
        TimeImg_new = nan(size(TimeImg));
        if strcmp(mtDir,'x')
            if mtUnits<0
                TimeImg_new(1-mtUnits:end,:,:,:) = TimeImg(1:end+mtUnits,:,:,:);
            else
                TimeImg_new(1:end-mtUnits,:,:,:) = TimeImg(1+mtUnits:end,:,:,:);
            end
            mtParams(runInd,1) = mtParams(runInd,1) + mtUnits;
        elseif strcmp(mtDir,'y')
            if mtUnits>0
                TimeImg_new(:,1+mtUnits:end,:,:) = TimeImg(:,1:end-mtUnits,:,:);
            else
                TimeImg_new(:,1:end+mtUnits,:,:) = TimeImg(:,1-mtUnits:end,:,:);
            end
            mtParams(runInd,3) = mtParams(runInd,1) + mtUnits;
        elseif strcmp(mtDir,'z')
            if mtUnits>0
                TimeImg_new(:,:,1+mtUnits:end,:) = TimeImg(:,:,1:end-mtUnits,:);
            else
                TimeImg_new(:,:,1:end+mtUnits,:) = TimeImg(:,:,1-mtUnits:end,:);
            end
            mtParams(runInd,3) = mtParams(runInd,1) + mtUnits;
        end
        
        plotContourMap(TimeImg_new);
        
        TimeImg = TimeImg_new;
    end
    % save functional image and parameters
    save([outdir 'mtParams.mat'],'mtParams')
    save([thisOutdir 'TimeImg.mat'],'TimeImg')
    saveas(gcf,[thisOutdir 'contour.png'])
end

load([indir 'RunFileInfo.mat'])
save([outdir 'RunFileInfo.mat'],'runTimes')

% smoothMt;
end

