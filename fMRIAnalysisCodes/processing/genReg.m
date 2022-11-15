% 05/06/2019 Peter Vincent
% 03/02/2021 Jingjing Li - modified (adding exclusion)

function genReg(targetDir,exclude)
% This function is effectively the same code as in generateRegressors
% script but packaged into a function for efficient calling.
%
% Args:
%   - exclude: the percentage threshold of artifacts a seizure contains to
%   be excluded. Current threshold used is 20. If no argument is passed in,
%   no seizures will be excluded.
if nargin==1
    exclude=101;
end

parentDir = '/gpfs/ysm/project/prv4/GAERS_fMRI/PV_Autumn_2018_';
runTimesFile = fullfile(targetDir,'RunFileInfo.mat');
load(runTimesFile);
tempF = targetDir;
runNum = size(runTimes,2);
prefix = targetDir;
for run = 1:runNum
    curFolder = [tempF '/runID_' num2str(run)];
    seizureInf= [parentDir '/SeizureTimes/runID_' num2str(run) '/SeizureDatabase.xlsx'];
    curSeizInf= parse_seizuretimes(seizureInf);
    curDate   = runTimes{3,run};
    curRun    = runTimes{4,run}; runBreak = strfind(curRun,'_');
    curRun    = curRun(runBreak+1:end); 
    curRun    = regexprep(curRun,'^0*','');
    validRow  = 0;
    for check = 1:length(curSeizInf)
        posDates  = curSeizInf(check).date;
        if strcmp(posDates,curDate)
            posRun = curSeizInf(check).run;
            if strcmp(posRun,curRun)
                validRow = check;
                break
            end
        end
    end
    if validRow == 0
        regressStr = [];
        if nargin==1
            runFold = ['SeizureTimes_' prefix '/runID_' num2str(run)];
        else
            runFold = ['SeizureTimes_' prefix '_Exclude_' num2str(exclude) '/runID_' num2str(run)];
        end
        mkdir(runFold);
        save([runFold '/regressors.mat'],'regressStr');
        continue
    end
    mriStart = curSeizInf(check).mri_st;
    szStart  = curSeizInf(check).sz_st - mriStart;
    szEnd    = curSeizInf(check).sz_end - mriStart;
    szDur    = curSeizInf(check).sz_dur;
    % Now we cycle through each of these points and correct for any epoch
    % removal, using the runTimes{1,:} information
    toDelete = zeros(length(szStart),1);
    seizReject = ones(1,length(szStart)) * -1;
    endReject  = ones(1,length(szStart)) * -1;
    for ii = 1:length(szStart)
        % Establish the seizure start point
        curStart = round(szStart(ii));
        startInd = [];
        while isempty(startInd)
            seizReject(ii) = seizReject(ii) + 1;
            startInd = find(curStart == runTimes{1,run});
            if isempty(startInd)
                curStart = curStart + 1;
            end
            if curStart  > max(runTimes{1,run})
                startInd = 0;
                toDelete(ii) = 1;
                break
            end
            if curStart > 601
                error('RunTimes has gone all wonky.  Check it');
            end
        end
        curStart = startInd;
        if curStart <= 0
            curStart = 0;
        end
        szStart(ii) = curStart(1);
        % Establish the seizure end point
        curEnd = round(szEnd(ii));
        endInd = [];
        while isempty(endInd)
            endReject(ii) = endReject(ii) + 1;
            endInd = find(curEnd == runTimes{1,run});
            if isempty(endInd)
                curEnd = curEnd - 1;
            end
            if curEnd < 0
                endInd = 0;
                break;
            end
        end
        curEnd = endInd;
        if curEnd <= 0
            toDelete(ii) = 1;
        end
        szEnd(ii) = curEnd(1);
        if szStart(ii) >= szEnd(ii)
            toDelete(ii) = 1;
        elseif 100 * (1 - (szEnd(ii)-szStart(ii)) / szDur(ii)) > exclude
            toDelete(ii) = 1;
        end
    end
    toDelete = logical(toDelete);
    
    regressStr = zeros(length(szStart),5);
    regressStr(:,1) = szStart'; regressStr(:,3) = szEnd';
    regressStr(:,2) = (szEnd - szStart)'; regressStr(:,4) = seizReject';
    regressStr(:,5) = endReject';
    regressStr(toDelete,:) = [];
    if nargin==1
        runFold = ['SeizureTimes_' prefix '/runID_' num2str(run)];
    else
        runFold = ['SeizureTimes_' prefix '_Exclude_' num2str(exclude) '/runID_' num2str(run)];
    end
    mkdir(runFold);
    save([runFold '/regressors.mat'],'regressStr');
end