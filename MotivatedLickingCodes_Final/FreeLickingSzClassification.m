%% First,convert the Seizures structure to a table for easier access

clc; clear; close all;
%addpath('HelperFunctions')
dataPath = '/Users/jacobprince/Desktop/BlumenfeldLab_Spring19/SzClassification';

FreeLickSeizTable = load(fullfile(dataPath, 'Seizures'));
FreeLickSeizTable = struct2table(FreeLickSeizTable.Seizures);

methodAccuracies = [];

visualizeSzCounts = 0;

%% Get labels of spared and impaired seizures

ictalOnsetIdx = 41; 

quickBhvTest = nanmean([FreeLickSeizTable.LicksStart(:,ictalOnsetIdx:end)],2); % find those few seizures that have some actual licking
impairedSzLog = quickBhvTest==0;
sparedSzLog =   quickBhvTest>0; % distinguishing spared and impaired

%% Basic data visualization

animalIDs = unique(FreeLickSeizTable.animal,'stable');
nAnimals = length(animalIDs);

sparedTable =   FreeLickSeizTable(sparedSzLog,:);
impairedTable = FreeLickSeizTable(impairedSzLog,:);

animalSzCounts = zeros(2,nAnimals);

for i = 1:nAnimals
    
    nSpared =   size(sparedTable(strcmp(sparedTable.animal, animalIDs{i}),:),1);
    nImpaired = size(impairedTable(strcmp(impairedTable.animal, animalIDs{i}),:),1);
    
    animalSzCounts(:,i) = [nImpaired; nSpared];
    
end
    
if visualizeSzCounts
    
    figure('Color', [1 1 1], 'Position', [100 100 1000 800]);
   
    bar(1:nAnimals, animalSzCounts')
    legend({'impaired', 'spared'}, 'Location', 'best')
    xticks(1:nAnimals)
    xticklabels(animalIDs), xtickangle(90), title('seizure count by animal')
    set(gca,'FontSize',16), box('off')
    
end

%% Classify spared vs. impaired using lick rate
% 
% X = [FreeLickSeizTable.LicksStart FreeLickSeizTable.LicksEnd];
% y = sparedSzLog;
% 
% methodAccuracies(1) = crossValPredict(X,y,0.10,10);

%% Classify spared vs. impaired using seizure duration

X = FreeLickSeizTable.Duration/1000; 
y = sparedSzLog;

[acc1,SD1] = crossValPredictBalanced(X,y,0.2,50);

%% Next, seizure vRMS amplitudes

% First express vRMS relative to baselines
vRMSStartNorm = zeros(size(FreeLickSeizTable,1),size(FreeLickSeizTable.vRMSStart,2));
vRMSEndNorm =   zeros(size(FreeLickSeizTable,1),size(FreeLickSeizTable.vRMSEnd,2));

for r_seizure = 1:size(FreeLickSeizTable,1)
    vRMSStartNorm(r_seizure,:) = FreeLickSeizTable(r_seizure,:).vRMSStart./nanmean(FreeLickSeizTable(r_seizure,:).vRMSStart(:,1:20));
    vRMSEndNorm(r_seizure,:) = FreeLickSeizTable(r_seizure,:).vRMSEnd./nanmean(FreeLickSeizTable(r_seizure,:).vRMSEnd(:,61:80));
end
FreeLickSeizTable.vRMSStartNorm = vRMSStartNorm;
FreeLickSeizTable.vRMSEndNorm = vRMSEndNorm;

X = vRMSStartNorm; X(isnan(X)) = 0; 
%figure; imagesc(X); colorbar;

[acc2,SD2] = crossValPredictBalanced(X,y,0.2,50);

X = vRMSEndNorm; X(isnan(X)) = 0; 
%figure; imagesc(X); colorbar;

[acc3,SD3] = crossValPredictBalanced(X,y,0.2,50);

X = [vRMSStartNorm vRMSEndNorm]; X(isnan(X)) = 0; 
%figure; imagesc(X); colorbar;

[acc4,SD4] = crossValPredictBalanced(X,y,0.2,50);



%% Next, spectral properties of seizures

% % First get all spectrograms into formats in which we can manipulate (take means, etc, of) them
% workaround3dstart = nan(100,80,size(FreeLickSeizTable,1)); % create a 3d array, in which the 3rd dimension is seizure (1 and 2 are time and frequency)
% workaround3dend = nan(100,80,size(FreeLickSeizTable,1)); % create a 3d array, in which the 3rd dimension is seizure (1 and 2 are time and frequency)
% for r_seizure = 1:size(FreeLickSeizTable,1)
%     workaround3dstart(:,:,r_seizure) = abs(Seizures(r_seizure).SpecStart); % this sort of indexing seems to work only on a structure and not on a table
%     timewisepowersum = repmat(nansum(workaround3dstart(:,:,r_seizure)),size(workaround3dstart,1),1); % find the sum of powers across all frequencies at each time point
%     SpectStartNorm(:,:,r_seizure) = workaround3dstart(:,:,r_seizure)./timewisepowersum; % and divide each individual frequency by this sum
%     workaround3dend(:,:,r_seizure) = abs(Seizures(r_seizure).SpecEnd); % this sort of indexing seems to work only on a structure and not on a table
%     timewisepowersum = repmat(nansum(workaround3dend(:,:,r_seizure)),size(workaround3dend,1),1);
%     SpectEndNorm(:,:,r_seizure) = workaround3dend(:,:,r_seizure)./timewisepowersum;
% end
% 
% X = SpectStartNorm;
% x1 = nanmean(X(:,:,y==1),3);
% x2 = nanmean(X(:,:,y==0),3);
% time = 0.25:0.5:21.75;
% freq = 1:50;
% figure;
% subplot(121)
% imagesc(time, freq, 10*log10(x1));
% set(gca,'YDir','Normal');
% title('spared')
% subplot(122)
% imagesc(time, freq, 10*log10(x2));
% set(gca,'YDir','Normal');
% title('impaired')
% 
% X2d = reshape(X, [size(X,1)*size(X,2), size(X,3)]);

SpectStartNorm = []; SpartEndNorm = [];
% First get all spectrograms into formats in which we can manipulate (take means, etc, of) them
workaround3dstart = nan(100,80,size(FreeLickSeizTable,1)); % create a 3d array, in which the 3rd dimension is seizure (1 and 2 are time and frequency)
workaround3dend = nan(100,80,size(FreeLickSeizTable,1)); % create a 3d array, in which the 3rd dimension is seizure (1 and 2 are time and frequency)
for r_seizure = 1:size(FreeLickSeizTable,1)
    workaround3dstart(:,:,r_seizure) = abs(cell2mat(FreeLickSeizTable(r_seizure,:).SpecStart)); % this sort of indexing seems to work only on a structure and not on a table
    timewisepowersum = repmat(nansum(workaround3dstart(:,:,r_seizure)),size(workaround3dstart,1),1); % find the sum of powers across all frequencies at each time point
    SpectStartNorm(:,:,r_seizure) = workaround3dstart(:,:,r_seizure)./timewisepowersum; % and divide each individual frequency by this sum
    workaround3dend(:,:,r_seizure) = abs(cell2mat(FreeLickSeizTable(r_seizure,:).SpecEnd)); % this sort of indexing seems to work only on a structure and not on a table
    timewisepowersum = repmat(nansum(workaround3dend(:,:,r_seizure)),size(workaround3dend,1),1);
    SpectEndNorm(:,:,r_seizure) = workaround3dend(:,:,r_seizure)./timewisepowersum;
end


SpectStartData = 10*log10(SpectStartNorm(1:50,1:73,:));

%figure; subplot(121),imagesc(flipud(nanmean(SpectStartData(:,:,sparedSzLog),3))); caxis([-32,-8]);
%subplot(122),imagesc(flipud(nanmean(SpectStartData(:,:,~sparedSzLog),3))); caxis([-32,-8]);

SpectEndData = 10*log10(SpectEndNorm(1:50,8:end,:));

figure; subplot(121),imagesc(flipud(nanmean(SpectEndData(:,:,sparedSzLog),3))); caxis([-32,-8]);
subplot(122),imagesc(flipud(nanmean(SpectEndData(:,:,~sparedSzLog),3))); caxis([-32,-8]);

SpectStartData(isnan(SpectStartData)) = 0;
SpectEndData(isnan(SpectEndData)) = 0;

X = permute(reshape(SpectStartData, [50*73 592]), [2 1]);
[acc4,SD4] = crossValPredictBalanced(X,y,0.2,50);

X = permute(reshape(SpectEndData, [50*73 592]), [2 1]);
[acc4,SD4] = crossValPredictBalanced(X,y,0.2,50);

X = permute(reshape([SpectStartData SpectEndData], [50*(73*2) 592]), [2 1]);
[acc4,SD4] = crossValPredictBalanced(X,y,0.2,50);


%% Dominant SWD frequency

SpectStartNorm = []; SpartEndNorm = [];
% First get all spectrograms into formats in which we can manipulate (take means, etc, of) them
workaround3dstart = nan(100,80,size(FreeLickSeizTable,1)); % create a 3d array, in which the 3rd dimension is seizure (1 and 2 are time and frequency)
workaround3dend = nan(100,80,size(FreeLickSeizTable,1)); % create a 3d array, in which the 3rd dimension is seizure (1 and 2 are time and frequency)
for r_seizure = 1:size(FreeLickSeizTable,1)
    workaround3dstart(:,:,r_seizure) = abs(cell2mat(FreeLickSeizTable(r_seizure,:).SpecStart)); % this sort of indexing seems to work only on a structure and not on a table
    timewisepowersum = repmat(nansum(workaround3dstart(:,:,r_seizure)),size(workaround3dstart,1),1); % find the sum of powers across all frequencies at each time point
    SpectStartNorm(:,:,r_seizure) = workaround3dstart(:,:,r_seizure)./timewisepowersum; % and divide each individual frequency by this sum
    workaround3dend(:,:,r_seizure) = abs(cell2mat(FreeLickSeizTable(r_seizure,:).SpecEnd)); % this sort of indexing seems to work only on a structure and not on a table
    timewisepowersum = repmat(nansum(workaround3dend(:,:,r_seizure)),size(workaround3dend,1),1);
    SpectEndNorm(:,:,r_seizure) = workaround3dend(:,:,r_seizure)./timewisepowersum;
end

[~,strongestwavfreqlocstrt] = max(SpectStartNorm(5:9,:,:),[],1); % find the location of the highest power between 5 and 9 Hz (wave frequency) for each seizure
strongestwavfreqlocstrt = strongestwavfreqlocstrt+4; % add 4 because index 1 corresponds to 5 Hz
FreeLickSeizTable.WaveFreqStart = permute(strongestwavfreqlocstrt,[3,2,1]); % slot these in to the table as a new variable
[~,strongestwavfreqlocend] = max(SpectEndNorm(5:9,:,:),[],1); % find the location of the highest power between 5 and 9 Hz (wave frequency) for each seizure
strongestwavfreqlocend = strongestwavfreqlocend+4; % add 4 because index 1 corresponds to 5 Hz
FreeLickSeizTable.WaveFreqEnd = permute(strongestwavfreqlocend,[3,2,1]); % slot these in to the table as a new variable

WaveFreqDataStart = FreeLickSeizTable.WaveFreqStart;
WaveFreqDataEnd = FreeLickSeizTable.WaveFreqEnd;

figure; subplot(121),plot(nanmean(WaveFreqDataStart(sparedSzLog,:))); hold on; plot(nanmean(WaveFreqDataStart(~sparedSzLog,:)));
subplot(122),plot(nanmean(WaveFreqDataEnd(sparedSzLog,:))); hold on; plot(nanmean(WaveFreqDataEnd(~sparedSzLog,:)));

X = WaveFreqDataStart;
[acc4,SD4] = crossValPredictBalanced(X,y,0.2,50);

X = WaveFreqDataEnd;
[acc4,SD4] = crossValPredictBalanced(X,y,0.2,50);


%% Ensemble of duration and spectral info

%X = [FreeLickSeizTable.Duration/1000 X2d']; 
%methodAccuracies(5) = crossValPredict(X,y,0.1,50);









