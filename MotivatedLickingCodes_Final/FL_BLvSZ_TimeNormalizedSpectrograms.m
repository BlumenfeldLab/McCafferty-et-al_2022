
%% B) Normalizing the Spectrogram using the times before 50s before seizure

%load FL_PreSzProps.mat
clear
load('/mnt/Cian_Data/GAERS_SeizureSeverity/Behavior/SpontaneousLicking/Analysis/ProcessedData/FL_PreSzProperties.mat');

% randSelect =  randi([0 1], 1,11407);
% randSelect = logical(randSelect);
% PreSzProperties = PreSzProperties(randSelect);

cutoffSeconds = -120; % number of seconds of preictal data needed
baselineSeconds = -100; % number of seconds being used as  baseline

cutoff = (cutoffSeconds + 120)*2 +1; % set to time in timecourse that there must be preictal data for.
onset = (0 + 120)*2 +1; % index where the ictal period begins
baselineidx = (baselineSeconds + 120)*2; % index where baseline ends
baselineTime = num2str(baselineSeconds);
totalTimeBins = 260;

workaround3d = nan(501,totalTimeBins,size(PreSzProperties,2));
% freqPowerAvgAll = nan(501,totalTimeBins-cutoff,size(PreSzProperties,2));
normalizedSpectAll = nan(501,totalTimeBins-cutoff+1,size(PreSzProperties,2));
tooShort = 0;
tooShortLog = ones(size(PreSzProperties,2),1); %this is a log of too short szs (too short = 0)
for r = 1:size(workaround3d,3)
    workaround3d(:,:,r) = (abs(real(PreSzProperties(r).Spect))); %takes out imaginary portion from DFT
    if sum(sum(isnan(workaround3d(:,cutoff:onset,r)))) == 0 %if there are no NaN...
        freqPowerAvgAll = nanmean(workaround3d(:,cutoff:baselineidx,r),2); %calculating baseline
        normalizedSpectAll(:,:,r) = workaround3d(:,cutoff:totalTimeBins,r)./freqPowerAvgAll;
    else
        tooShort = tooShort + 1;
        tooShortLog(r,1) = 0; %sets this sz as too short (1) in the log
    end
end

clear workaround3d;
tooShortLog = logical(tooShortLog); % 0's too short
normalizedSpect = normalizedSpectAll(:,:,tooShortLog);

meanNormalized = nanmean(normalizedSpect,3); %mean of unlogged

SpectTimes = PreSzProperties(1).Time;
SpectFrequency = PreSzProperties(1).Frequency;

figure;imagesc(SpectTimes(cutoff:totalTimeBins),SpectFrequency,meanNormalized);ylabel('Frequency (Hz)'); set(gca,'YDir','Normal'); 
title(['FL Spectrogram Prior to Seizure Normalized by Pre- ' baselineTime 's Power Averages']);
xlabel('Time (s)'); set(gca,'XTick',(SpectTimes(cutoff):10:SpectTimes(totalTimeBins)),'XTickLabel',(cutoffSeconds:10:10),'TickDir','Out');
colorbar('eastoutside'); c = colorbar; c.Label.String = ['Proportion of Power per Frequency Band Compared to Pre-' baselineTime 's Average'];
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);h = gcf; set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);
%%

%with more ticks
figure;imagesc(SpectTimes(cutoff:totalTimeBins),SpectFrequency,meanNormalized);ylabel('Frequency (Hz)'); set(gca,'YDir','Normal'); 
title(['Spectrogram Prior to Seizure Normalized by Pre- ' baselineTime 's Power Averages']);
xlabel('Time (s)'); set(gca,'XTick',(SpectTimes(cutoff):5:SpectTimes(totalTimeBins)),'XTickLabel',(cutoffSeconds:5:10),'TickDir','Out');
colorbar('eastoutside'); c = colorbar; c.Label.String = ['Proportion of Power per Frequency Band Compared to Pre-' baselineTime 's Average'];
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);h = gcf; set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);

% with only higher frequencies for contrast
figure;imagesc(SpectTimes(cutoff:totalTimeBins),SpectFrequency(126:end,1),meanNormalized(126:end,:));ylabel('Frequency (Hz)'); set(gca,'YDir','Normal'); 
title(['Spectrogram Prior to Seizure Normalized by Pre- ' baselineTime 's Power Averages']);
xlabel('Time (s)'); set(gca,'XTick',(SpectTimes(cutoff):5:SpectTimes(totalTimeBins)),'XTickLabel',(cutoffSeconds:5:10),'TickDir','Out');
colorbar('eastoutside'); c = colorbar; c.Label.String = ['Proportion of Power per Frequency Band Compared to Pre-' baselineTime 's Average'];
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);h = gcf; set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);


%begin finding power in larger bands

idx = 0; %represents time points
szcount = 0; %represents seizures
%first, get the sum of power in each band for each seizure at each time point
for n_seizure = 1:size(normalizedSpect,3) %goes through each seizure
    for r_times = 1:size(normalizedSpect,2) %goes through each time
        idx = idx + 1;
        if sum(isnan(normalizedSpect(2:5,r_times,n_seizure))) == 0 %delta 1-4
            SZ_deltapowersum(idx,1) = nansum(normalizedSpect(2:5,r_times,n_seizure)); %puts sum of power in 1-4Hz band at that time point
            SZ_deltapowersum(idx,1) = SZ_deltapowersum(idx,1)/size(normalizedSpect(2:5,r_times,n_seizure),1);
        else
            SZ_deltapowersum(idx,1) = NaN;
        end
        if sum(isnan(normalizedSpect(9:14,r_times,n_seizure))) == 0 %alpha 8-13
            SZ_alphapowersum(idx,1) = nansum(normalizedSpect(9:14,r_times,n_seizure)); %puts sum of power in 4-10Hz band at that time point
            SZ_alphapowersum(idx,1) = SZ_alphapowersum(idx,1)/size(normalizedSpect(9:14,r_times,n_seizure),1);
        else
            SZ_alphapowersum(idx,1) = NaN;
        end
        if sum(isnan(normalizedSpect(5:11,r_times,n_seizure))) == 0 %theta 4-10
            SZ_thetapowersum(idx,1) = nansum(normalizedSpect(5:11,r_times,n_seizure)); %puts sum of power in 4-10Hz band at that time point
            SZ_thetapowersum(idx,1) = SZ_thetapowersum(idx,1)/size(normalizedSpect(5:11,r_times,n_seizure),1);
        else
            SZ_thetapowersum(idx,1) = NaN;
        end
        if sum(isnan(normalizedSpect(13:31,r_times,n_seizure))) == 0 %beta 12-30
            SZ_betapowersum(idx,1) = nansum(normalizedSpect(31:31,r_times,n_seizure)); %puts sum of power in 12-30Hz band at that time point
            SZ_betapowersum(idx,1) = SZ_betapowersum(idx,1)/size(normalizedSpect(31:31,r_times,n_seizure),1);
        else
            SZ_betapowersum(idx,1) = NaN;
        end
        if sum(isnan(normalizedSpect(31:56,r_times,n_seizure))) == 0 %gamma1 30-55
            SZ_gamma1powersum(idx,1) = nansum(normalizedSpect(31:56,r_times,n_seizure)); %puts sum of power in 30-60Hz band at that time point
            SZ_gamma1powersum(idx,1) = SZ_gamma1powersum(idx,1)/size(normalizedSpect(31:56,r_times,n_seizure),1);
        else
            SZ_gamma1powersum(idx,1) = NaN;
        end
        if nansum(isnan(normalizedSpect(66:501,r_times,n_seizure))) == 0 %gamma2 65-500
            SZ_gamma2powersum(idx,1) = nansum(normalizedSpect(66:501,r_times,n_seizure)); %puts sum of power in 65-500Hz band at that time point
            SZ_gamma2powersum(idx,1) = SZ_gamma2powersum(idx,1)/size(normalizedSpect(66:501,r_times,n_seizure),1);
        else
            SZ_gamma2powersum(idx,1) = NaN;
        end
        if nansum(isnan(normalizedSpect(151:501,r_times,n_seizure))) == 0 %gamma2 150-500
            SZ_gamma150powersum(idx,1) = nansum(normalizedSpect(151:501,r_times,n_seizure)); %puts sum of power in 150-500Hz band at that time point
            SZ_gamma150powersum(idx,1) = SZ_gamma150powersum(idx,1)/size(normalizedSpect(151:501,r_times,n_seizure),1);
        else
            SZ_gamma150powersum(idx,1) = NaN;
        end
        if sum(isnan(normalizedSpect(8:14,r_times,n_seizure))) == 0 %seizure band 7-13
            SZ_szpowersum(idx,1) = nansum(normalizedSpect(8:14,r_times,n_seizure)); %puts sum of power in 7-13Hz band at that time point
            SZ_szpowersum(idx,1) = SZ_szpowersum(idx,1)/size(normalizedSpect(8:14,r_times,n_seizure),1);
        else
            SZ_szpowersum(idx,1) = NaN;
        end
    end 
    idx = 0;
    szcount = szcount + 1;
    %for the following arrays, each column is a seizure, each row is a different time point
    SZ_Deltapower(:,szcount) = SZ_deltapowersum;
    SZ_Thetapower(:,szcount) = SZ_thetapowersum;
    SZ_Alphapower(:,szcount) = SZ_alphapowersum;
    SZ_Betapower(:,szcount) = SZ_betapowersum;
    SZ_Gamma1power(:,szcount) = SZ_gamma1powersum;
    SZ_Gamma2power(:,szcount) = SZ_gamma2powersum;
    SZ_Gamma150power(:,szcount) = SZ_gamma150powersum;
    SZ_szpower(:,szcount) = SZ_szpowersum;
    
end

%for each row (time point) of each power band, get SEM
for r_times = 1:size(normalizedSpect,2)
    SZ_SEM_D(1,r_times) = nanstd(SZ_Deltapower(r_times,:)) / (sqrt (size(normalizedSpect,3))); %SEM
    SZ_SEM_T(1,r_times) = nanstd(SZ_Thetapower(r_times,:)) / (sqrt (size(normalizedSpect,3))); 
    SZ_SEM_A(1,r_times) = nanstd(SZ_Alphapower(r_times,:)) / (sqrt (size(normalizedSpect,3)));
    SZ_SEM_B(1,r_times) = nanstd(SZ_Betapower(r_times,:)) / (sqrt (size(normalizedSpect,3))); 
    SZ_SEM_G1(1,r_times) = nanstd(SZ_Gamma1power(r_times,:)) / (sqrt (size(normalizedSpect,3))); 
    SZ_SEM_G2(1,r_times) = nanstd(SZ_Gamma2power(r_times,:)) / (sqrt (size(normalizedSpect,3))); 
    SZ_SEM_G150(1,r_times) = nanstd(SZ_Gamma150power(r_times,:)) / (sqrt (size(normalizedSpect,3)));
    SZ_SEM_sz(1,r_times) = nanstd(SZ_szpower(r_times,:)) / (sqrt (size(normalizedSpect,3))); 
end

cd('/mnt/Cian_Data/GAERS_SeizureSeverity/Behavior/SensoryDetection/Analysis/ProcessedData/');
save('Bandpower_pre10sNormalized30sCut.mat', 'SpectTimes', 'SpectFrequency', 'SZ_Deltapower','SZ_SEM_D','SZ_Thetapower', 'SZ_SEM_T','SZ_Alphapower','SZ_SEM_A','SZ_Betapower','SZ_SEM_B','SZ_Gamma1power','SZ_SEM_G1','SZ_Gamma2power','SZ_SEM_G2','SZ_szpower','SZ_SEM_sz', 'SZ_Gamma150power', 'SZ_SEM_G150');

% load the above saved file on computer (not cluster)
load('\\MWMJ03EEUL\GAERS_SeizureSeverity\Behavior\SensoryDetection\Analysis\ProcessedData\Bandpower_pre10sNormalized30sCut.mat');

cutoffSeconds = -120; % number of seconds of preictal data needed
baselineSeconds = -50; % number of seconds being used as  baseline

cutoff = (cutoffSeconds + 120)*2; % set to time in timecourse that there must be preictal data for.
onset = 241; % index where the ictal period begins
baselineidx = (baselineSeconds + 120)*2; % index where baseline ends
baselineTime = num2str(baselineSeconds);
totalTimeBins = 260;

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),nanmean(SZ_Deltapower,2),SZ_SEM_D,'-r')
title(['Proportion of Power in Delta Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in Delta Band Compared to Pre- ' baselineTime 's Average']);
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out'); hold on;
plot([120,120],[0.9,1.6],'k--');

% %-60 to 0s
% figure; shadedErrorBar(SpectTimes(cutoff+1:240),nanmean(SZ_Deltapower(1:120,:),2),SZ_SEM_D(:,1:120),'-r')
% title(['Proportion of Power in Delta Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in Delta Band Compared to Pre- ' baselineTime 's Average']);
% xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:120),'XTickLabel',(-120:5:0),'TickDir','Out'); hold on;
% 
% %-60 to -10s
% figure; shadedErrorBar(SpectTimes(cutoff+1:220),nanmean(SZ_Deltapower(1:100,:),2),SZ_SEM_D(:,1:100),'-r')
% title(['Proportion of Power in Delta Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in Delta Band Compared to Pre- ' baselineTime 's Average']);
% xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:120),'XTickLabel',(-120:5:-10),'TickDir','Out'); hold on;

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),nanmean(SZ_Alphapower,2),SZ_SEM_A,'-r')
title(['Proportion of Power in Alpha Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in Alpha Band Compared to Pre- ' baselineTime 's Average']);
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');
hold on; plot([120,120],[0.9,6.5],'k--');

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),nanmean(SZ_Thetapower,2),SZ_SEM_T,'-r')
title(['Proportion of Power in Theta Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in Theta Band Compared to Pre- ' baselineTime 's Average']);
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');
hold on; plot([120,120],[0.9,6.5],'k--');

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),nanmean(SZ_Betapower,2),SZ_SEM_B,'-r')
title(['Proportion of Power in Beta Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in Beta Band Compared to Pre- ' baselineTime 's Average']);
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');
hold on; plot([120,120],[0.9,6.5],'k--');

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),nanmean(SZ_Gamma1power,2),SZ_SEM_G1,'-r')
title(['Proportion of Power in Gamma1 Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in Gamma1 Band Compared to Pre- ' baselineTime 's Average']);
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');
hold on; plot([120,120],[0.9,3.5],'k--');

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),nanmean(SZ_Gamma2power,2),SZ_SEM_G2,'-r')
title(['Proportion of Power in Gamma2 Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in Gamma2 Band Compared to Pre- ' baselineTime 's Average']);
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');
hold on; plot([120,120],[0.7,1.2],'k--');

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),nanmean(SZ_Gamma150power,2),SZ_SEM_G150,'-r')
title(['Proportion of Power in 150+ Hz Band Compared to Pre- ' baselineTime 's Average']);ylabel(['Proportion of Power in 150+ Band Compared to Pre- ' baselineTime 's Average']);
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');
hold on; plot([120,120],[0.7,1.2],'k--');
scatter(SpectTimes(cutoff+1:totalTimeBins),nanmean(SZ_Gamma150power,2));

% % theta-delta ratio
% SZ_TDRatios = SZ_Thetapower./SZ_Deltapower; %beta-delta ratio for each seizure and each time point
% SZ_TDRatioMean = nanmean(SZ_TDRatios,2); %mean beta-delta ratio per time point for all seizures
% 
% %SEM
% for r_times = 1:size(SpectTimes(cutoff:totalTimeBins),2)
%     SZ_SEM_BD(1,r_times) = nanstd(SZ_TDRatios(r_times,:)) / (sqrt (size(SZ_TDRatios,1))); %SEM for SZ ratios
% end
% 
% figure; shadedErrorBar(SpectTimes(cutoff:totalTimeBins),SZ_TDRatioMean,SZ_SEM_BD,'-r')
% title('Theta-Delta Ratio');ylabel('Theta to Delta Power Ratio');
% xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');
% 
% beta-delta ratio
SZ_BDRatios = SZ_Betapower./SZ_Deltapower; %beta-delta ratio for each seizure and each time point
SZ_BDRatioMean = nanmean(SZ_BDRatios,2); %mean beta-delta ratio per time point for all seizures

%SEM
for r_times = 1:size(SpectTimes(cutoff+1:totalTimeBins),2)
    SZ_SEM_BD(1,r_times) = nanstd(SZ_BDRatios(r_times,:)) / (sqrt (size(SZ_BDRatios,1))); %SEM for SZ ratios
end

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),SZ_BDRatioMean,SZ_SEM_BD,'-r')
title('Beta-Delta Ratio');ylabel('Beta to Delta Power Ratio');
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');

% % theta-beta ratio
% SZ_TBRatios = SZ_szpower./SZ_Deltapower; %beta-delta ratio for each seizure and each time point
% SZ_TBRatioMean = nanmean(SZ_TBRatios,2); %mean beta-delta ratio per time point for all seizures
% 
% %SEM
% for r_times = 1:size(SpectTimes(cutoff:totalTimeBins),2)
%     SZ_SEM_TB(1,r_times) = nanstd(SZ_TBRatios(r_times,:)) / (sqrt (size(SZ_TBRatios,1))); %SEM for SZ ratios
% end
% 
% figure; shadedErrorBar(SpectTimes(cutoff:totalTimeBins),SZ_TBRatioMean,SZ_SEM_TB,'-r')
% title('Theta-Beta Ratio');ylabel('Theta to Beta Power Ratio');
% xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');

% alpha-delta ratio
SZ_ADRatios = SZ_Alphapower./SZ_Deltapower; %alpha-delta ratio for each seizure and each time point
SZ_ADRatioMean = nanmean(SZ_ADRatios,2); %mean beta-delta ratio per time point for all seizures

%SEM
for r_times = 1:size(SpectTimes(cutoff+1:totalTimeBins),2)
    SZ_SEM_AD(1,r_times) = nanstd(SZ_ADRatios(r_times,:)) / (sqrt (size(SZ_ADRatios,1))); %SEM for SZ ratios
end

figure; shadedErrorBar(SpectTimes(cutoff+1:totalTimeBins),SZ_ADRatioMean,SZ_SEM_AD,'-r')
title('Alpha-Delta Ratio');ylabel('Alpha to Delta Power Ratio');
xlabel('Time Relative to Seizure Start (s)');set(gca,'XTick',(0:5:130),'XTickLabel',(-120:5:10),'TickDir','Out');
