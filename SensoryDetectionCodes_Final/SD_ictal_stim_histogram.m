% Aug 2021 Xinyuan Zheng
% plot the histograms of sz on/off time relative to stim

%%
% column information of input data:
%1. animal
%2. session
%3. if lick is within 10s of stim & before next seizure(1 yes 0 no)
%4. if lick is within the sz duration (1 yes 0 no)
%5. latency
%6. stim time from seizure end
%7. stim time from seizure start

% data = [ictal_stim]; 
% data(:,6) = data(:,6)./1000; %converts to seconds
% data(:,7) = data(:,7)./1000; 
% data(:,6) = floor(data(:,6)); %rounds stim stimes to nearest smaller integer
% data(:,7) = floor(data(:,7)); 

load('data')

% no need to remove these anymore
% data(any(data(:,7)>10,2),:) = []; %removes lines with stim >10s after sz
% data(any(data(:,6)<-10,2),:) = []; %removes lines with stim >10s before sz

%%
% stim time from seizure end
szoff = data(:,6);
% stim time from seizure start
szon = data(:,7); % szoff is negative and szon is postive
% compute sz duration
duration = -szoff+szon;

% select seizures that are longer than 5 seconds 
length_thres = 0; % length_thres = 0;

long_duration = duration(duration > length_thres);
long_szoff = szoff(duration > length_thres);
long_szon = szon(duration > length_thres);

%% plots
% histogram of sz duration
h1 = histogram(long_duration)
h1.BinWidth = 1;
title(['Seizure duration with Ictal Stimuli - removed sz < ',num2str(length_thres),'s']); 
xlabel('Time Bin (s)'); ylabel('Number of Seizures');
ylim([0 60]);
saveas(h1, ['duration_rm',num2str(length_thres),'s.jpg'])

% histogram of ictal stim relative to sz end
h2 = histogram(long_szoff)
h2.BinWidth = 1;
title(['Ictal Stimuli Relative to Sz End - removed sz < ',num2str(length_thres),'s']); 
xlabel('Relative to Sz End (s)'); ylabel('Number of Stimuli');
ylim([0 100]);xlim([-60 0]);
saveas(h2, ['szoff_rm',num2str(length_thres),'s'])

% histogram of ictal stim relative to sz start
h3 = histogram(long_szon)
h3.BinWidth = 1;
title(['Ictal Stimuli Relative to Sz start - removed sz < ',num2str(length_thres),'s']); 
xlabel('Relative to Sz Start (s)'); ylabel('Number of Stimuli');
ylim([0 60]); xlim([0 60]);
saveas(h3, ['szon_rm',num2str(length_thres),'s'])

%% normalized probablity graph
% szoff_table = array2table(tabulate(szoff),'VariableNames',{'Value','Count','Percent'})
% duration_table = array2table(tabulate(duration),'VariableNames',{'Value','Count','Percent'});

% 'normalize' the ictal stim time relative to sz end using sz duration
percent = zeros(max(abs(long_szoff)),2);
temp_index = 1;

for i = min(long_szoff):max(long_szoff) % szoff: from -55 to -1
    
    szoff_count = sum(long_szoff == i);
    duration_count = sum(long_duration >= -i); % duration longer than 55s to longer than 1s
    
    percent(temp_index, 1) = i; %  -55 to -1
    percent(temp_index, 2) = szoff_count/duration_count;
    temp_index = temp_index +1;
end

h4 = plot(percent(:,1), percent(:,2))
title(['Probability of Ictal Stimuli Relative to Sz End - removed sz < ',num2str(length_thres),'s']); 
xlabel('Time Relative to Sz End (s)'); ylabel('Probability');
ylim([0 0.5]);
saveas(h4, ['prob_rm',num2str(length_thres),'s'])
