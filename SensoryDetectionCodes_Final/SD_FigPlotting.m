%% Plotting SD Responses in Different States
% First load SD_Stim_correct_response_from_szstart.mat and
% SD_Stim_correct_response_totals.mat

pctresponse = nan(max(baseline_stim(:,1)),3);
resplatency = nan(size(pctresponse));
stimcount = nan(size(pctresponse));
states = {'Baseline','Seizure','Post-Seizure'};

% preliminary: for by-animal analyses, make sure each animal has a
% reasonable number (5?) of stimuli in each state
        
for animal = 1:max(baseline_stim(:,1)) % go through each animal
    stimcount(animal,1) = sum(baseline_stim(:,1)==animal);
    stimcount(animal,2) = sum(ictal_stim(:,1)==animal);
    stimcount(animal,3) = sum(postictal_stim(:,1)==animal);
    pctresponse(animal,1) = nanmean(baseline_stim(baseline_stim(:,1)==animal,3))*100;
    pctresponse(animal,2) = nanmean(ictal_stim(ictal_stim(:,1)==animal,4))*100;
    pctresponse(animal,3) = nanmean(postictal_stim(postictal_stim(:,1)==animal,3))*100;
    resplatency(animal,1) = nanmean(baseline_stim(baseline_stim(:,1)==animal,4))/1000;
    resplatency(animal,2) = nanmean(ictal_stim(ictal_stim(:,1)==animal,5))/1000;
    resplatency(animal,3) = nanmean(postictal_stim(postictal_stim(:,1)==animal,4))/1000;
end

pctresponse(any(stimcount<5,2),:) = []; % get rid of any animals that don't have an observation in each state
resplatency(any(stimcount<5,2),:) = []; % get rid of any animals that don't have an observation in each state

% Measure 1: percentage response rate

[pseiz,h] = signrank(pctresponse(:,1),pctresponse(:,2))
[ppost,h] = signrank(pctresponse(:,1),pctresponse(:,3))

% First scatter plot
figure;plotSpread(pctresponse,'showMM',4,'xNames',states)
ylabel('Percent Correct');title('Auditory Stimulus Response Rates by State')
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);
set(findall(h,'type','line'),'markerSize',50)
t = annotation('textbox','String',['Wilcoxon P values = ' num2str(pseiz) ' (seizure), ' num2str(ppost) ' (post-seizure); n = ' num2str(size(pctresponse,1)) ' animals']);
t.Position = [0.4 0.15 0.1 0.1];
set(gca,'TickDir','Out')

% Then bar chart
figure;barwitherr(nansem(pctresponse),nanmean(pctresponse))
ylabel('Percent Correct');title('Auditory Stimulus Response Rates by State')
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);
set(gca,'XTickLabel',states,'TickDir','Out')
t = annotation('textbox','String',['Wilcoxon P values = ' num2str(pseiz) ' (seizure), ' num2str(ppost) ' (post-seizure); n = ' num2str(size(pctresponse,1)) ' animals']);
t.Position = [0.3 0.8 0.1 0.1];
set(gca,'TickDir','Out')

% Measure 1: response latency

[pseiz,h] = signrank(resplatency(:,1),resplatency(:,2))
[ppost,h] = signrank(resplatency(:,1),resplatency(:,3))

% First scatter plot
figure;plotSpread(resplatency,'showMM',4,'xNames',states)
ylabel('Mean Response Time (s)');title('Auditory Stimulus Response Latencies by State')
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);
set(findall(h,'type','line'),'markerSize',50)
t = annotation('textbox','String',['Wilcoxon P values = ' num2str(pseiz) ' (seizure), ' num2str(ppost) ' (post-seizure); n = ' num2str(size(resplatency,1)) ' animals']);
t.Position = [0.4 0.5 0.1 0.1];
set(gca,'TickDir','Out')

% Then bar chart
figure;barwitherr(nansem(resplatency),nanmean(resplatency))
ylabel('Mean Response Time (s)');title('Auditory Stimulus Response Latencies by State')
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);
set(gca,'XTickLabel',states,'TickDir','Out')
t = annotation('textbox','String',['Wilcoxon P values = ' num2str(pseiz) ' (seizure), ' num2str(ppost) ' (post-seizure); n = ' num2str(size(resplatency,1)) ' animals']);
t.Position = [0.3 0.8 0.1 0.1];
set(gca,'TickDir','Out')