%% FREE LICKING STRUCTURE PLOTTING
% This script creates various plots from the free licking structure (1
% element per seizure) created by FreeLickSzProperties.

%% First, convert the Seizures structure to a table for easier access
FreeLickSeizTable = struct2table(Seizures);
FreeLickContTable = struct2table(StrtControls);
FreeLickEndContTable = struct2table(EndControls);

Animals = unique(FreeLickSeizTable.animal);
SzStrtLicksAnimal = nan(size(Animals,1),size(FreeLickSeizTable.LicksStart,2));
SzEndLicksAnimal = nan(size(Animals,1),size(FreeLickSeizTable.LicksEnd,2));

for animal = 1:size(Animals,1)
    AnimalSeizs = FreeLickSeizTable(strcmp(FreeLickSeizTable.animal, Animals{animal,1}),:);
    SzStrtLicksAnimal(animal,:) = nanmean(AnimalSeizs.LicksStart);
    SzEndLicksAnimal(animal,:) = nanmean(AnimalSeizs.LicksEnd);
end

figure;A=shadedErrorBar((-19.5:0.5:20),nanmean(SzStrtLicksAnimal),nansem(SzStrtLicksAnimal),'r');hold on;
plot([0,0],[0,0.5],'r--');
title('Lick rate around seizure start');ylabel('Licks per second');xlabel('Seconds from seizure start');
set(gca,'XTick',(-20:10:0),'XTickLabel',(-20:10:0),'TickDir','Out')
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);h = gcf; set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);
xlim([-115,120]);
t = annotation('textbox','String',['n = ' num2str(size(FreeLickSeizTable,1)) ' seizures from ' num2str(size(Animals,1)) ' animals ']);
t.Position = [0.6 0.6 0.1 0.1];

figure;A=shadedErrorBar((-19.5:0.5:20),nanmean(SzEndLicksAnimal),nansem(SzEndLicksAnimal),'r');hold on;
plot([0,0],[0,0.5],'r--');
title('Lick rate around seizure end');ylabel('Licks per second');xlabel('Seconds from seizure end');
set(gca,'XTick',(0:10:20),'XTickLabel',(0:10:20),'TickDir','Out')
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);h = gcf; set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);
xlim([-115,120]);
t = annotation('textbox','String',['n = ' num2str(size(FreeLickSeizTable,1)) ' seizures from ' num2str(size(Animals,1)) ' animals ']);
t.Position = [0.2 0.6 0.1 0.1];

% Calculating mean licks in state
states = {'Baseline','Seizure','Post-Seizure'};


szmeanlicks(:,1) = nanmean(SzStrtLicksAnimal(:,1:40),2);
szmeanlicks(:,2) = nanmean(SzStrtLicksAnimal(:,41:80),2);
szmeanlicks(:,3) = nanmean(SzEndLicksAnimal(:,41:80),2);

[pseiz,h] = signrank(szmeanlicks(:,1),szmeanlicks(:,2));
[ppost,h] = signrank(szmeanlicks(:,1),szmeanlicks(:,3));

figure;plotSpread(szmeanlicks,'showMM',4,'xNames',states,'distributionColors',{'y','r','b'})
ylabel('Licks per Second');
h = gcf; set(findall(h,'type','text'),'fontWeight','bold','fontSize',20);set(findall(h,'type','axes'),'fontWeight','bold','fontSize',20);
set(findall(h,'type','line'),'markerSize',50)
t = annotation('textbox','String',['Wilcoxon P values = ' num2str(pseiz) ' (seizure), ' num2str(ppost) ' (post-seizure); n = ' num2str(size(szmeanlicks,1)) ' animals']);
t.Position = [0.4 0.8 0.1 0.1];
set(gca,'TickDir','Out')
