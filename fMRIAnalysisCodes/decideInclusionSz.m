function include = decideInclusionSz(TimeImg,VOIname,maxPercentChange,maxCoMChange)
% Seizure inclusion decision making

homedir =  'F:\xinyuan\GAERS\fmri\glmCodes\';
% load VOI mask
mask = load_nii([homedir 'VOI\' VOIname '.nii']);
mask = imresize(logical(mask.img),0.25);

include = true;

[xx,yy,zz] = ndgrid((1:64)-32.5,(1:32)-16.5,(1:12)-6.5);
COMx = zeros(1,size(TimeImg,4));
COMy = zeros(1,size(TimeImg,4));
COMz = zeros(1,size(TimeImg,4));
ROItc = zeros(1,size(TimeImg,4));
for tsInd=1:size(TimeImg,4)
    temp = TimeImg(:,:,:,tsInd);
    temp = temp(mask);
    ROItc(tsInd) = nanmean(temp);
    COMx(tsInd) = nansum(xx(mask).*temp)/nansum(temp);
    COMy(tsInd) = nansum(yy(mask).*temp)/nansum(temp);
    COMz(tsInd) = nansum(zz(mask).*temp)/nansum(temp);
end

% exclude based on percent change and CoM change
if (max(ROItc)-min(ROItc))/min(ROItc) > maxPercentChange*0.01
    include = false;
elseif max(COMx)-min(COMx) > maxCoMChange || max(COMy)-min(COMy) > maxCoMChange || max(COMz)-min(COMz) > maxCoMChange
    include = false;
end

end

