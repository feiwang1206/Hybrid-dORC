function [mean_dw0, mean_dphi, dw, dphi, dw0, dphi0, mean_dw_part, dw_part] = DORK_frequency_pulseq(filename,idx_k0,Ni)

% Calculates the frequency shift over the timeseries of a MREG scan. This
% can be used for scanner drift, respiration etc. correction

% Input: filename, or just the MID
% Output: array of frequencies for all timeframes

% Feb 2012, Jakob Asslaender, Pierre LeVan
% Sept 2015, Implementation of Pfeuffer 2002 + multi-shot: Bruno Riemenschneider

if isempty(Ni)
    Ni=1;
end

if isempty(idx_k0)
    idx_k0 = 6954; %ja_sos
end

if isnumeric(filename)
    mid = filename;
    filename = filenameByMID(mid);
end

[pathname,filename_no_ext,ext] = fileparts(filename);
if isempty(pathname)
    pathname = '';
end

twx_obj_name = [filename_no_ext, '_twx_obj.mat'];

if ~exist([pathname,'/twx_obj'],'dir')
    mkdir([pathname,'/twx_obj']);
end

if ~exist(fullfile([pathname,'/twx_obj'],twx_obj_name),'file')
    twx = mapVBVD(filename);
    if iscell(twx)
        twx = twx{end};
    end
    save(fullfile([pathname,'/twx_obj'],twx_obj_name),'twx');
else
    load(fullfile([pathname,'/twx_obj'],twx_obj_name));
end

%%%%%%%%%%%%%%%%%%%%%%%
%meas_header = readHeader(filename);
%if ~isfield(meas_header, 'DummyScans')
%    meas_header.DummyScans = 0;
%end
% te = cell2mat(twx.hdr.MeasYaps.alTE)/1e6;

%x_TN = loadRawData(filename,1,[],[],[],0,[],[],[],meas_header.IDEA_version);
%x_TN = permute(squeeze(x_TN),[2 3 1 4 5]);
%x_TN = reshape(x_TN,size(x_TN,1),size(x_TN,2),size(x_TN,3)*size(x_TN,4)*size(x_TN,5));
data_TN = squeeze(twx.image(30,:,:));
% x_TN = reshape(x_TN,size(x_TN,1),size(x_TN,2),size(x_TN,3)*size(x_TN,4));

%x_TE = loadRawData(filename,1,[],[],[],fix(idx_k0/500),[],[],[],meas_header.IDEA_version);
%x_TE = permute(squeeze(x_TE),[2 3 1 4 5]);
%x_TE = reshape(x_TE,size(x_TE,1),size(x_TE,2),size(x_TE,3)*size(x_TE,4)*size(x_TE,5));
data_TE = squeeze(twx.image(idx_k0,:,:));
% x_TE = reshape(x_TE,size(x_TE,1),size(x_TE,2),size(x_TE,3)*size(x_TE,4));


% Little hack: If dummy scans exist, it fakes them to make sure, the first
% shot including dummy scans is assumed to be in phase.
%dummy = zeros(size(x_TN, 1), size(x_TN, 2), meas_header.DummyScans);
%x_TN = cat(3, dummy, x_TN);

%dummy = zeros(size(x_TE, 1), size(x_TE, 2), meas_header.DummyScans);
%x_TE = cat(3, dummy, x_TE);


%data_TN = squeeze(mean(x_TN(5:15,:,:),1));
% data_TN = squeeze(x_TN(10,:,:));
% data_TE = squeeze(x_TE(mod(idx_k0,size(x_TE,1)),:,:));
te(1)=20*5e-6;
te(2)=idx_k0*5e-6;
% x should now be ADC x coils x timeframes
ref_idx = Ni*8+1;% Steady-state time frame to use as reference
ref_idx = min(size(data_TE,2), ref_idx); 
if Ni>1
    ref_idx = ref_idx-mod(ref_idx,Ni)+1;
    ref_idx = ref_idx:ref_idx+Ni-1; % Steady-state time frame to use as reference
end

data_TN = angle(data_TN);
data_TE = angle(data_TE);

for i = 1:Ni
    data_TN(:,i:Ni:end) = data_TN(:,i:Ni:end) - repmat(data_TN(:,ref_idx(i)),1,size(data_TN(:,i:Ni:end),2));
    data_TE(:,i:Ni:end) = data_TE(:,i:Ni:end) - repmat(data_TE(:,ref_idx(i)),1,size(data_TE(:,i:Ni:end),2));
end
data_TN = unwrap(data_TN,[],2);
data_TN = unwrap(data_TN,[],1);

data_TE = unwrap(data_TE,[],2);
data_TE = unwrap(data_TE,[],1);


dw = (data_TE' - data_TN')/(te(2) - te(1));
dw0=dw;
dphi = (te(2)*data_TN'-te(1)*data_TE')/(te(2) - te(1));
dphi0=dphi;
dw_part = data_TE'/te(2);
mean_dw = mean(dw,2);
mean_dphi = mean(dphi,2);
mean_dw_part = mean(dw_part,2);
% % Use linear extrapolation to fit first few points
if(length(mean_dw)>1000)
    linear_trend = [[1:1000]' ones(1000,1)];
    beta = linear_trend(ref_idx:1000,:) \ mean_dw(ref_idx:1000);
else
    linear_trend = [(1:length(mean_dw))' ones(length(mean_dw),1)];
    beta = linear_trend(ref_idx:length(mean_dw),:) \ mean_dw(ref_idx:length(mean_dw));
end
mean_dw0=mean_dw;
mean_dw(1:ref_idx-1) = linear_trend(1:ref_idx-1,:)*beta;
mean_dw0=mean_dw0-mean_dw(1);
mean_dw = mean_dw0;
% mean_dw = mean_dw - linear_trend(1,:)*beta;
dw = dw + mean_dw(ref_idx(1));

% Smooth time series with moving average
% DummyScans are excluded from averaging
%moving_average_length = 3;
%mean_dw = conv(mean_dw,ones(1,moving_average_length)/moving_average_length,'same');
%data_TE(meas_header.DummyScans+1:end,:) = conv2(ones(moving_average_length,1)/moving_average_length,1,padarray(data_TE(meas_header.DummyScans+1:end,:),(moving_average_length-1)/2,'replicate','both'),'valid');

end