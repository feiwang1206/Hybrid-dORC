function  A = orc_segm_nuFTOperator_multi_savetime_smaps_wi(s, sensmaps,wi, wmap)

if isempty(sensmaps)
    s.numCoils = 1;
else
    if size(s.trajectory{1},2) == 3 && length(size(sensmaps))== 4
        s.numCoils = size(sensmaps, length(size(sensmaps)));
    end
    if size(s.trajectory{1},2) == 3 && length(size(sensmaps))== 3
        s.numCoils = 1;
    end
    if size(s.trajectory{1},2) == 2 && length(size(sensmaps))== 3
        s.numCoils = size(sensmaps, length(size(sensmaps)));
    end
    if size(s.trajectory{1},2) == 2 && length(size(sensmaps))== 2
        s.numCoils = 1;
    end
end
if isempty(sensmaps)
    s.sensmaps{1} = 1;
else
    for k=1:s.numCoils
        if size(s.trajectory{1},2) == 3
            if s.numCoils > 1
                s.sensmaps{k} = sensmaps(:,:,:,k);
            else
                s.sensmaps{1}=sensmaps;
            end
        else
             s.sensmaps{k} = sensmaps(:,:,k);
        end
    end
end
s.sensmaps_scale = bsxfun(@times,s.scaling_factor,sensmaps);

sta=0;
for n=1:length(s.tADC)
    for k = 1:s.nSegments+1
        if length(size(wmap)) == 3
            s.wmap(:,:,:,k+sta) = exp(1i*wmap*s.t(k));
        else
            s.wmap(:,:,k+sta) = exp(1i*wmap*s.t(k));                
        end
    end
    
    sta=sta+s.nSegments+1;
end

% s1=s;
% trajectory=[];
% for n=1:length(s.trajectory)
%     trajectory=[trajectory;s.trajectory{n};];
% end
% s1.nufftStruct= nufft_init(trajectory, s.imageDim*2, s.nufftNeighbors, round(4*s.imageDim), ceil(s.imageDim), 'minmax:kb');
s.wi=wi;

A = class(s,'orc_segm_nuFTOperator_multi_savetime_smaps_wi');


    

