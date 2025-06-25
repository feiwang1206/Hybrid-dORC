function  A = orc_segm_nuFTOperator_multi_sub_savetime(s, wmap)


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
s.imageDim2=size(wmap);

A = class(s,'orc_segm_nuFTOperator_multi_sub_savetime');

    

