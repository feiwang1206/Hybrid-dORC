function Q = mtimes(A,B)

% nufft_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling)
% nufft_adj_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling, A.imageDim)


if strcmp(class(A),'orc_segm_nuFTOperator_multi_savetime_smaps')
    
    if A.adjoint

        Q = zeros([A.imageDim A.numCoils]);
        B = reshape(B,[],A.numCoils);
        B = B.*A.wi;
        for m=1:length(A.interp_filter)
            x = reshape(A.interp_filter{m}'*B,[A.oversampling A.numCoils]);
            if length(A.imageDim)==3
                x = prod(A.imageDim)*ifft(ifft(ifft(x,[],3),[],2),[],1);
                if any(A.imageDim < A.oversampling)
                    x = x(1:A.imageDim(1),1:A.imageDim(2),1:A.imageDim(3),:);
                end
            else
                x = prod(A.imageDim)*ifft(ifft(x,[],2),[],1);
                if any(A.imageDim < A.oversampling)
                    x = x(1:A.imageDim(1),1:A.imageDim(2),:);
                end
            end
            if length(A.imageDim)==3
                Q = Q + x.*conj(repmat(A.wmap(:,:,:,m),[1 1 1 A.numCoils]));
            else
                Q = Q + x.*conj(repmat(A.wmap(:,:,m),[1 1 A.numCoils]));
            end
        end
        Q = Q.*conj(A.sensmaps_scale);
%         Q = sum(Q,4)/sqrt(prod(A.imageDim))/length(A.tADC);
        if length(A.imageDim)==3
            Q = sum(Q,4)/sqrt(prod(A.imageDim));
        else
            Q = sum(Q,3)/sqrt(prod(A.imageDim));
        end
        Q = Q./conj(A.scaling_factor)./A.scaling_factor;

    else
        
        if length(A.imageDim)==3
            B = B(:,:,:,ones(A.numCoils,1)).*A.sensmaps_scale;
            Q = zeros(A.trajectory_length,A.numCoils);
            for m=1:length(A.interp_filter)
                x = B.*repmat(A.wmap(:,:,:,m),[1 1 1 A.numCoils]);
                x = reshape(fft(fft(fft(x,A.oversampling(3),3),A.oversampling(2),2),A.oversampling(1),1),[],A.numCoils);
                Q = Q + A.interp_filter{m}*double(x);
            end
        else
            B = B(:,:,ones(A.numCoils,1)).*A.sensmaps_scale;
            Q = zeros(A.trajectory_length,A.numCoils);
            for m=1:length(A.interp_filter)
                x = B.*repmat(A.wmap(:,:,m),[1 1 A.numCoils]);
                x = reshape(fft(fft(x,A.oversampling(2),2),A.oversampling(1),1),[],A.numCoils);
                Q = Q + A.interp_filter{m}*x;
            end
        end
      	Q = (Q) / sqrt(prod(A.imageDim));
    end
    
    
% now B is the operator and A is the vector
elseif strcmp(class(B),'orc_segm_nuFTOperator_multi_savetime')
    Q = mtimes(B',A')';

else
   error('orc_segm_nuFTOperator_multi:mtimes', 'Neither A nor B is of class orc_segm_nuFTOperator_multi');
end
    
end