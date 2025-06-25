function Q = mtimes(A,B)

% nufft_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling)
% nufft_adj_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling, A.imageDim)


if strcmp(class(A),'orc_segm_nuFTOperator_multi_sub')
    
%     if A.adjoint
%         Q=reshape(double(A.Gb'*B(:)),A.imageDim);
%     else
%         Q=double(A.Gb*B(:));
%     end
    
    if A.adjoint
        Q = zeros([size(A) A.numCoils]);
        B = reshape(B,[],A.numCoils);
        for m=1:length(A.interp_filter)
            x = reshape(A.interp_filter{m}'*double(B),[A.oversampling A.numCoils]);
            if length(A.imageDim)==3
                x = prod(A.imageDim)*ifft(ifft(ifft(x,[],3),[],2),[],1);
%                 x = ifft(ifft(ifft(x,[],3),[],2),[],1);
                if any(A.imageDim < A.oversampling)
                    x = x(1:A.imageDim(1),1:A.imageDim(2),1:A.imageDim(3),:);
                end
            else
                x = prod(A.imageDim)*ifft(ifft(x,[],2),[],1);
%                 x = ifft(ifft(x,[],2),[],1);
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
        if length(A.imageDim)==3
            Q = sum(Q,4);%;
        else
            Q = sum(Q,3);
        end
        Q=(Q)/sqrt(prod(A.imageDim));
        

    else
        if length(A.imageDim)==3
            B = repmat(reshape(B,A.imageDim),[1,1,1,A.numCoils]).*A.sensmaps_scale;
            Q = zeros(A.trajectory_length,A.numCoils);
            for m=1:length(A.interp_filter)
                x = B.*repmat(A.wmap(:,:,:,m),[1 1 1 A.numCoils]);
                x = fft(fft(fft(x,A.oversampling(3),3),A.oversampling(2),2),A.oversampling(1),1);
                x = reshape(x,[],A.numCoils);
                x = A.interp_filter{m}*double(x);
                Q = Q + x;
            end
        else
            B = repmat(reshape(B,A.imageDim),[1,1,A.numCoils]).*A.sensmaps_scale;
            Q = zeros(A.trajectory_length,A.numCoils);
            for m=1:length(A.interp_filter)
                x = B.*repmat(A.wmap(:,:,m),[1 1 A.numCoils]);
                x = reshape(fft(fft(x,A.oversampling(2),2),A.oversampling(1),1),[],A.numCoils);
                Q = Q + A.interp_filter{m}*double(x);
            end
        end
      	Q = (Q)/sqrt(prod(A.imageDim));
        
    end
    
    
% now B is the operator and A is the vector
elseif strcmp(class(B),'orc_segm_nuFTOperator_multi_sub')
    Q = mtimes(B',A')';

else
   error('orc_segm_nuFTOperator_multi:mtimes', 'Neither A nor B is of class orc_segm_nuFTOperator_multi');
end
    
end