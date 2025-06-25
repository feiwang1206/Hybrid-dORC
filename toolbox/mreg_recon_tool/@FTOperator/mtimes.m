function Q = mtimes(A,B)

% nufft_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling)
% nufft_adj_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling, A.imageDim)


if strcmp(class(A),'FTOperator')
    
    if A.adjoint

        Q = zeros([size(A) A.numCoils]);
        if length(A.imageDim)==3
%             x = prod(A.imageDim)*ifft(ifft(ifft(B,[],3),[],2),[],1);
            x = ifff(ifff(ifftf(B,1),2),3);
            if any(A.imageDim < A.oversampling)
                x = x(1:A.imageDim(1),1:A.imageDim(2),1:A.imageDim(3),:);
            end
        else
%             x = prod(A.imageDim)*ifft(ifft(B,[],2),[],1);
            x = ifff(ifff(B,1),2);
            if any(A.imageDim < A.oversampling)
                x = x(1:A.imageDim(1),1:A.imageDim(2),:);
            end
        end
        Q = (x.*conj(A.sensmaps_scale));
        if length(A.imageDim)==3
            Q = sum(Q,4);
        else
            Q = sum(Q,3);
        end
%         Q=col(Q);

    else
        
        if length(A.imageDim)==3
            B = B(:,:,:,ones(A.numCoils,1)).*A.sensmaps_scale;
%             Q = fft(fft(fft(B,A.oversampling(3),3),A.oversampling(2),2),A.oversampling(1),1);
            Q=fff(fff(fff(B,1),2),3);
        else
            B = B(:,:,ones(A.numCoils,1)).*A.sensmaps_scale;
%             Q = fft(fft(B,A.oversampling(2),2),A.oversampling(1),1);
            Q=fff(fff(B,1),2);
        end
%       	Q = col(Q) / sqrt(prod(A.imageDim));
%       	Q = (Q) / sqrt(prod(A.imageDim));
    end
    
    
% now B is the operator and A is the vector
elseif strcmp(class(B),'FTOperator')
    Q = mtimes(B',A')';

else
   error('nuFTOperator:mtimes', 'Neither A nor B is of class nuFTOperator');
end
    
end