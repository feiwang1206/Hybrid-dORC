function Q = mtimes(A,B)

% nufft_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling)
% nufft_adj_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling, A.imageDim)


if strcmp(class(A),'nuFTOperator')
    
    if A.adjoint

        Q = zeros([size(A) A.numCoils]);
        B = reshape(B,[],A.numCoils);
        x = reshape(A.interp_filter'*B,[A.oversampling A.numCoils]);

                            
        if length(A.imageDim)==3
            x = prod(A.imageDim)*ifft(ifft(ifft(x,[],3),[],2),[],1);
%             x = ifft(ifft(ifft(x,[],3),[],2),[],1);
            if any(A.imageDim < A.oversampling)
                x = x(1:A.imageDim(1),1:A.imageDim(2),1:A.imageDim(3),:);
            end
        else
            x = prod(A.imageDim)*ifft(ifft(x,[],2),[],1);
%             x = ifft(ifft(x,[],2),[],1);
            if any(A.imageDim < A.oversampling)
                x = x(1:A.imageDim(1),1:A.imageDim(2),:);
            end
        end
        Q = x.*conj(A.sensmaps_scale);
        if length(A.imageDim)==3
            Q = sum(Q,4);
        else
            Q = sum(Q,3);
        end
        Q=Q/sqrt(prod(A.imageDim));

    else
        
        B=reshape(B,A.imageDim);
        if length(A.imageDim)==3
            B = B(:,:,:,ones(A.numCoils,1)).*A.sensmaps_scale;
            Q = zeros(A.trajectory_length,A.numCoils);
            x = fft(fft(fft(B,A.oversampling(3),3),A.oversampling(2),2),A.oversampling(1),1);
            x = reshape(x,[],A.numCoils);
            Q = Q + A.interp_filter*double(x);
        else
            B = B(:,:,ones(A.numCoils,1)).*A.sensmaps_scale;
            Q = zeros(A.trajectory_length,A.numCoils);
            x = reshape(fft(fft(B,A.oversampling(2),2),A.oversampling(1),1),[],A.numCoils);
            Q = Q + A.interp_filter*double(x);
        end
    
      	Q = (Q) / sqrt(prod(A.imageDim));
%       	Q = col(Q) / sqrt(prod(A.imageDim));
    end
    
    
% now B is the operator and A is the vector
elseif strcmp(class(B),'nuFTOperator')
    Q = mtimes(B',A')';

else
   error('nuFTOperator:mtimes', 'Neither A nor B is of class nuFTOperator');
end
    
end

function kb=kaiser_bessel_kernel(x,beta)
    x = beta * (1 - x^2)^0.5;
    t = x / 3.75;
    if x < 3.75
             kb=1 + 3.5156229 * t^2 + 3.0899424 * t^4 +...
                1.2067492 * t^6 + 0.2659732 * t^8 +...
                0.0360768 * t^10 + 0.0045813 * t^12;
    else
             kb=x^-0.5 * exp(x) * (...
                0.39894228 + 0.01328592 * t^-1 +...
                0.00225319 * t^-2 - 0.00157565 * t^-3 +...
                0.00916281 * t^-4 - 0.02057706 * t^-5 +...
                0.02635537 * t^-6 - 0.01647633 * t^-7 +...
                0.00392377 * t^-8);
    end
end