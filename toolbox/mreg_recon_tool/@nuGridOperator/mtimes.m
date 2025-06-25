function Q = mtimes(A,B)

% nufft_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling)
% nufft_adj_simple(B .* A.sensmaps{k}, A.scaling_factor, A.interpolation_matrix{m}, A.oversampling, A.imageDim)


if strcmp(class(A),'nuGridOperator')
    
    if A.adjoint
        
%         Q = reshape(A.interp_filter'*B,[A.oversampling size(B,2)]);
        Q = zeros([A.imageDim size(B,2)]);
%         B = reshape(B,[],A.numCoils);
%         x = reshape(A.interp_filter'*B,[A.oversampling A.numCoils]);

                            
        if length(A.imageDim)==3
            nx=A.imageDim(1);
            ny=A.imageDim(2);
            nz=A.imageDim(3);
            input=B;
            output = zeros([A.imageDim,size(B,2)]);
            traj=A.trajectory/pi*32+32;
            for i =1:length(traj)
                kx = traj(i,3);
                ky = traj(i,2);
                kz = traj(i,1);

                x0 = ceil(kx - A.nufftNeighbors(1) / 2);
                y0 = ceil(ky - A.nufftNeighbors(2) / 2);
                z0 = ceil(kz - A.nufftNeighbors(3) / 2);

                x1 = floor(kx + A.nufftNeighbors(1) / 2);
                y1 = floor(ky + A.nufftNeighbors(2) / 2);
                z1 = floor(kz + A.nufftNeighbors(3) / 2);

                for z =z0:(z1)
                    wz = kaiser_bessel_kernel((z - kz) / (A.nufftNeighbors(3) / 2), A.beta);

                    for y =y0:(y1)
                        wy = wz * kaiser_bessel_kernel((y - ky) / (A.nufftNeighbors(2) / 2), A.beta);

                        for x =x0:(x1)
                            w = wy * kaiser_bessel_kernel((x - kx) / (A.nufftNeighbors(1) / 2), A.beta);

                            for b =1:size(B,2)
                                output(mod(z,nz)+1,mod(y,ny)+1,mod(x,nx)+1,b)=output(mod(z,nz)+1,mod(y,ny)+1,mod(x,nx)+1,b)+w*input(i,b);
                            end
                        end
                    end
                end
            end
            output=output/prod(A.nufftNeighbors)*prod(A.imageDim);
            x=output;
        else
            nx=A.imageDim(1);
            ny=A.imageDim(2);
            input=B;
            output = zeros([A.imageDim,size(B,2)]);
            traj=A.trajectory/pi*32+32;
            for i =1:length(traj)
                kx = traj(i,2);
                ky = traj(i,1);

                x0 = ceil(kx - A.nufftNeighbors(1) / 2);
                y0 = ceil(ky - A.nufftNeighbors(2) / 2);

                x1 = floor(kx + A.nufftNeighbors(1) / 2);
                y1 = floor(ky + A.nufftNeighbors(2) / 2);

                    for y =y0:(y1)
                        wy = kaiser_bessel_kernel((y - ky) / (A.nufftNeighbors(2) / 2), A.beta);

                        for x =x0:(x1)
                            w = wy * kaiser_bessel_kernel((x - kx) / (A.nufftNeighbors(1) / 2), A.beta);

                            for b =1:size(B,2)
                                output(mod(y,ny)+1,mod(x,nx)+1,b)=output(mod(y,ny)+1,mod(x,nx)+1,b)+w*input(i,b);
                            end
                        end
                    end
            end
            output=output/prod(A.nufftNeighbors)*prod(A.imageDim);
        end
        Q = output;

    else
%         x = reshape(B,[],numel(B)/size(A.interp_filter,2));
%         Q = A.interp_filter*double(x);
        if length(A.imageDim)==3
            dim=size(B);
            dim(end+1)=1;
            input=B;
            output = zeros(length(A.trajectory),dim(4));
            nx=A.imageDim(1);
            ny=A.imageDim(2);
            nz=A.imageDim(3);

            traj=A.trajectory/pi*32+32;
            for i =1:length(traj)
                    kx = traj(i,3);
                    ky = traj(i,2);
                    kz = traj(i,1);

                    x0 = ceil(kx - A.nufftNeighbors(1) / 2);
                    y0 = ceil(ky - A.nufftNeighbors(2) / 2);
                    z0 = ceil(kz - A.nufftNeighbors(3) / 2);

                    x1 = floor(kx + A.nufftNeighbors(1) / 2);
                    y1 = floor(ky + A.nufftNeighbors(2) / 2);
                    z1 = floor(kz + A.nufftNeighbors(3) / 2);

                    for z =z0:(z1)
                        wz = kaiser_bessel_kernel((z - kz) / (A.nufftNeighbors(3) / 2), A.beta);

                        for y =y0:(y1)
                            wy = wz * kaiser_bessel_kernel((y - ky) / (A.nufftNeighbors(2) / 2), A.beta);

                            for x =x0:(x1)
                                w = wy * kaiser_bessel_kernel((x - kx) / (A.nufftNeighbors(1) / 2), A.beta);

                                for b =1:dim(4)
                                    output(i,b) = output(i,b)+w * input(mod(z,nz)+1,mod(y,ny)+1,mod(x,nx)+1,b);
                                end
                            end
                        end
                    end
            end
        else
            input=B;
            dim=size(B);
            dim(end+1)=1;
            output = zeros(length(A.trajectory),dim(3));
            nx=A.imageDim(1);
            ny=A.imageDim(2);

            traj=A.trajectory/pi*32+32;
            for i =1:length(traj)
                    kx = traj(i,2);
                    ky = traj(i,1);

                    x0 = ceil(kx - A.nufftNeighbors(1) / 2);
                    y0 = ceil(ky - A.nufftNeighbors(2) / 2);

                    x1 = floor(kx + A.nufftNeighbors(1) / 2);
                    y1 = floor(ky + A.nufftNeighbors(2) / 2);

                        for y =y0:(y1)
                            wy = kaiser_bessel_kernel((y - ky) / (A.nufftNeighbors(2) / 2), A.beta);

                            for x =x0:(x1)
                                w = wy * kaiser_bessel_kernel((x - kx) / (A.nufftNeighbors(1) / 2), A.beta);

                                for b =1:dim(3)
                                    output(i,b) = output(i,b)+w * input(mod(y,ny)+1,mod(x,nx)+1,b);
                                end
                            end
                        end
            end
        end
        Q=output/prod(A.nufftNeighbors);
%         Q=col(Q);
    end
    
    
% now B is the operator and A is the vector
elseif strcmp(class(B),'nuGridOperator')
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
