function  A = nuGridOperator(trajectory, imageDim, dim,os, neighbors, kernel)

%% Usage:
%    A = nuFTOperator(trajectory, imageDim, sensmaps, os, neighbors, kernel)
%
% trajectory =  [N_timepoints x N_dimensions]:
%               Arbitrary trajectory in 2D or 3D k-space (obligatory)
%
% imageDim =    [Nx Ny Nz]:
%               Dimensions of the image (Only 2 entries for 2D; obligatory)
%
% sensmaps =    [Nx Ny Nz Ncoils]:
%               Coil sensitivity maps ([Nx Ny Ncoils] for 2D; optional)
%
% os =          Scalar: Oversampling (optional; default = 1)
%
% neighbors =   [x_neighbors y_neighbors] for 2D, 
%               [x_neighbors y_neighbors z_neighbors] for 3D
%               Number of neighbors to include into interpolation;
%               (optional; default 5 in each dimension);
%
% kernel =      'kaiser' for Kaiser-Bessel interpolation or 
%               'minmax:kb' for Fessler's Min-Max kernel with Kaiser-Bessel
%               based scaling; See nufft_init for more options (optional;
%               default = 'kaiser')
%
% 30.09.2011  Thimo Hugger
% 2010 - 2013 Jakob Asslaender: Minor changes + major documentation ;)


if nargin==0 % default constructor
    s.imageDim = [];
    s.adjoint = 0;
    s.trajectory_length = [];
    s.nufftNeighbors = [];
    s.nufftStruct = [];
else
    s.imageDim = imageDim;
    % By default the operator is not adjoint, to get the adjoint, just call
    % A'
    s.adjoint = 0;
    s.trajectory_length = size(trajectory,1);
    
    % Size of neighborhood for gridding:
    if nargin < 5 || isempty(neighbors)
        if size(trajectory,2) == 3      % 3D
            s.nufftNeighbors = [5 5 5]/1;
        else                            % 2D
            s.nufftNeighbors = [5 5];
        end
    else
        s.nufftNeighbors = neighbors;
    end
    
    if nargin<=3 || isempty(os)
        os = 1;
    end
    if nargin < 6 || isempty(kernel)
        kernel = 'kaiser';
    end
    
    
    % Siemens dimensions 2 Fessler dimensions (always fun to shuffle)
%     if size(trajectory,2) == 3
%         trajectory = [trajectory(:,2), -trajectory(:,1) , trajectory(:,3)];
%     else
%         trajectory = [trajectory(:,2), -trajectory(:,1)];
%     end
    
    % Now everything is in place and we can initialize the nuFFT. The
    % gridding kernel can be e.g. 'kaiser' or 'minmax:kb'
    s.nufftStruct = nufft_init(trajectory, imageDim, s.nufftNeighbors, round(os*imageDim), ceil(imageDim/2), kernel);

    s.interp_filter = spdiags(ones(s.trajectory_length,1),0,s.trajectory_length,s.trajectory_length)*s.nufftStruct.p.arg.G;
    s.oversampling = s.imageDim;
    s.trajectory=trajectory;
    s.beta = pi * (((s.nufftNeighbors(1) / os) * (os-0.5))^2-0.8)^0.5;
%     s.numCoils=dim;
end

A = class(s,'nuGridOperator');
