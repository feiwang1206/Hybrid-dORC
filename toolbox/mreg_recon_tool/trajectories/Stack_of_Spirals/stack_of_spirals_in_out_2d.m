function T = stack_of_spirals_in_out_2d(R,Nradial,Nz,fov,resolution)

%% usage: T = stack_of_spirals(R,Nradial,Nz,fov,resolution, pf)
% Needs the toolbox of Benni Zahneisen

%% INPUT:
% R = reduction factor: [Rrad_min Rrad_max Rz_min Rz_max]
%   to set FOV_z smaller simply increas Rz
% Nradial: interleaves in radial direction
% Nz:      interleavs in z - direction
% fov in m: It is set isotropic but can be made unisotropic by changing R
% resolution in m: (!!! This definition is different to Benni's shells !!!)
%           This is only implemented for isotropic resolution but could 
%           easily be made unisotropic by changing kz_max
% pf:       Partial Fourier factor. If you don't want to use partial
%           fourier, use one or leave empty. 0.5 is half Fourier, 1 is full
%           sampling etc. K-space is cut of at the beginning.
% alt_order: If 1, the aquisition direction in z is alternated every other
%            step. Otherwise choose 0 or leave empty.

%% OUTPUT:
% T: Trajectory struct (defined in Benni's toolbox)

% Jakob Asslaender August 2011


Rrad_min = R(1);
Rrad_max = R(2);
Rz_min   = R(3);
Rz_max   = R(4);

SYSTEM=GradSystemStructure('custom', [], 150);


%% Definition of the size of k-space
k_max = 1/(2*resolution(1));     % 1D edge of k-space in 1/m
kz(1) = 0;
kr_max(1) = k_max;
i = 1;
while kz(i) < k_max
    kz(i+1) = min(kz(i) + (Rz_min + kz(i)/k_max * (Rz_max - Rz_min))/fov(3), k_max);
    kr_max(i+1) = max(sqrt(kr_max(1)^2 - kz(i+1)^2), k_max/10);
    i = i + 1;
end
% kz = [kz(end:-1:2), -kz];
% kr_max = [kr_max(end:-1:2), kr_max];

%% Inversion to demonstrate different offresonance behavior
% kz = kz(end:-1:1);

%% Create all single spirals
for iz=1:Nz
    for element = 1:2
        if element==1
            in_out = -1;
        else
            in_out=1;
        end
        Rmin = Rrad_min;
        Rmax = Rrad_max;
        T(element, iz, 1) = single_element_spiral(kz(1), kr_max(1), Rmin, Rmax, fov(1), in_out, SYSTEM);
        
    end
end

display('All single elements created...')

%% ramp up first element
for iradial=1:Nradial
    for iz=1:size(T, 2)
        temp = T(1,iz,iradial);
        T(1,iz,iradial) = trajectStruct_rampUp(T(1,iz,iradial));
        T(1,iz,iradial).index(1) = length(T(1,iz,iradial).G) - length(temp.G);
    end
end

% connect elements by bending the endings ('rip' mode).
for iradial =1:Nradial
    for iz=1:size(T, 2)
        Tc(iz,iradial) = T(1,iz,iradial);
        for element=2:size(T, 1)
            if ~isempty(T(element,iz,iradial).K)
                Tc(iz,iradial) = trajectStruct_connect(Tc(iz,iradial),T(element,iz,iradial), 'rip');
            end
        end
    end
end
display('...elements connected')
T =Tc(:);

%% raup down last element
for k=1:length(T)
    T(k).index(2) = length(T(k).G);
    T(k)=trajectStruct_rampDown(T(k));
end

%%


% determine longest segments
points_max=0;
for k=1:length(T)
    points_max=max(points_max,size(T(k).K,1));
end

for k=1:length(T)
    if size(T(k).K,1) < points_max
        T(k) = trajectStruct_zeroFill(T(k),points_max - size(T(k).K,1));
    end
end

% return information for trajectStruct_export
for i=1:length(T)
    T(i).fov    = fov;
    T(i).N      = fov./resolution;
    % The 200 makes sure that not beginning of the trajectory (which usually is in the k-space center) does not count as TE
    [~, te]    = min(makesos(T(i).K(200:end-200,:), 2));  % The Echotime of the first trajectory is taken. Hopefully they are all similar...
    T(i).TE    = (te + 200) * 10; % [us]
end

% T.K = [T.K(:,2), T.K(:,3), T.K(:,1)];
% T.G = [T.G(:,2), T.G(:,3), T.G(:,1)];


display('finished')