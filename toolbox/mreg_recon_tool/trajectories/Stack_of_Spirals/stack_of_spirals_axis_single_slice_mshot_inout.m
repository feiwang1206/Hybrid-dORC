function T = stack_of_spirals_axis_single_slice_mshot_inout(R,fov,resolution)

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
Rz_min = R(3);
Rz_max = R(4);

SYSTEM=GradSystemStructure('custom', [], 150);
% SYSTEM=GradSystemStructure('slow');


%% Definition of the size of k-space
k_max = 1./(2*resolution);     % 1D edge of k-space in 1/m
% k_min = 1/(fov(3));
% kz=0:k_min*gap:k_max;
kz(1) = 0;
kr_max(1) = k_max(1);
i = 1;
while kz(i) < k_max(3)
    kz(i+1) = min(kz(i) + (Rz_min + kz(i)/k_max(3) * (Rz_max - Rz_min))/fov(3), k_max(3));
    kr_max(i+1) = max(k_max(1)*sqrt(1 - (kz(i+1)/k_max(3))^2), k_max(1)/10);
    i = i + 1;
end
kz = [kz((end-1):-1:2), -kz];
kr_max=repmat(k_max(1),[1 length(kz)]);

%% Inversion to demonstrate different offresonance behavior
% kz = kz(end:-1:1);

%% Create all single spirals
Rmax = Rrad_max;
Rmin = Rrad_min;

T(1) = single_element_spiral(kz(1), kr_max(1), Rmin, Rmax, fov(1), -1, SYSTEM);
T(2) = single_element_spiral(kz(1), kr_max(1), Rmin, Rmax, fov(1), 1, SYSTEM);
% gap
% T(1).K((end+1):(end+length(T(1).K)),:)=-T(1).K(end:-1:1,:);
% T(1).G((end+1):(end+length(T(1).G)),:)=-T(1).G(end:-1:1,:);
for i=1:2
    %% ramp up first element
    temp = T(i);
    T(i) = trajectStruct_rampUp(T(i));
    T(i).index(1) = length(T(i).G) - length(temp.G);
    %% raup down last element
    T(i).index(2) = length(T(i).G);
    T(i)=trajectStruct_rampDown(T(i));
end
%% connect
Tc = T(1);
Tc = trajectStruct_connect(Tc,T(2));
T=Tc;
%%
for ie=2:length(kz)
    T(ie)=T(1);
    T(ie).K(:,3) = T(1).K(:,3)*(kz(ie)/kz(1));    
    T(ie).G(:,3) = T(1).G(:,3)*(kz(ie)/kz(1));
end

% T(2)=T(1);
% T(2).K=-T(1).K;T(2).G=-T(1).G;

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