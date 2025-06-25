function T=shells_hemisphere(R,fov,resolution,Npolar,Nradial,varargin)
% This function creates a single/multi shot concentric shells trajectory
% with variable density undersampling (see MRM...).

% INPUT:
%   R = [radial0 radialMax polar0 polarMax] : undersampling factors in
%       radial and polar direction at center of kspace (...0) and periphery (...Max)
%
%   fov : Field-Of-View in [m]
%
%   resolution : Resolution in pixel. Currently only isotropic resolution
%                is supported.
%
%   TE : Time of smallest shell element relative to start of trajectory.
%        Value must be in [s].
%
%   Npolar : Number of undersampling per shot in polar direction. This
%            means interleaved (multishot) acquisition so that after Npolar
%            shots the global polar sampling density (specified by R(3) and
%            R(4)) is achieved. For each shot and each shell element the
%            number of turns on the surface is therfore decreased by factor
%            Npolar. Default is single shot acquisition (Npolar =1).
%
%   Nradial : Number of undersampling per shot in radial direction... (to be continoued by jakob)
%
% VARARGIN:
% In varargin several additional parameter/value pairs can be specified.
% (function vararg_pair_bz is needed)
%
% parameter     : value
% 'GradSystem'  : 'slow'|'medium';|'fast'.
% 'SlewMode'    : 'const'|'optim'.
% 'alpha'       : value in radians.
%
%
% OUTPUT:
% Output is a trajectory structure containing the actual trajectory,
% gradient shape and various additional information. This gradient
% structure can be exported as a scanner readable *.grad-file.


if nargin < 5
    Nradial = 1;
end
if nargin < 4
    Npolar  = 1;
end
if nargin < 3
    resolution = 64;
end
if nargin < 2
    fov = 0.256;
end
if nargin <1
    R = [2 5 3 5];
end

%default arguments for optional varargins
args.GradSystem     = 'fast';
args.SlewMode   = 'optim';
args.alpha      = 0; %additional rotation bewtween shell elements
args.wave = 0;

%check if any default args are overwritten by user input
args = vararg_pair(args, varargin);

Rrad0       = R(1);
RradMax     = R(2);
Rpolar0     = R(3);
RpolarMax   = R(4);

% Calculate global maximum k-space from FOV and voxel size
KMAX = (1/fov)*(resolution/2.0); %should be in[1/m]
deltaK_full = KMAX/(resolution/2); %k-space increment for full sampling

%Create structure that specifies gradient system
% SYSTEM=GradSystemStructure(args.GradSystem);
SYSTEM=GradSystemStructure('custom',[],150);



%% Set up the parameters for the individual shells
[k_vd NofShells] = radial_sampling_density(Rrad0,RradMax,0,resolution/2);

shell_radii = KMAX*k_vd; %array of shell radii in [1/m], corresponds to kR from Eq.x
% NofShells = length(k_vd).

%Calculate parameter a for full sampling for each shell element
a_full = pi./asin((deltaK_full./(2*shell_radii))); %see Eq.x in paper

% slope of acceleration factor
Rp_ink=(RpolarMax- Rpolar0)/length(shell_radii(:));

if Rp_ink == 0
    Rp = Rpolar0*ones(1,length(a_full));
else
    Rp = Rpolar0 :Rp_ink:RpolarMax;
end
a_accelerated = a_full./Rp(1:length(a_full)); %Number of revolutions for each shell element (total undersample k-space)
a_polar= a_accelerated/Npolar; % interleaved acquisition in polar direction further decreases the number of revolutions per shot

%views with slightly shifted radii according to number Nradial
shell_radius_tmp = zeros(Nradial, size(shell_radii, 2));
for iradial = 1:Nradial
    shell_radius_tmp(iradial,:) = shell_radii - (iradial-1) * diff([0 shell_radii])/Nradial;
end
shell_radii = shell_radius_tmp;


gamma=4258*1e4;


%% Create all single elements ...

% ...with constant slew rate for all elements
if strcmp(args.SlewMode,'const')
    for iradial=1:Nradial
        sign=1;
        for elem=1:NofShells
            for ipolar=1:Npolar
                T(elem,ipolar,iradial)=single_element_shell(ceil(a_polar(elem)/2),shell_radii(iradial,elem),sign,SYSTEM,0);
                sign = -sign;
                T(elem,ipolar,iradial)=trajectStruct_rotate(T(elem,ipolar,iradial),(2*pi/Npolar)*(ipolar-1),[0 0 1]);
            end
        end
    end
end

% ... and with optimized (individual) slew rates for better PNS performance
if strcmp(args.SlewMode,'optim') % Jakobs method
    
    %store original gradient system structure
    SYSTEM_ORG = SYSTEM;
    
    for iradial=1:1
        sign=1;
        for elem=1:NofShells
            % SYSTEM.SLEW = slew(elem); %T/m/s
            SYSTEM.SLEW = 400 * exp(-shell_radii(iradial, elem)/52) + 125;
%             if elem == NofShells    % davon ausgehend, dass dieser an Anfang gestellt wird
%                 SYSTEM.SLEW = SYSTEM.SLEW + 50;
%             end
            SYSTEM.SLEW = min(SYSTEM.SLEW, SYSTEM_ORG.SLEW);
            display(['slewrate = ', num2str(SYSTEM.SLEW)]);
            
            SYSTEM.SLEW_per_GRT = SYSTEM.SLEW*SYSTEM.GRT/1000; %[mT/m]
            SYSTEM.SLEW_per_GRT_SI = SYSTEM.SLEW*SYSTEM.GRT_SI; %[T/m];
            T(elem,1,iradial)=single_element_shell(ceil(a_polar(elem)/2),shell_radii(iradial,elem),sign,SYSTEM,0);
            
                if args.wave
                    slew=diff(T(elem,1,iradial).G)*1e5;
                    slew(end+1,:)=0;
                    step=40;
                    for nn=1:step:(length(T(elem,1,iradial).K )-step)
                        k0temp=T(elem,1,iradial).K(nn:nn+step-1,:);
                        g0temp=T(elem,1,iradial).G(nn:nn+step-1,:);
                        s0temp=slew(nn:nn+step-1,:);
                        for ii=3
                            k1temp=[cos(2*pi/step*2*(1:step/2)')-1;-(cos(2*pi/step*2*(1:step/2)')-1)];
                            % if nn==1
                            %     g1temp(1,1)=0;
                            % else
                            %     g1temp(1,1)=(T(elem,1,iradial).K(nn,ii)-T(elem,1,iradial).K(nn-1,ii))/1e-5/gamma;
                            % end
                            % g1temp(2:step,1)=diff(k1temp)/1e-5/gamma;
                            % if nn==1
                            %     s1temp(1,1)=0;
                            % else
                            %     s1temp(1,1)=(T(elem,1,iradial).G(nn,ii)-T(elem,1,iradial).G(nn-1,ii))*1e5;
                            % end
                            % s1temp(2:step,1)=diff(g1temp)*1e5;
                            g1temp(:,1)=diff(k1temp)/1e-5/gamma;
                            s1temp(:,1)=diff(g1temp)*1e5;

                            % h(1)=min(abs((-SYSTEM.GMAX_SI-g0temp(1:step,ii))./g1temp));
                            % h(2)=min(abs((SYSTEM.GMAX_SI-g0temp(1:step,ii))./g1temp));
                            % h(3)=min(abs((-SYSTEM.SLEW-s0temp(1:step,ii))./s1temp));
                            % h(4)=min(abs((SYSTEM.SLEW-s0temp(1:step,ii))./s1temp));
                            temp=((-SYSTEM_ORG.GMAX_SI-g0temp(1:step-1,ii))./g1temp);
                            temp=min(temp(temp>0));
                            h(1)=temp(1);
                            temp=((SYSTEM_ORG.GMAX_SI-g0temp(1:step-1,ii))./g1temp);
                            temp=min(temp(temp>0));
                            h(2)=temp(1);
                            temp=((-SYSTEM_ORG.SLEW-s0temp(1:step-2,ii))./s1temp);
                            temp=min(temp(temp>0));
                            h(3)=temp(1);
                            temp=((SYSTEM_ORG.SLEW-s0temp(1:step-2,ii))./s1temp);
                            temp=min(temp(temp>0));
                            h(4)=temp(1);
                            % printf([num2str(nn) ' ' num2str(temp)])
                            h2=min(abs(h));
                            T(elem,1,iradial).K(nn:nn+step-1,ii)=h2*k1temp+k0temp(:,ii);
                            T(elem,1,iradial).G(nn:nn+step-1,ii)=gradient(T(elem,1,iradial).K(nn:nn+step-1,ii),1e-5)/gamma;
                        end
                    end
                    [tem T(elem,1,iradial).G] = gradient(T(elem,1,iradial).K,1e-5);
                    T(elem,1,iradial).G = T(elem,1,iradial).G/gamma;
                end

            % sign = -sign;
            for ipolar=2:Npolar
                T(elem,ipolar,iradial)=trajectStruct_rotate(T(elem,1,iradial),(2*pi/Npolar)*(ipolar-1),[0 0 1]);
            end
        end
    end
end
T=T(end:-1:1,:,:);
for pp=1:Npolar
    for rr=1:Nradial
        T5(:,pp,rr)=T(rr:Nradial:end,pp,1);
        for nn=1:length(T5(:,pp,rr))
            T5(nn,pp,rr).K=T5(nn,pp,rr).K*(-1)^nn;
            T5(nn,pp,rr).G=T5(nn,pp,rr).G*(-1)^nn;
        end
    end
end
T=T5;
    
for pp=1:Npolar
    for rr=1:Nradial
        T5=T(:,pp,rr);
        for nn=1:length(T5)
            T1=T5;
            T1(nn).K=T5(nn).K(1:floor(length(T5(nn).K)/2),:);
            T1(nn).G=T5(nn).G(1:floor(length(T5(nn).G)/2),:);
            T2=T5;
            T2(nn).K=T5(nn).K(1+floor(length(T5(nn).K)/2):end,:);
            T2(nn).G=T5(nn).G(1+floor(length(T5(nn).K)/2):end,:);
            if mod(nn,2)==1
                T3(nn)=T1(nn);
                T4(nn)=T2(nn);
            else
                T3(nn)=T2(nn);
                T4(nn)=T1(nn);
            end
        end
        T4=T4(end:-1:1);
        for nn=1:length(T5)
            T6(nn,pp,rr)=T3(nn);
            % T6(nn+length(T3),1,rr)=T4(nn);
        end
    end
end
T=T6;
for pp=1:Npolar
    for rr=1:Nradial
        for nn=2:size(T,1)
            last2 = T(nn-1,pp,rr).K(end-1:end,1)+ 1i * T(nn-1,pp,rr).K(end-1:end,2);
            first2 = T(nn,pp,rr).K(1:2,1)+ 1i* T(nn,pp,rr).K(1:2,2);
            alpha = angle(diff(last2,1,1)) - angle(diff(first2,1,1));
            T(nn, pp, rr) = trajectStruct_rotate(T(nn,pp,rr),alpha,[0 0 1]);
        end
    end
end
display('All single elements created...')


%% ramp up first element
for iradial=1:Nradial
    for ipolar=1:Npolar
        temp = T(1,ipolar,iradial);
        T(1,ipolar,iradial) = trajectStruct_rampUp(T(1,ipolar,iradial));
        T(1,ipolar,iradial).index(1) = length(T(1,ipolar,iradial).G) - length(temp.G);
    end
end


%% connect elements for continous acquisition of k-space
for iradial =1:Nradial
    for ipolar=1:Npolar
        Tc(ipolar,iradial) = T(1,ipolar,iradial);
        alpha= args.alpha; % additional rotation between shell elements during continous acquisition of k-space
        for l=2:length(T(:,ipolar,iradial))
            Tc(ipolar,iradial) = trajectStruct_connect(Tc(ipolar,iradial),trajectStruct_rotate(T(l,ipolar,iradial),alpha,[0 0 1]),'rip');
        end
    end
end
display('...elements connected')


%% ramp down all elements
%make one array of trajectory elements that are acquired sequentially
T =Tc(:); %T(n) is a part of k-space that is acquired in one shot

for k=1:length(T)
    T(k).index(2) = length(T(k).G);
    T(k)=trajectStruct_rampDown(T(k));

    T(k).K((end+1):(end+length(T(k).K)),:)=-T(k).K(end:-1:1,:);
    [tem T(k).G] = gradient(T(k).K,1e-5);
    T(k).G = T(k).G/gamma;
    T(k).index(2)=length(T(k).K)-T(k).index(1);
end

%%
% for k=1:length(T)
%     if args.wave
%         slew=diff(T(k).G)*1e5;
%         slew(end+1,:)=0;
%         step=40;
%         for nn=T(k).index(1):step:(length(T(k).K )-step)
%             k0temp=T(k).K(nn:nn+step-1,:);
%             g0temp=T(k).G(nn:nn+step-1,:);
%             s0temp=slew(nn:nn+step-1,:);
%             for ii=1:3
%                 k1temp=[cos(2*pi/step*2*(1:step/2)')-1;-(cos(2*pi/step*2*(1:step/2)')-1)];
%                 if nn==1
%                     g1temp(1,1)=0;
%                 else
%                     g1temp(1,1)=(T(k).K(nn,ii)-T(k).K(nn-1,ii))/1e-5/gamma;
%                 end
%                 g1temp(2:step,1)=diff(k1temp)/1e-5/gamma;
%                 if nn==1
%                     s1temp(1,1)=0;
%                 else
%                     s1temp(1,1)=(T(k).G(nn,ii)-T(k).G(nn-1,ii))*1e5;
%                 end
%                 s1temp(2:step,1)=diff(g1temp)*1e5;
%                 % g1temp(:,1)=diff(k1temp)/1e-5/gamma;
%                 % s1temp(:,1)=diff(g1temp)*1e5;
% 
%                 % h(1)=min(abs((-SYSTEM.GMAX_SI-g0temp(1:step,ii))./g1temp));
%                 % h(2)=min(abs((SYSTEM.GMAX_SI-g0temp(1:step,ii))./g1temp));
%                 % h(3)=min(abs((-SYSTEM.SLEW-s0temp(1:step,ii))./s1temp));
%                 % h(4)=min(abs((SYSTEM.SLEW-s0temp(1:step,ii))./s1temp));
%                 temp=((-SYSTEM_ORG.GMAX_SI-g0temp(1:step,ii))./g1temp);
%                 temp=min(temp(temp>0));
%                 h(1)=temp(1);
%                 temp=((SYSTEM_ORG.GMAX_SI-g0temp(1:step,ii))./g1temp);
%                 temp=min(temp(temp>0));
%                 h(2)=temp(1);
%                 temp=((-SYSTEM_ORG.SLEW-s0temp(1:step,ii))./s1temp);
%                 temp=min(temp(temp>0));
%                 h(3)=temp(1);
%                 temp=((SYSTEM_ORG.SLEW-s0temp(1:step,ii))./s1temp);
%                 temp=min(temp(temp>0));
%                 h(4)=temp(1);
%                 % printf([num2str(nn) ' ' num2str(temp)])
%                 h2=min(abs(h));
%                 T(k).K(nn:nn+step-1,ii)=h2*k1temp+k0temp(:,ii);
%                 T(k).G(nn:nn+step-1,ii)=gradient(T(k).K(nn:nn+step-1,ii),1e-5)/gamma;
%             end
%         end
%     end
% end
for k=1:length(T)
    [tem T(k).G] = gradient(T(k).K,1e-5);
    T(k).G = T(k).G/gamma;
end
%%
nn=length(T);
for k=1:nn
    T(nn+k)=T(k);
    T(nn+k).K=-T(k).K;
    T(nn+k).G=-T(k).G;
end

%%
%determine longest segments and zero fill all other elements for equal
% number of ADCs in sequence
points_max=0;
for k=1:length(T)
    points_max=max(points_max,size(T(k).K,1));
end

for k=1:length(T)
    if size(T(k).K,1) < points_max
        T(k) = trajectStruct_zeroFill(T(k),points_max - size(T(k).K,1));
    end
end


%% reorder gradient axis. slow gradients are on physical y-axis
% if strcmp(args.SlewMode,'optim')
%     for i = 1 : Nradial * Npolar
%         T(i).SYS = SYSTEM_ORG;
%         T(i).K = circshift(T(i).K, [0 2]);  % slow gradient should be y due to stimulation
%         T(i).G = circshift(T(i).G, [0 2]);
%     end
% end


%% return information for trajectStruct_export
T(1).fov    = fov;
T(1).N      = resolution;
% [~, te]    = min(makesos(T.K(200:end-200,:), 2));
T(1).TE    = 0;%(te + 200) * 10; % [us]

display('finished')