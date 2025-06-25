function T = stack_of_spirals_pulseq2d_cartesian(fov,N)
SYSTEM=GradSystemStructure('custom', [], 150);

%% Definition of the size of k-space
dk=1/fov;
gamma=4258*1e4;
for i=1:N
    T(i).K(:,1)=dk*(-N/2:(N/2-1))*(-1)^i;
    T(i).K(:,2)=dk*(i-N/2-1);
    T(i).K(:,3)=0;
%     if i==1
%         clear temp
%         temp(:,1)=(T(i).K(1,1)/100:T(i).K(1,1)/100:T(i).K(1,1));
%         temp(:,2)=(T(i).K(1,2)/100:T(i).K(1,2)/100:T(i).K(1,2));
%         temp(:,3)=(T(i).K(1,3)/100:T(i).K(1,3)/100:T(i).K(1,3));
%         T(i).K=[temp;T(i).K];
%         clear temp
%         temp(:,1)=(T(i).K(1,1)/100:T(i).K(1,1)/100:T(i).K(1,1));
%         temp(:,2)=(T(i).K(1,2)/100:T(i).K(1,2)/100:T(i).K(1,2));
%         temp(:,3)=(T(i).K(1,3)/100:T(i).K(1,3)/100:T(i).K(1,3));
%         
%     end
    [~, T(i).G]=gradient(T(i).K,1e-5);
    T(i).G=T(i).G/gamma;
    T(i).SYS=SYSTEM;
    T(i).duration=N*1e-5;
    T(i).rampUp='false';
    T(i).rampDown='false';
end

display('All single elements created...')

%% ramp up first element
temp = T(1);
T(1) = trajectStruct_rampUp(T(1));
T(1).index(1) = length(T(1).G) - length(temp.G);

% connect elements by bending the endings ('rip' mode).
Tc = T(1);
for element=2:length(T)
    Tc = trajectStruct_connect(Tc,T(element));
%     Tc.K=[Tc.K;T(element).K];
%     Tc.G=[Tc.G;T(element).G];
end
display('...elements connected')
T =Tc(:);

%% raup down last element
for k=1:length(T)
    T(k).index(2) = length(T(k).G);
    T(k)=trajectStruct_rampDown(T(k));
    T(k).K(:,3)=0;
    T(k).G(:,3)=0;
end

for i=1:length(T)
    T(i).fov    = [fov fov fov/N];
    T(i).N      = N;
    % The 200 makes sure that not beginning of the trajectory (which usually is in the k-space center) does not count as TE
    [~, te]    = min(makesos(T(i).K(200:end-200,:), 2));  % The Echotime of the first trajectory is taken. Hopefully they are all similar...
    T(i).TE    = (te + 200) * 10; % [us]
end


display('finished')