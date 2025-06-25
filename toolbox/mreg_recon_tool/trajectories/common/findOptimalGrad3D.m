%% Function that use linear programing framework
function [G exitflag]=findOptimalGrad3D(SYS,grad,Kinc)
%SYS = gradient system structure
%grad specifies start and end value of gradient [gxbegin gx_end; gy..;
%gz..], if grad is not an array
%targets specify the k-space difference to be reached at length n
%targets are defined by deltak-value Kinc (k-space position x,y,z)

n=1;
exitflag(1) =0; exitflag(2) =0; exitflag(3)=0;
while ~( exitflag(1) == 1 && exitflag(2) == 1 && exitflag(3) ==1)
    
    % gx=[]; gy=[]; gz=[];
    % n=n+1;
    % [gx exitflag(1)]=findOptimalGrad1D(n,SYS,grad(1,:),[Kinc(1) n]);
    % [gy exitflag(2)]=findOptimalGrad1D(n,SYS,grad(2,:),[Kinc(2) n]);
    % [gz exitflag(3)]=findOptimalGrad1D(n,SYS,grad(3,:),[Kinc(3) n]);

    n=n+1;
    for ii=1:3
        [g{ii} exitflag(ii)]=findOptimalGrad1D(n,SYS,grad(ii,:),[Kinc(ii) n]);
        if exitflag(ii) == 1
            SYS1=SYS;
            exitflag1=1;
            g0{ii}=g{ii};
            while (exitflag1==1 && (SYS1.SLEW_per_GRT_SI>SYS.SLEW_per_GRT_SI*0.01))
                SYS1.SLEW_per_GRT_SI=SYS1.SLEW_per_GRT_SI*0.8;
                [g{ii} exitflag1]=findOptimalGrad1D(n,SYS1,grad(ii,:),[Kinc(ii) n]);
                if ~(exitflag1==1)
                    g{ii}=g0{ii};
                end
            end
            g0{ii}=g{ii};
        end
    end
end
gx=g{1};
gy=g{2};
gz=g{3};

G=[gx gy gz];
end
