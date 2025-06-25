function image=wavelet_coef_reshape(C,S)
    x=1:S(1,1);
    y=1:S(1,2);
    idx=1:S(1,1)*S(1,2);
    image(x,y)=reshape(C(1,idx),[S(1,1) S(1,2)]);
    for ii=2:length(S)-1
        x1=x(end)+(1:S(ii,1));
        y1=(1:S(ii,2));
        idx=idx(end)+(1:S(ii,1)*S(ii,2));
        image(x1,y1)=reshape(C(1,idx),S(ii,:));
    
        x1=(1:S(ii,1));
        y1=y(end)+(1:S(ii,2));
        idx=idx(end)+(1:S(ii,1)*S(ii,2));
        image(x1,y1)=reshape(C(1,idx),S(ii,:));
    
        x1=x(end)+(1:S(ii,1));
        y1=y(end)+(1:S(ii,2));
        idx=idx(end)+(1:S(ii,1)*S(ii,2));
        image(x1,y1)=reshape(C(1,idx),S(ii,:));
    
        x=x1;
        y=y1;
    end
end
% figure,imagesc(abs(image));colorbar;colormap gray
    
