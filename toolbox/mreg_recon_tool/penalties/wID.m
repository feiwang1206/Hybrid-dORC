function dif_mat=wID(sz,sideinfo)
    dif_mat=cell(length(sz),1);
    for jj=1:length(sz)
        for kk=1:prod(sz)
            x_index(kk)=kk;
            y_index(kk)=kk;
            dif_val(kk)=sideinfo(kk);
        end
        dif_mat{jj}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));
    end

