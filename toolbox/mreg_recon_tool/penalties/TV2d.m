function dif_mat=TV2d(sz)

dif_mat=cell(length(sz),1);
for jj=1:length(sz)
    if jj==1
        count=1;
        clear x_index y_index dif_val
        for kk2=1:sz(2)
            for kk1=1:sz(1)
                x_temp=(kk2-1)*sz(1)+kk1;
                y_temp=x_temp;
                y_temp1=x_temp+1;
                if y_temp>0 && y_temp<=prod(sz) && kk1<sz(1)
                    x_index(count)=x_temp;
                    y_index(count)=y_temp;
                    dif_val(count)=-1;
                    count=count+1;
                    x_index(count)=x_temp;
                    y_index(count)=y_temp1;
                    dif_val(count)=1;
                    count=count+1;
                end
            end
        end
        dif_mat{jj}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));
    elseif jj==2
        count=1;
        clear x_index y_index dif_val
        for kk2=1:sz(2)
            for kk1=1:sz(1)
                x_temp=(kk2-1)*sz(1)+kk1;
                y_temp=x_temp;
                y_temp1=x_temp+sz(1);
                if y_temp>0 && y_temp<=prod(sz) && kk2<sz(2)
                    x_index(count)=x_temp;
                    y_index(count)=y_temp;
                    dif_val(count)=-1;
                    count=count+1;
                    x_index(count)=x_temp;
                    y_index(count)=y_temp1;
                    dif_val(count)=1;
                    count=count+1;
                end
            end
        end
        dif_mat{jj}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));
    end
end

