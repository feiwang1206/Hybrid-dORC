function dif_mat=TVTV2d(sz)

dif_mat0=cell(length(sz),1);
for jj=1:length(sz)
    if jj==1
        count=1;
        clear x_index y_index dif_val
        for kk2=1:sz(2)
            for kk1=1:sz(1)
                x_temp=(kk2-1)*sz(1)+kk1;
                y_temp=x_temp;
                y_temp1=x_temp+1;
                if y_temp>0 && y_temp<=prod(sz) && y_temp1>0 && y_temp1<=prod(sz)
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
        dif_mat0{jj}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));
    elseif jj==2
        count=1;
        clear x_index y_index dif_val
        for kk2=1:sz(2)
            for kk1=1:sz(1)
                x_temp=(kk2-1)*sz(1)+kk1;
                y_temp=x_temp;
                y_temp1=x_temp+sz(1);
                if y_temp>0 && y_temp<=prod(sz) && y_temp1>0 && y_temp1<=prod(sz)
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
        dif_mat0{jj}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));
    end
end
% for jj=1:length(sz)
%     if jj==1
%         count=1;
%         clear x_index y_index dif_val
%         for kk2=1:sz(2)
%             for kk1=1:sz(1)
%                 x_temp=(kk2-1)*sz(1)+kk1;
%                 y_temp=x_temp-1;
%                 y_temp1=x_temp+1;
%                 if y_temp>0 && y_temp<=prod(sz) && y_temp1>0 && y_temp1<=prod(sz)
%                     x_index(count)=x_temp;
%                     y_index(count)=y_temp;
%                     dif_val(count)=-0.5;
%                     count=count+1;
%                     x_index(count)=x_temp;
%                     y_index(count)=y_temp1;
%                     dif_val(count)=0.5;
%                     count=count+1;
%                 end
%             end
%         end
%         dif_mat0{jj}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));
%     elseif jj==2
%         count=1;
%         clear x_index y_index dif_val
%         for kk2=1:sz(2)
%             for kk1=1:sz(1)
%                 x_temp=(kk2-1)*sz(1)+kk1;
%                 y_temp=x_temp-sz(1);
%                 y_temp1=x_temp+sz(1);
%                 if y_temp>0 && y_temp<=prod(sz) && y_temp1>0 && y_temp1<=prod(sz)
%                     x_index(count)=x_temp;
%                     y_index(count)=y_temp;
%                     dif_val(count)=-0.5;
%                     count=count+1;
%                     x_index(count)=x_temp;
%                     y_index(count)=y_temp1;
%                     dif_val(count)=0.5;
%                     count=count+1;
%                 end
%             end
%         end
%         dif_mat0{jj}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));
%     end
% end
dif_mat{1}=dif_mat0{1}*dif_mat0{1};
dif_mat{2}=dif_mat0{1}*dif_mat0{2};
dif_mat{3}=dif_mat0{2}*dif_mat0{1};
dif_mat{4}=dif_mat0{2}*dif_mat0{2};