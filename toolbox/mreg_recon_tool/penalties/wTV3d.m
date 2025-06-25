function dif_mat=wTV3d(sz,sideinfo)
        dif_mat=cell(length(sz),1);
        for jj=1:length(sz)
            if jj==1
                count=1;
                clear x_index y_index dif_val
                for kk2=1:sz(2)
                    for kk1=1:sz(1)
                        x_temp=(kk2-1)*sz(1)+kk1;
                        y_temp=(kk2-1)*sz(1)+kk1;
                        if x_temp<=prod(sz) && y_temp<=prod(sz)
                            x_index(count)=x_temp;
                            y_index(count)=y_temp;
                            dif_val(count)=-1*sideinfo(x_index(count));
                            count=count+1;
                        end
                        x_temp=(kk2-1)*sz(1)+kk1;
                        y_temp=(kk2-1)*sz(1)+kk1;
                        if x_temp<=prod(sz) && y_temp<=prod(sz)
                            x_index(count)=x_temp;
                            y_index(count)=y_temp;
                            dif_val(count)=1*sideinfo(x_index(count));
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
                        y_temp=(kk2-1)*sz(1)+kk1;
                        if x_temp<=prod(sz) && y_temp<=prod(sz)
                            x_index(count)=x_temp;
                            y_index(count)=y_temp;
                            dif_val(count)=-1*sideinfo(x_index(count));
                            count=count+1;
                        end
                        x_temp=(kk2-1)*sz(1)+kk1;
                        y_temp=(kk2)*sz(1)+kk1;
                        if x_temp<=prod(sz) && y_temp<=prod(sz)
                            x_index(count)=x_temp;
                            y_index(count)=y_temp;
                            dif_val(count)=1*sideinfo(x_index(count));
                            count=count+1;
                        end
                    end
                end
                dif_mat{jj}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));
            end
        end
