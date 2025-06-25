function dif_mat=wTV2d2(sz,sideinfo)
        dif_mat=cell(1,1);
        for kk=1:size(sideinfo,3)
            temp=sideinfo(:,:,kk);
            a=find(temp>0.5);
            count=1;
            for i=1:length(a)
                j=floor(rand*length(a))+1;
                if j>length(a)
                    j=length(a);
                end
                if j==i
                    j=length(a);
                end
                x_index(count)=a(i);
                y_index(count)=a(i);
                dif_val(count)=-1;
                count=count+1;
                x_index(count)=a(i);
                y_index(count)=a(j);
                dif_val(count)=1;
                count=count+1;
            end
        end
        dif_mat{1}=sparse(x_index,y_index,dif_val,prod(sz),prod(sz));

