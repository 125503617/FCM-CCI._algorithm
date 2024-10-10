function [center]=getcenter(num,value,k)
A=num(num~=0);%num
B=value(num~=0);%value
C=A.*B;
num1=length(C);
j=round(num1/k);
center=zeros(1,k);
for i=1:k
    if i==1
        center(:,i)=uint8(sum(C((1:j),:))/sum(A((1:j),:)));
    end
    if i>1&&i<k
        center(:,i)=uint8(sum(C((j*(i-1):j*i),:))/sum(A((j*(i-1):j*i),:)));
    end
    if i==k
        center(:,i)=uint8(sum(C((j*(i-1):num1),:))/sum(A((j*(i-1):num1),:)));
    end
end
    


