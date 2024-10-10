%%
% This code writted by Xiaomiao Tao at Lanzhou Jiaotong University,Lanzhou China.
%% Intialization
clc
clear all
close all;
tic
%% Parameters
error=0.001; 
density=0;
cluster_num=4; 
max_iter=100; 
m=2; 
sigma_d=5;
sigma_r=2.5;
k=3;
h=5;
%% Normalization 
f_uint8=imread('5.bmp');
f=double(f_uint8)/255;
figure,imshow(f);
[row,col,depth]=size(f);
N=row*col;
%% Construct mixed noise
f_n=imnoise(f,'gaussian',0,density);
f_n=imnoise(f_n,'salt & pepper',density);
f_n=imnoise(f_n,'speckle',density);
figure,imshow(f_n);
%% Process fast bilateral filtering
f_bilateral=zeros(row, col, depth);
for i=1:depth
    f_bilateral(:,:,i) = bilateralFilter(f_n(:,:,i),[],0,1,sigma_d,sigma_r);
end
figure,imshow(f_bilateral);
%% Obtain the number of gray level, gray value and prediction center of filtered image
center=zeros(1,cluster_num,depth);
f_bilateral_uint8=uint8(f_bilateral*255);
for i=1:depth
    [num,value]=imhist(f_bilateral_uint8(:,:,i));
    center(:,:,i)=getcenter(num,value,cluster_num); 
end
center=center/255;
%% Pixel reshaping
f_padded=padarray(f_n,[k k],'replicate'); 
sigma=zeros(row,col,depth);
for d=1:depth
    for i=-k:k
        for j=-k:k
            if i==0 && j==0
                continue
            end
            sigma=sigma+abs((f_padded(i+k+1:end+i-k,j+k+1:end+j-k,d)-f_n));
        end
    end
end
sigma=sigma./(2*k+1)^2;
all_pixel=repmat(reshape(f_n, N, 1, depth), [1 cluster_num 1]);
all_pixel_bi=repmat(reshape(f_bilateral, N, 1, depth), [1 cluster_num 1]);
sigma=repmat(reshape(sigma, N, 1, depth), [1 cluster_num 1]);
b=sigma;
a=1-b;
%% Allocate memory for the objective function
J=zeros(1,max_iter);
%% FCM Clustering
for iter=1:max_iter
    A=(all_pixel-repmat(center,[N,1,1])).^2;
    B=(all_pixel_bi-repmat(center,[N,1,1])).^2;
    
    U=sum((a.*A+b.*B),3).^(-1/(m-1))./sum(sum((a.*A+b.*B),3).^(-1/(m-1)),2);
    U=reshape(U,[row,col,cluster_num]);
    U=repmat(reshape(U,[N,cluster_num]),[1,1,depth]);
    U_m=U.^m;
    
    J(iter)=sum(sum(sum(U_m.*(a.*A+b.*B))));
    
    center=sum(U_m.*(a.*all_pixel+b.*all_pixel_bi))./sum(U_m.*(a+b));
    fprintf('µÚ%d´Î¾ÛÀà£¬J = %f\n',iter,J(iter));
    if iter > 1 && abs(J(iter) - J(iter - 1)) <= error
        fprintf('Objective function is converged\n');
        break;
    end
    if iter > 1 && iter == max_iter && abs(J(iter) - J(iter - 1)) > error
        fprintf('Objective function is not converged. Max iteration reached\n');
        break;
    end
end
U=reshape(U(:,:,1),[row,col,cluster_num]);
U_padded=padarray(U,[h h],'replicate');
sigma1=zeros(row,col,cluster_num);
    for depth2=1:cluster_num 
       for i=-h:h
          for j=-h:h
           sigma1(:,:,depth2)=sigma1(:,:,depth2)+U_padded(i+h+1:end+i-h,j+h+1:end+j-h,depth2);
           end
       end
    end
U=sigma1./repmat(sum(sigma1,3),[1,1,cluster_num]);
U=repmat(reshape(U,[N,cluster_num]),[1,1,depth]);
[~, cluster_indice] = max(U(:,:,1), [], 2);
cluster_indice = reshape(cluster_indice, [row, col]);
% Visualize all labels
FCM_result = Label_image(f_uint8, reshape(cluster_indice, row, col));%f_bilateral_uint8
figure, imshow(FCM_result);
title('Segmentation result');
toc
% Visualize objective function
%figure, plot(J);
figure,plot(1:iter, J(1:iter));
title('Objective function J');