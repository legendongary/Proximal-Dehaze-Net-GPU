clear;clc;close all;

image = imread('hazy.png');

tic;

% network trained on our dataset
[resim, trans] = cnn_ours_eval(image);

% network trained on RESIDE dataset with 1 stage
% [resim, trans] = cnn_reside_s1_eval(image);

% network trained on RESIDE dataset with 2 stages
% [resim, trans] = cnn_reside_s2_eval(image);

toc;

figure, imshow(image)
figure, imshow(resim)
figure, imshow(trans)