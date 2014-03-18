
%clear all;
close all;
clc;

im = im2double( rgb2gray( imread('fruits.png') ) );
SRfactor = 2;

% build SRfactor x SRfactor downsample operator
[M, N] = size(im);
% size downsampled img
MD = ceil(double(M)/SRfactor); 
ND = ceil(double(N)/SRfactor);
[x,y] = meshgrid(1:N,1:M);
cols = (x(:)-1)*SRfactor*MD+y(:);
rows = (floor((x(:)-1)/SRfactor))*MD+floor((y(:)-1)/SRfactor)+1;
vals = 1/SRfactor^2*ones(M*N,1);
D = sparse(rows,cols,vals,MD*ND,M*N);

% create input image
% (the low resolution thingy
g = reshape(D*im(:),M/SRfactor,N/SRfactor);

lambda = 1000;

uG = superresolution_Dummy(g,D,lambda);
nearest = imresize(g,[M N],'nearest');

%figure(2,2);
% top left: reconstructed thing
% top right: diff to nearest
% bot left: original image
% bot right: upsampled img using "nearest" operator
disp = [4*uG, 10*(4*uG - nearest) ...
        im, nearest ];
imshow(disp);
