
%clear all;
close all;
%clc;
im = imread('fruits.png');
%im = imread('grass_small.png');
if size(im, 3) == 3
    im = rgb2gray( im );
end
im = im2double(im);
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
% (the low resolution thingy)
g = reshape(D*im(:),M/SRfactor,N/SRfactor);

lambdas = [0.1, 1, 100:200:2000];
lambdas = 100:50:1000;

errors = zeros(size(lambdas));
nearest = imresize(g,[M N],'nearest');

for i = 1:length(lambdas)
    lambda = lambdas(i);
    [uG, iterations, costs] = superresolution_sm(g,D,lambda);

    %figure;
    title(['Resize factor: ', num2str(SRfactor)]);
    % top left: reconstructed thing
    % top right: upsampled img using "nearest" operator
    % bot left: diff to nearest
    % bot right: original image
    error = (uG - im).^2;
    sum_error = sum(error(:));
    errors(i) = sum_error;
    fprintf('Lambda = %f, SSE: %f, iterations=%i \n', lambda, sum_error, iterations)
    displayed_images = [uG, nearest; ...
                        10 * error, im];
    imshow(displayed_images);
end
%% Visualize error under different lambdas
error_nearest = (nearest - im).^2;
loglog(lambdas, errors, lambdas, repmat(sum(error_nearest(:)), size(errors)));
