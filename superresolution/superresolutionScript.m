
clear all;
close all;
clc;
SHOW_COSTS = true;
%img_name = 'fruits.png';
img_name = 'grass_small.png';
im = imread(img_name);
if size(im, 3) == 3
    im = rgb2gray( im );
end
im = im2double(im);
%im = zeros(100);
%im(45:65, 45:65) = 0.25;
% Resize to about a fifth for debugging
%im = imresize(im, [40 70]);
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

lambdas = 100; %[100:1000:10000];

SSEs = zeros(size(lambdas));
ssim_vals = zeros(size(lambdas));
nearest = imresize(g,[M N],'nearest');

for i = 1:length(lambdas)
    lambda = lambdas(i);
    [uG, iterations, costs] = superresolution_dual_sm(g,D,lambda, 2, img_name);
    if SHOW_COSTS
        figure;
        plot(costs);
        title(['Lambda=', num2str(lambda)]);
    end
    figure;
    title(['Resize factor: ', num2str(SRfactor)]);
    % top left: reconstructed thing
    % top right: upsampled img using "nearest" operator
    % bot left: diff to nearest
    % bot right: original image
    error = (uG - im).^2;
    ssim_vals(i) = ssim(uG, im);
    SSEs(i) = sum(error(:));
    fprintf('Lambda = %f, SSD: %f, SSIM: %f, iterations=%i \n', ...
        lambda, SSEs(i), ssim_vals(i), iterations)
    displayed_images = [uG, nearest; ...
                        10 * error, im];
    imshow(displayed_images);
    drawnow
end
%% Visualize error under different lambdas
error_nearest = (nearest - im).^2;

figure, plot(lambdas, SSEs, ...
    lambdas, repmat(sum(error_nearest(:)), size(SSEs)));
xlabel('Lambda');
ylabel('SSD');