function [u, i, costs] = superresolution_dual_sm(g, D, lambda, norm_used, img_name)
% input: g: double gray scaled image
%        D: downscaling matrix
%        lambda: parameter % output: u: inpainted image
%        norm_used: norm for the regularization term, either 0 (ignore
%        regularization term), 1 or 2.
% output: u: the optimized image
%         i: number of iterations used.
%         costs: value of cost function at every iteraton.
[MD, ND] = size(g);
[MND, MN] = size(D);
SRfactor = sqrt(MND/MN);
M = MD / SRfactor;
N = ND / SRfactor;

% Initialize all variables.
u = g;
y_n = zeros(size(g));
x_n = zeros(size(g));
xbar_n = zeros(size(g));

% Maximum number of iterations learnt
max_iterations = 100;
% For displaying changes in costs later on.
costs = zeros(max_iterations,1);

tau = 0.01;
sigma = 1/0.09;
for i=1:max_iterations
    div_xbar_n = divergence(xbar_n);
    y_n1 = y_n + sigma * div_xbar_n / max(1, norm(yn + sigma * div_xbar_n));
    x_n1 = x_n + tau * divergence(y_n1) + tau * lambda * D.* g / ( 1 + tau * lambda * D);
    xbar_n1 = x_n1 + theta * (xn_1 - x_n);
    
    % Adapt for next timestep, n + 1 -> n
    y_n = y_n1;
    x_n = x_n1;
    xbar_n = xbar_n1;
end
u = reshape(u,M,N);
end