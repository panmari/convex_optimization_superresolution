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

% Constants
tau = 0.01;
sigma = 1/0.09;
theta = 0.3;

% Initialize all variables.
y_n = zeros(M, N);
x_n = zeros(M, N);
% TODO: better initial guess?
xbar_n = zeros(M, N);
DTD = D' * D;
% Used for computing x_n1, instead of inverting it here we use slash
% operator below.
x_n1_divisor = inv(speye(size(DTD)) + tau * lambda * DTD);
x_n1_right_summand = reshape(tau * lambda * D' * g(:), M, N);
% Maximum number of iterations learnt
max_iterations = 1000;
% For displaying changes in costs later on.
costs = zeros(max_iterations,1);

for i=1:max_iterations
    div_xbar_n = divergence(xbar_n, true);
    % TODO: Norm
    y_n1 = y_n + sigma * div_xbar_n / max(1, norm(y_n + sigma * div_xbar_n));
    div_y_n1 = divergence(y_n1, false);
    x_n1 = x_n1_divisor * reshape(x_n + tau * div_y_n1 + x_n1_right_summand, [], 1);
    x_n1 = reshape(x_n1, M, N);
    xbar_n1 = x_n1 + theta * (x_n1 - x_n);
    
    % Adapt for next timestep, n + 1 -> n
    y_n = y_n1;
    x_n = x_n1;
    xbar_n = xbar_n1;
end
u = reshape(xbar_n,M,N);
end