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
% K_a is upper bound.
K_a = 8;
% Should be rather small
tau = 1e-3; 
% Should be rather large
sigma = 1/(tau*K_a);

theta = 1;

% Initialize all variables.
y_n = zeros(M, N, 2);
x_n = zeros(M, N);
% TODO: better initial guess?
xbar_n = reshape(D' * g(:), M, N);
DTD = D' * D;
% Used for computing x_n1, instead of inverting it here we use slash
% operator below.
x_n1_matrix = speye(size(DTD)) + tau * lambda * DTD;
%x_n1_divisor = inv(x_n1_matrix);
x_n1_right_summand = reshape(tau * lambda * D' * g(:), M, N);
% Maximum number of iterations learnt
max_iterations = 2000;
% For displaying changes in costs later on.
costs = zeros(max_iterations,1);

for i=1:max_iterations
    % Compute y_n1
    y_n1_nominator = y_n + sigma * gradient(xbar_n, true);
    y_n1_denominator = max(1,  sqrt(sum(y_n1_nominator.^2, 3)));
    y_n1(:,:,1) = y_n1_nominator(:,:,1) ./ y_n1_denominator;
    y_n1(:,:,2) = y_n1_nominator(:,:,2) ./ y_n1_denominator;

    % Compute x_n1
    div_y_n1 = divergence(y_n1, false);
    x_n1 = x_n1_matrix \ reshape(x_n - tau * div_y_n1 + x_n1_right_summand, [], 1);
    x_n1 = reshape(x_n1, M, N);
    
    % Compute xbar_n1
    xbar_n1 = x_n1 + theta * (x_n1 - x_n);
    
    % Adapt for next timestep, n + 1 -> n
    y_n = y_n1;
    x_n = x_n1;
    if mod(i, 50) == 1
       disp('Update!');
       imagesc([xbar_n, imresize(g,[M N],'nearest')]);
       drawnow
    end
    xbar_n = xbar_n1;
    costs(i) = energy_term_for(xbar_n, g, D, lambda);
end
u = reshape(xbar_n,M,N);
end

function E_u = energy_term_for(u, g, D, lambda)
    similarity = D * u(:) - g(:);
    smoothness = gradient(u, true);
    smoothness_sqr = smoothness.^2;
    smoothness_energy = sum(sqrt(smoothness_sqr(:)));
    E_u = (lambda*0.5)*l2_norm(similarity) + smoothness_energy;
end

% Computes the l2 norm of a matrix/vector, i. e. squares every element,
% then sums up over them and takes the square root.
function n = l2_norm(X)
    n = X.^2;
    n = sum(n(:));
    n = sqrt(n);
end