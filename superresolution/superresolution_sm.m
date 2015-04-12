function [u, i, costs] = superresolution_sm(g, D, lambda, norm_used, img_name)
% input: g: double gray scaled image
%        D: downscaling matrix
%        lambda: parameter % output: u: inpainted image
%        norm_used: norm for the regularization term, either 0 (ignore
%        regularization term), 1 or 2.
% output: u: the optimized image
%         i: number of iterations used.
[MD, ND] = size(g);
[MND, MN] = size(D);
SRfactor = sqrt(MND/MN);
M = MD / SRfactor;
N = ND / SRfactor;
DTg = D' * g(:); % Precomputed here instead of in every iteration.
u = DTg; % initial guess for solution u

% Some parameters for learning rate.
if lambda < 1000
    starting_t = 0.01;
    minimal_t = 0.001;
elseif lambda < 5000
    starting_t = 0.001;
    minimal_t =  0.0001;
else
    starting_t = 0.0001;
    minimal_t =  0.00001;
end
t = starting_t;
% Maximum number of iterations learnt
max_iterations = 10000;
% For displaying changes in costs later on.
costs = zeros(max_iterations,1);
delta = 1e-6;
for i=1:max_iterations
    % Decrease learning rate for later iterations.
    if mod(i, 1000) == 0 && t > minimal_t
        t = t / 10;
    end
    u = reshape(u,M,N);
    
    if mod(i, 1000) == 1
        [~, name] = fileparts(img_name);
        imwrite(u, ... 
            sprintf('output/%s_lambda_%d_iteration_%d.png', ...
            name, lambda, i - 1));
    end

    % Pad u direction by 2 into every direction.
    % The inital u can be accessed via u_pad(3:end-2, 3:end-2).
    % u[i+1, j] -> u_pad(4:end-1, 3:end-2)
    % u[i, j+1] -> u_pad(3:end-2, 4:end-1)
    u_pad = padarray(u, [2 2], 'symmetric', 'both'); 
    
    if norm_used == 2
        % Compute tau (with one padding into each direction)
        tau_term1 = u_pad(3:end,2:end-1) - u_pad(2:end-1,2:end-1);
        tau_term2 = u_pad(2:end-1,3:end) - u_pad(2:end-1,2:end-1);
        tau_sqr = tau_term1.^2 + tau_term2.^2;
        tau = (tau_sqr + delta).^(1/2);

        % Compute the elements of the regularization term
        reg_term1 = (2*u(:,:) - u_pad(4:end-1, 3:end-2) - u_pad(3:end-2, 4:end-1)) ...
            ./ tau(2:end-1, 2:end-1);
        reg_term2 = (u(:,:) - u_pad(2:end-3, 3:end-2)) ./ tau(1:end-2, 2:end-1);
        reg_term3 = (u(:,:) - u_pad(3:end-2, 2:end-3)) ./ tau(2:end-1, 1:end-2);
    elseif norm_used == 1
        reg_term1 = -sign(u_pad(4:end-1, 3:end-2) - u(:,:)) - sign(u_pad(3:end-2, 4:end-1) - u(:,:));
        reg_term2 = sign(u(:,:) - u_pad(2:end-3, 3:end-2));
        reg_term3 = sign(u(:,:) - u_pad(3:end-2, 2:end-3));
    elseif norm_used == 0
        reg_term1 = 0;
        reg_term2 = 0;
        reg_term3 = 0;
    end
    
    % Put them together to the full regularization term.
    regularization_term = reg_term1 + reg_term2 + reg_term3;
    
    % Derivative of 1/2 || Du - g ||^2
    grad_fitting_term = D' * D * u(:) - DTg;

    gradient_e = regularization_term(:) + lambda*grad_fitting_term;
    % Update u
    u = u(:) - gradient_e*t;
    
    % Compute costs and break if change is too little, not used.
    costs(i) = sum(tau(:)) + lambda/2 * norm(D*u(:) - g(:));
%     if i > 10
%       costs_ratio = costs(i) / costs(i - 1);
%       if (1 - costs_ratio < 1e-10)
%           costs = costs(1:i);
%           break;
%       end
%     end
end
u = reshape(u,M,N);
end