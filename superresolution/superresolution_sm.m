function [u, i, costs] = superresolution_sm(g,D,lambda)
% input: g: double gray scaled image
%        D: downscaling matrix
% lambda: parameter % output: u: inpainted image
% output: u: the optimized image
%         i: number of iterations used.

DEBUG=false;
[MD, ND] = size(g);
[MND, MN] = size(D);
SRfactor = sqrt(MND/MN);
M = MD / SRfactor;
N = ND / SRfactor;
DTg = D' * g(:); % Precomputed here instead of in every iteration.
u = DTg; % initial guess for solution u
t = 0.0001;
max_iterations = 1000;
costs = zeros(max_iterations,1);
delta = 0.000001;

for i=1:max_iterations
    u = reshape(u,M,N);
    % Pad u direction by 2 into every direction.
    % The inital u can be accessed via u_pad(3:end-2, 3:end-2).
    % u[i+1, j] -> u_pad(4:end-1, 3:end-2)
    % u[i, j+1] -> u_pad(3:end-2, 4:end-1)
    u_pad = padarray(u, [2 2], 'symmetric', 'both'); 
    
    % Compute tau (with one padding into each direction)
    tau_sqr = (u_pad(3:end,2:end-1) - u_pad(2:end-1,2:end-1)).^2 + ...
        (u_pad(2:end-1,3:end) - u_pad(2:end-1,2:end-1)).^2;
    tau = (tau_sqr + delta).^(1/2);

    % Compute the elements of the regularization term
    reg_term1 = (2*u(:,:) - u_pad(4:end-1, 3:end-2) - u_pad(3:end-2, 4:end-1)) ...
        ./ tau(2:end-1, 2:end-1);
    reg_term2 = (u(:,:) - u_pad(2:end-3, 3:end-2)) ./ tau(1:end-2, 2:end-1);
    reg_term3 = (u(:,:) - u_pad(3:end-2, 2:end-3)) ./ tau(2:end-1, 1:end-2);
    
    % Put them together to the full regularization term.
    regularization_term = reg_term1 + reg_term2 + reg_term3;
    
    % Derivative of 1/2 || Du - g ||^2
    grad_fitting_term = D' * D * u(:) - DTg;

    gradient_e = regularization_term(:) + lambda*grad_fitting_term;
    % Update u
    u = u(:) - gradient_e*t;
    
    
    costs(i) = sum(tau(:)) + lambda/2 * norm(D*u(:) - g(:));
    if i > 10
      costs_ratio = costs(i) / costs(i - 1);
      if (costs_ratio > 0.9999)
          costs = costs(1:i);
          break;
      end
    end
end
u = reshape(u,M,N);
end