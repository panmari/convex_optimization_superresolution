function [u, i] = superresolution_sm(g,D,lambda)
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
u = D'*g(:); % initial guess for solution u
DTg = D' * g(:); % Precomputed here instead of in every iteration.
t = 0.01;
max_iterations = 3000;
gradient_norms = zeros(max_iterations,1);
delta = 0.000001;

for i=1:max_iterations
    u = reshape(u, M,N);
    % only gradient in x direction so far...
    % Pad u direction by 2 into every direction.
    u_pad = padarray(u, [2 2], 'symmetric', 'both'); 
    
    tau_sqr = (u_pad(3:end,2:end-1) - u_pad(2:end-1,2:end-1)).^2 + ...
        (u_pad(2:end-1,3:end) - u_pad(2:end-1,2:end-1)).^2;
    tau = (tau_sqr + delta).^(1/2);

    reg_term1 = (2*u(:,:) - u_pad(4:end-1, 3:end-2) - u_pad(3:end-2, 4:end-1))./ tau(2:end-1, 2:end-1);
    reg_term2 = (u(:,:) - u_pad(2:end-3, 3:end-2)) ./ tau(1:end-2, 2:end-1);
    reg_term3 = (u(:,:) - u_pad(3:end-2, 2:end-3)) ./ tau(2:end-1, 1:end-2);

    regularization_term = reg_term1 + reg_term2 + reg_term3;
    
    % Derivative of 1/2 || Du - g ||^2
    grad_fitting_term = D' * D * u(:) - DTg;

    gradient_e = regularization_term(:) + lambda*grad_fitting_term(:);
    % Update u
    u = u(:) - gradient_e*t;
    
    gradient_norms(i) = norm(gradient_e);
    if i > 500
      gradient_norm_ratio = gradient_norms(i) / gradient_norms(i - 1);
      if (gradient_norm_ratio > 0.9999)
          break;
      end
    end
end
if (DEBUG)
    semilogy(gradient_norms);
end
u = reshape(u,M,N);

end