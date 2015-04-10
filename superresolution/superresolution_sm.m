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
t = 0.01;
max_iterations = 3000;
gradient_norms = zeros(max_iterations,1);

for i=1:max_iterations
    u = reshape(u, M,N);
    % only gradient in x direction so far...
    % 2*u(x,y) - u(x, y + 1) - u(x, y - 1) 
    grad_reg_term_x = 2*(2*u(:,:) - [u(:,2:end) u(:, end - 1)] - [u(:,2) u(:,1:end-1)]);
    grad_reg_term_y = 2*(2*u(:,:) - [u(2:end,:); u(end - 1,:)] - [u(2,:); u(1:end-1,:)]);
    grad_reg_term = grad_reg_term_x + grad_reg_term_y;
    
    grad_other_term = D*u(:) - g(:);
    % multiply additionally with D for inner gradient (chain rule!)
    grad_other_term_ext2 = zeros(M,N);
    %grad_other_term = diag(D).*grad_other_term;
    
    % constrain all 3 created pixels the same way as original:
    grad_other_term_ext = repmat(grad_other_term, [1,2])';
    % alternatively don't constrain them at all:
    % grad_other_term_ext = [grad_other_term zeros(length(grad_other_term), 1)]';
    grad_other_term_ext = reshape(grad_other_term_ext(:), M, N/2);
   
    for row=1:N
        %if (mod(row, 2) == 0)
            grad_other_term_ext2(:,row) = grad_other_term_ext(:,ceil(row/2));
        %end
    end
    
    % add up to get gradient of whole energy term
    if (DEBUG)
        debug_img(:,:,1) = abs(reshape(grad_reg_term(:), M,N));
        debug_img(:,:,2) = abs(reshape(lambda*grad_other_term_ext2(:), M,N));
        debug_img(:,:,3) = zeros(M,N);
        imtool(3*debug_img);
    end

    gradient_e = grad_reg_term(:) + lambda*grad_other_term_ext2(:);
    % Update u
    u = u(:) - gradient_e*t;
    
    gradient_norms(i) = norm(gradient_e);
    if i > 2
      gradient_norm_ratio = gradient_norms(i) / gradient_norms(i - 1);
      if (gradient_norm_ratio > 0.99)
          break;
      end
    end
end
if (DEBUG)
    semilogy(gradient_norms);
end
u = reshape(u,M,N);

end