function [u] = superresolution_Dummy(g,D,lambda)
% input: g: double gray scaled image
%        D: downscaling matrix
% lambda: parameter % output: u: inpainted image

% dummy output
[MD, ND] = size(g);
[MND, MN] = size(D);
SRfactor = sqrt(MND/MN);
M = MD / SRfactor;
N = ND / SRfactor;
u = D'*g(:); % initial guess for solution u
t = 0.0001;
for i=1:3
    u = reshape(u, M,N);
    % only gradient in x direction so far...
    % 2*u(x,y) - u(x, y + 1) - u(x, y - 1) 
    grad_reg_term_x = 2*(2*u(:,:) - [u(:,2:end) u(:, end - 1)] - [u(:,2) u(:,1:end-1)]);
    grad_reg_term_y = 2*(2*u(:,:) - [u(2:end,:); u(end - 1,:)] - [u(2,:); u(1:end-1,:)]);
    grad_reg_term = grad_reg_term_x + grad_reg_term_y;
    imtool(5*grad_reg_term);
    grad_other_term = D*u(:) - g(:);
    % constrain all 3 created pixels the same way as original:
    %grad_other_term_ext = repmat(grad_other_term, [1,4])';
    % alternatively dont constrain them at all:
    grad_other_term_ext = [grad_other_term zeros(length(grad_other_term), 3)]';
    
    % add up to get gradient of whole energy term
    gradient_e = grad_reg_term(:) + lambda*grad_other_term_ext(:);
    u = u(:) - gradient_e*t;
end
u = reshape(u,M,N);

end