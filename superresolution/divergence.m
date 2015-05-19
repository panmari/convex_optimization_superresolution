function [ div ] = divergence(u, forward)    
    % Add x and y gradient for divergence.
    grad_x = gradient(u(:,:,1), forward);
    grad_y = gradient(u(:,:,2), forward);
    div = grad_x(:,:,1) + grad_y(:,:,2);
end

