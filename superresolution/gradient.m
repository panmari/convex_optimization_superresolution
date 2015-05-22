function [ grad ] = gradient(u, forward)
    if forward
        % forward gradients, n_i - n_{i - 1}
        u_pad = padarray(u, [1 1], 'symmetric', 'pre'); 
        u_x_grad = u - u_pad(1:end-1, 2:end);
        u_y_grad = u - u_pad(2:end, 1:end-1);
    else
        % backward gradients, n_i - n_{i + 1}
        u_pad = padarray(u, [1 1], 'symmetric', 'post');
        u_x_grad = u - u_pad(2:end, 1:end-1);
        u_y_grad = u - u_pad(1:end-1, 2:end);
    end
    
    grad = u_x_grad;
    grad(:,:,2) = u_y_grad;
end

