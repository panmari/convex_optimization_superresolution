function [ div ] = divergence(u, forward)    
    % The inital u can be accessed via u_pad(2:end-1, 2:end-1).
    if forward
        % forward gradients, n_i - n_{i - 1}
        u_pad = padarray(u, [1 1], 'symmetric', 'pre'); 
        u_x_grad = u_pad(2:end,2:end) - u_pad(1:end-1, 2:end);
        u_y_grad = u_pad(2:end,2:end) - u_pad(2:end, 1:end-1);
    else
        % backward gradients, n_i - n_{i + 1}
        u_pad = padarray(u, [1 1], 'symmetric', 'post');
        u_x_grad = u_pad(1:end-1,1:end-1) - u_pad(2:end, 1:end-1);
        u_y_grad = u_pad(1:end-1,1:end-1) - u_pad(1:end-1, 2:end);
    end

    % Combine x and y gradient to divergence.
    div = u_x_grad + u_y_grad;
end

