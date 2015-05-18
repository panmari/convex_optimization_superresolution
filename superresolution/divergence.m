function [ div ] = divergence( u )
    % TODO: Adapt amount of padding, prolly 1 is enough.
    
    % The inital u can be accessed via u_pad(2:end-1, 2:end-1).
    u_pad = padarray(u, [1 1], 'symmetric', 'both'); 
    % Gradients (with one padding into each direction)
    u_x_grad = u_pad(2:end,2:end-1) - u_pad(1:end-2,2:end-1);
    u_y_grad = u_pad(2:end-1,2:end) - u_pad(2:end-1,1:end-2);

    % Combine x and y gradient into one big matrix.
    div = u_x_grad;
    div(:,:,2) = u_y_grad;
end

