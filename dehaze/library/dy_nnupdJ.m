function varargout = dy_nnupdJ(J, T, U, I, hopts, D)

if nargin == 5
    
    J_new = forward(J, T, U, I, hopts);
    varargout = {J_new};
    
elseif nargin == 6
    
    [DT, DU] = backward(J, T, U, I, hopts, D);
    varargout = {0, DT, DU, 0, 0};
    
else
    
    error('Invalid output number.');
    
end

end

% function J_new = forward(J, T, U, I, opts)
% 
% % rho   = opts(2);
% rho   = 0;
% wsize = opts(3);
% 
% [~, index] = dy_dark_channel(J, wsize);
% x1 = T .* (I+T-1);
% y1 = T .^2 + 1e-6;
% [x2, y2] = dy_place_back(U, index, wsize);
% 
% X = x1 + rho * x2;
% Y = y1 + rho * y2 + 1e-6;
% J_new = X ./ Y;
% 
% end
% 
% function [DT, DU] = backward(J, T, U, I, opts, D)
% 
% % rho   = opts(2);
% rho   = 0;
% wsize = opts(3);
% 
% [~, index] = dy_dark_channel(J, wsize);
% x1 = T .* (I+T-1);
% y1 = T .^2 + 1e-6;
% [x2, y2] = dy_place_back(U, index, wsize);
% 
% X = x1 + rho * x2;
% Y = y1 + rho * y2 + 1e-6;
% 
% DX = I + 2*T - 1;
% DY = 2 * T;
% DT = DX./Y - X.*DY./Y./Y;
% DT = D .* DT;
% DT = sum(DT, 3);
% 
% DU = rho * D ./ Y;
% DU = dy_dark_extract(DU, index);
% 
% end

function J_new = forward(J, T, U, I, opts)

J_new = (I-1) ./ T + 1;

end

function [DT, DU] = backward(J, T, U, I, opts, D)

DT = -(I-1)./(T.^2) .* D;
DT = sum(DT, 3);
DU = 0;

end