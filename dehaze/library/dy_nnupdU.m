function varargout = dy_nnupdU(J, T, I, hopts, D)

if nargin == 4
    
    U = forward(J, T, I, hopts);
    varargout = {U};
    
elseif nargin == 5
    
    [DJ, DT] = backward(J, T, I, hopts, D);
    varargout = {DJ, DT, 0, 0};
    
else
    
    error('Invalid input number.');
    
end

end

function U = forward(J, T, I, opts)

alpha = opts(1);
rho   = opts(2);
wsize = opts(3);

DJ = dy_dark_channel(J, wsize);
DI = dy_dark_channel(I, wsize);
c1 = alpha*T.*(DI+T-1) + rho*DJ;
c2 = alpha*T.^2 + rho + 1e-6;
U = c1 ./ c2;

end

function [DJ, DT] = backward(J, T, I, opts, D)

alpha = opts(1);
rho   = opts(2);
wsize = opts(3);

[darkJ, index] = dy_dark_channel(J, wsize);
darkI = dy_dark_channel(I, wsize);

X = alpha*T.*(darkI+T-1) + rho*darkJ;
Y = alpha*T.^2 + rho + 1e-6;

% compute dl/dJ
DJ = rho * D ./ Y;
DJ = dy_place_back(DJ, index, wsize);
% compute dl/dT
DX = 2*alpha * (2*T+darkI-1);
DY = 2*alpha * T;
DT = DX./Y - X.*DY./Y./Y;
DT = D .* DT;

end