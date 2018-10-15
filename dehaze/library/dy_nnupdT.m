function varargout = dy_nnupdT(J, U, I, hopts, D)

if nargin == 4
    
    T = forward(J, U, I, hopts);
    varargout = {T};
    
elseif nargin == 5
    
    [DJ, DU] = backward(J, U, I, hopts, D);
    varargout = {DJ, DU, 0, 0};
    
else
    
    error('Invalid output number.');
    
end

end

function T = forward(J, U, I, opts)

alpha = opts(1);
wsize = opts(3);

DI = dy_dark_channel(I, wsize);

x1 = (J-1) .* (I-1);
x1 = sum(x1, 3);
x2 = (U-1) .* (DI-1);
y1 = (J-1) .^2;
y1 = sum(y1, 3);
y2 = (U-1) .^2;

X = x1 + alpha * x2;
Y = y1 + alpha * y2 + 1e-6;

T = X ./ Y;
T = max(min(T, 1), 0.1);

end

function [DJ, DU] = backward(J, U, I, opts, D)

alpha = opts(1);
wsize = opts(3);

DI = dy_dark_channel(I, wsize);

x1 = (J-1) .* (I-1);
x1 = sum(x1, 3);
x2 = (U-1) .* (DI-1);
y1 = (J-1) .^2;
y1 = sum(y1, 3);
y2 = (U-1) .^2;

X = x1 + alpha * x2;
Y = y1 + alpha * y2 + 1e-6;

T = X ./ Y;
D = D .* (T>0.1) .* (T<1);

DX = I - 1;
DY = 2 * (J-1);
DJ = DX./Y - X.*DY./Y./Y;
DJ = D .* DJ;

DX = alpha * (DI-1);
DY = 2*alpha * (U-1);
DU = DX./Y - X.*DY./Y./Y;
DU = D .* DU;

end