function varargout = dy_nnimpt(t, im, d)
im = mean(im, 3);
if nargin == 2
    o1 = forward(t, im);
    varargout = {o1};
else
    o1 = backward(im, d);
    varargout = {o1, 0};
end

end

function tt = forward(t, im)

lambda = 0.1;
beta   = (im-1) .^2;
t = 1 ./ t;
tt = (beta + lambda * t) ./ (beta + lambda);

end


function dt = backward(im, d)

lambda = 0.1;
beta = (im-1) .^2;
dt = d .* lambda ./ (beta + lambda);
dt = -1 ./ (dt.^2 + 1e-8);

end
