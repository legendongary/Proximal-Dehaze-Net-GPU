function y = dy_nnzero(x, d)
% DY_NNZERO set zero elements non-zero

if nargin==1
    iszero = (x==0);
    y = x + 1e-6 * iszero;
else
    y = d;
end

end

