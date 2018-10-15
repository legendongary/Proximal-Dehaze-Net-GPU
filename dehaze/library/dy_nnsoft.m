function [o1, o2] = dy_nnsoft(x, t, d)

if nargin == 2
    o1 = sign(x) .* max(abs(x)-t, 0);
else
    o1 = abs(x) > t;
    o1 = o1 .* d;
    o2 = abs(x) > t;
    o2 = o2 .* sign(x) .* d;
end

end