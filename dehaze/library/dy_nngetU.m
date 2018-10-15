function varargout = dy_nngetU(x, d)

if nargin == 1
    [sx, sy, ~] = size(x);
    varargout = {gpuArray.zeros(sx, sy)};
else
    varargout = {0, 0};
end

end