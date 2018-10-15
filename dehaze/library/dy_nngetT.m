function varargout = dy_nngetT(x, d)

if nargin == 1
    y = x(:,:,1,:);
    y(:) = 1;
    varargout = {y};
else
    varargout = {0, 0};
end

end