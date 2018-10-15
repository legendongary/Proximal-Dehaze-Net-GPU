function y = dy_nnupsample(x, d)

if nargin==1
    y = dy_upsample(x);
    
elseif nargin==2
    y = vl_nnpool(d, [2,2], 'stride', 2, 'method', 'avg');
else
    error('Invalid input.')
end


end