function varargout = dy_nnpad(x, r, d)

if nargin == 2
    y = padarray(x, [r,r], 'symmetric');
    varargout = {y};
else
    [sx,sy,~] = size(d);
    d(r+1:r+r,:,:,:) = d(r+1:r+r,:,:,:) + d(r:-1:1,:,:,:);
    d(:,r+1:r+r,:,:) = d(:,r+1:r+r,:,:) + d(:,r:-1:1,:,:);
    d(sx-2*r+1:sx-r,:,:,:) = d(sx-2*r+1:sx-r,:,:,:) + d(sx:-1:sx-r+1,:,:,:);
    d(:,sy-2*r+1:sy-r,:,:) = d(:,sy-2*r+1:sy-r,:,:) + d(:,sy:-1:sy-r+1,:,:);
    y = d(r+1:sx-r,r+1:sy-r,:,:);
    varargout = {y, 0};
end

end