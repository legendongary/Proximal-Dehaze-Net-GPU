function varargout = dy_nnperloss(x, l, d)
%DY_NNPERLOSS perceptual loss

global vggnet;

vgg_x = vl_simplenn(vggnet, x, []);
vgg_l = vl_simplenn(vggnet, l, []);
per_x = vgg_x(end).x;
per_l = vgg_l(end).x;
[h, w, c, ~] = size(per_x);

if nargin == 2
    y1 = 0.5 * sum((per_x(:)-per_l(:)).^2) / (h*w*c);
    varargout = {y1};
else
    dx = d .* (per_x - per_l) / (h*w*c);
    vgg_x = vl_simplenn(vggnet, x, dx, vgg_x);
    y1 = vgg_x(1).dzdx;
    varargout = {y1, 0};
end

end

