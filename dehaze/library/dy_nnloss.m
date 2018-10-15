function varargout = dy_nnloss(x, l, t, d)

switch t
    case 1
        if nargin == 3
%             plot_image(x, l);
            r = x - l;
            y = sum(abs(r(:))) / 2;
            varargout = {y};
        elseif nargin == 4
            y = d .* sign((x - l));
            varargout = {y, 0, 0};
        else
            error('Invalid input to loss function.')
        end
    case 2
        if nargin == 3
%             plot_image(x, l);
            r = x - l;
            y = sum(r(:).^2) / 2;
            varargout = {y};
        elseif nargin == 4
            y = d .* (x - l);
            varargout = {y, 0, 0};
        else
            error('Invalid input to loss function.')
        end
        
    otherwise
        error('No such error function.');
end

end

function plot_image(x, l)

if size(x,3) == 3
    return;
else
    image = cat(1, x(:,:,:,1), l(:,:,:,1));
    figure(10), imshow(image);
end

end

