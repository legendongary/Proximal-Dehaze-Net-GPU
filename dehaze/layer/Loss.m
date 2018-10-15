function layer = Loss(obj, varargin)

layer = Layer(@dy_nnloss, obj, varargin{:});

end

