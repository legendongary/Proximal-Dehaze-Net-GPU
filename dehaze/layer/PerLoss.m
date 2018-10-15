function layer = PerLoss(obj, varargin)

layer = Layer(@dy_nnperloss, obj, varargin{:});

end

