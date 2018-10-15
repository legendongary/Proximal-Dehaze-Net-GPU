function layer = ImpT(obj, varargin)

layer = Layer(@dy_nnimpt, obj, varargin{:});

end