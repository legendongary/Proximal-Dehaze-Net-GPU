function layer = GetU(obj, varargin)

layer = Layer(@dy_nngetU, obj, varargin{:});

end