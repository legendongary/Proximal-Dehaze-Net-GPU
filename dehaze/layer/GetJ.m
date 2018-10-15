function layer = GetJ(obj, varargin)

layer = Layer(@dy_nngetJ, obj, varargin{:});

end