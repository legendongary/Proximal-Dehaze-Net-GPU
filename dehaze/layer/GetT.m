function layer = GetT(obj, varargin)

layer = Layer(@dy_nngetT, obj, varargin{:});

end