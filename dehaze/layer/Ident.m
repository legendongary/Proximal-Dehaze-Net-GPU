function layer = Ident(obj, varargin)

layer = Layer(@dy_nnident, obj, varargin{:});

end