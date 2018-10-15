function layer = NonZero(obj, varargin)

layer = Layer(@dy_nnzero, obj, varargin{:});

end