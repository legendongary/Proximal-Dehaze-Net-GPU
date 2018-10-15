function layer = Soft(obj, varargin)

layer = Layer(@dy_nnsoft, obj, varargin{:});

end

