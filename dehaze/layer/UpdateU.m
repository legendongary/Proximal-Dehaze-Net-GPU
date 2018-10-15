function layer = UpdateU(obj, varargin)

layer = Layer(@dy_nnupdU, obj, varargin{:});

end

