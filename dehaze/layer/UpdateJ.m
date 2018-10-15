function layer = UpdateJ(obj, varargin)

layer = Layer(@dy_nnupdJ, obj, varargin{:});

end

