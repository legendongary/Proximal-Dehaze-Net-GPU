function layer = Upsample(obj, varargin)

layer = Layer(@dy_nnupsample, obj, varargin{:});

end

