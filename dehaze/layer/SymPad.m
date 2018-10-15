function layer = SymPad(obj, varargin)

layer = Layer(@dy_nnpad, obj, varargin{:});

end

