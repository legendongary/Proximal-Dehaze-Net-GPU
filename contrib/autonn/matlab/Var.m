classdef Var < Param
%VAR Defines a network variable explicitly
%   Var('value', VALUE) defines an explicit variable in the network, with a
%   given value.
%
%   Most of the time this is not needed; variables are created implicitly
%   to hold the values of any Inputs, Params, and output values computed by
%   Layers (typically instantiated with overloaded function calls).
%   Constants such as strings or Matlab arrays are stored as arguments to
%   any function call that uses them (in a Layer).
%
%   An explicit variable (Var) acts like a constant argument to a Layer,
%   but may still be useful for the following reasons:
%   1. Its value can be moved to and from the GPU using NET.MOVE.
%   2. Its value can be set after network construction, with NET.SETVALUE.
%   3. Its derivative can be retrieved with NET.GETDER.
%
%   Internally, a Var is just a Param with trainMethod set to 'none'.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  methods
    function obj = Var(varargin)
      obj@Param('trainMethod','none', 'weightDecay',0, ...
        'learningRate',0, varargin{:})
    end
  end
end
