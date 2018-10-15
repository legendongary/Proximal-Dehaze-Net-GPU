function y = gpuArray_wrapper(x, gpuMode, dzdy)
%GPUARRAY_WRAPPER AutoNN wrapper for gpuArray
%   Wrapper for gpuArray, disabled automatically when in CPU mode.

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  if nargin < 3  % forward mode
    if gpuMode
      y = gpuArray(x) ;
    else
      y = x ;
    end
  else  % backward mode
    if ~isa(x, 'gpuArray')
      y = gather(dzdy) ;  % convert derivative to same type as input
    else
      y = dzdy ;  % keep same type (was already a gpuArray)
    end
  end
end
