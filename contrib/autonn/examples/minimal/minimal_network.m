% MINIMAL_NETWORK
%   Demonstrates a simple logistic regression network.

run('../../setup_autonn.m') ;  % add AutoNN to the path


% load simple data
s = load('fisheriris.mat') ;
data_x = single(reshape(s.meas.', 1, 1, 4, [])) ;  % features in 3rd channel
[~, ~, data_y] = unique(s.species) ;  % convert strings to class labels



% define inputs
x = Input() ;
y = Input() ;

% predict using a convolutional layer. create and initialize parameters automatically
prediction = vl_nnconv(x, 'size', [1, 1, 4, 3]) ;

% define loss, and classification error
loss = vl_nnloss(prediction, y) ;
error = vl_nnloss(prediction, y, 'loss','classerror') ;

% use workspace variables' names as the layers' names, and compile net
Layer.workspaceNames() ;
net = Net(loss, error) ;



% simple SGD
learningRate = 1e-3 ;
outputs = zeros(1, 100) ;
rng(0) ;
params = [net.params.var] ;

for iter = 1:100,
  % draw minibatch
  idx = randperm(numel(data_y), 50) ;
  
  % evaluate network
  net.eval({'x', data_x(:,:,:,idx), 'y', data_y(idx)}) ;
  
  % update weights
  w = net.getValue(params) ;
  dw = net.getDer(params) ;
  
  for k = 1:numel(params),
    w{k} = w{k} - learningRate * dw{k} ;
  end
  
  net.setValue(params, w) ;
  
  % plot error
  outputs(iter) = net.getValue(error) / numel(idx) ;
end

figure(3) ;
plot(outputs) ;
xlabel('Iteration') ; ylabel('Error') ;

loss
net

