function [net, info] = cnn_mnist_autonn(varargin)
% CNN_MNIST_AUTONN  Demonstrates MatConNet on MNIST.

try run('../../setup_autonn.m') ; catch; end  % add AutoNN to the path
addpath([vl_rootnn '/examples/mnist']) ;

% opts.modelType = 'mlp' ;
opts.modelType = 'lenet' ;
opts.networkType = 'autonn' ;
opts.batchNormalization = false ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
opts.expDir = fullfile(vl_rootnn, 'data', ['mnist-baseline' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data', 'mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');

opts.train.batchSize = 100 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
if opts.batchNormalization
  opts.train.learningRate = 0.01 ;
else
  opts.train.learningRate = 0.001 ;
end
opts.train.expDir = opts.expDir ;
opts.train.numSubBatches = 1 ;
opts.train.plotStatistics = true ;
opts.train.plotDiagnostics = false ;
opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getMnistImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end


rng('default') ;
rng(0) ;

switch opts.networkType
case 'autonn'
  bn = opts.batchNormalization ;

  images = Input('gpu', true) ;
  labels = Input() ;

  switch opts.modelType
  case 'mlp'  % multi-layer perceptron
    x = vl_nnconv(images, 'size', [28, 28, 1, 100]) ;
    x = vl_nnrelu(x) ;
    x = vl_nnconv(x, 'size', [1, 1, 100, 100]) ;
    x = vl_nnrelu(x) ;
    x = vl_nndropout(x, 'rate', 0.5) ;
    x = vl_nnconv(x, 'size', [1, 1, 100, 10]) ;
    
  case 'lenet'  % LeNet
    x = vl_nnconv(images, 'size', [5, 5, 1, 20], 'weightScale', 0.01) ;
    if bn, x = vl_nnbnorm(x) ; end
    x = vl_nnpool(x, 2, 'stride', 2) ;
    x = vl_nnconv(x, 'size', [5, 5, 20, 50], 'weightScale', 0.01) ;
    if bn, x = vl_nnbnorm(x) ; end
    x = vl_nnpool(x, 2, 'stride', 2) ;
    x = vl_nnconv(x, 'size', [4, 4, 50, 500], 'weightScale', 0.01) ;
    if bn, x = vl_nnbnorm(x) ; end
    x = vl_nnrelu(x) ;
    x = vl_nnconv(x, 'size', [1, 1, 500, 10], 'weightScale', 0.01) ;
  end
  
  
%   % diagnose outputs of conv layers
%   Layer.setDiagnostics(x.find(@vl_nnconv), true) ;

  % diagnose all Params associated with conv layers (1 depth up from them)
  convs = x.find(@vl_nnconv) ;
  convParams = cellfun(@(x) x.find('Param', 'depth', 1), convs, 'Uniform', false) ;
  Layer.setDiagnostics(convParams, true) ;
  
  
  objective = vl_nnloss(x, labels, 'loss', 'softmaxlog') ;
  error = vl_nnloss(x, labels, 'loss', 'classerror') ;

  Layer.workspaceNames() ;  % assign layer names based on workspace variables (e.g. 'images', 'objective')
  net = Net(objective, error) ;  % compile network
  
  
case 'dagnn'
  % test conversion from a DagNN
  assert(strcmp(opts.modelType, 'lenet')) ;

  % create DagNN, and rename its inputs
  net = cnn_mnist_init('batchNormalization', opts.batchNormalization, ...
                       'networkType', 'dagnn') ;

  net.renameVar('input', 'images');
  net.renameVar('label', 'labels');
  
  % convert it
  layers = Layer.fromDagNN(net) ;
  
  images = layers{1}.find('images', 1) ;  % find images input
  images.gpu = true ;  % automatically move images to the GPU if needed
  
  net = Net(layers{:}) ;  % compile network

otherwise
  error('Unsupported network type ''%s''.', opts.networkType) ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net, info] = cnn_train_autonn(net, imdb, @getBatch, ...
  opts.train, 'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
inputs = {'images', images, 'labels', labels} ;

% --------------------------------------------------------------------
function imdb = getMnistImdb(opts)
% --------------------------------------------------------------------
% prepare the imdb structure, returns image data with mean image subtracted
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8') ;
fclose(f) ;
x1=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8') ;
fclose(f) ;
x2=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8') ;
fclose(f) ;
y1=double(y1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8') ;
fclose(f) ;
y2=double(y2(9:end)')+1 ;

set = [ones(1,numel(y1)) 3*ones(1,numel(y2))] ;
data = single(reshape(cat(3, x1, x2),28,28,1,[])) ;
dataMean = mean(data(:,:,:,set == 1), 4) ;
data = bsxfun(@minus, data, dataMean) ;

imdb.images.data = data ;
imdb.images.data_mean = dataMean ;
imdb.images.labels = cat(2, y1, y2) ;
imdb.images.set = set ;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = arrayfun(@(x)sprintf('%d',x),0:9,'uniformoutput',false) ;

