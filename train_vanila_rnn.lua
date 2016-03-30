local torch = require 'torch'
local nn = require 'nn'
require 'optim'


cmd = torch.CmdLine()
cmd:option('-rnn_size', 256, 'size of the hidden layer')
cmd:option('-train_length',20,'')
cmd:option('-batch_size',50,'')
cmd:option('-max_iterations',1000,'Number of training iterations (mini-batches)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- Load the dictionary (the set of characters considered by the model)
dictionary = torch.load('data/dictionary.t7')
dictionary_size = 0
for _ in pairs(dictionary) do 
    dictionary_size = dictionary_size + 1 
end

-- Load the dataset
dataset = torch.load('data/input.t7')


--Define the Recurrent neural network:
 -- Two inputs: x (the character at time t) and prev_h (the hidden state at time t-1)
 -- Two outputs: h (the hidden state at time t), y (the log probability of the next character)
local n = nn.Sequential()
  local inp = nn.ParallelTable()
  inp:add(nn.Linear(dictionary_size, opt.rnn_size))
  inp:add(nn.Linear(opt.rnn_size, opt.rnn_size))
n:add(inp)
n:add(nn.CAddTable())
n:add(nn.Tanh())
  local f = nn.ConcatTable()
  f:add(nn.Identity())
  local h2o = nn.Sequential()
  h2o:add(nn.Linear(opt.rnn_size, dictionary_size))
  h2o:add(nn.LogSoftMax())
  f:add(h2o)
n:add(f)

-- Training criterion: Negative Log Likelihood
criterion = nn.ClassNLLCriterion()

-- Auxiliary function to return the One-hot encoding of a list of inputs
--   The one-hot encoding contains zeros everywhere (the size of the dictionary), with a single 1 
function one_hot_encoding(x, n)
  local one_hot = torch.zeros(x:size(1), n)
  for i = 1, x:size(1) do
    one_hot[{i,x[i]}] = 1
  end
  return one_hot
end

-- Get all parameters in a single vector, and the gradients in another vector
params,grads = n:getParameters()

-- Create a closure that evaluates a single minibatch of data
function feval(x)
  if x ~= params then
      params:copy(x)
  end
  grads:zero()

  x = torch.Tensor(opt.batch_size, opt.train_length)
  y = torch.Tensor(opt.batch_size, opt.train_length)

  -- Randomly select some samples from the dataset. 
  --    X is the input (a sequence of "train_length" characters)
  --    Y is the desired output (i.e. the next character we want the network to predict).
  -- We model this by having selecting a sequence of characters X, and consider y[t] = x[t+1]
  for i = 1, opt.batch_size do
    start = torch.random(1, dataset:size(1) - opt.train_length-1)
    x[{i,{}}] = dataset[{{start, start-1 + opt.train_length}}]
    y[{i,{}}] = dataset[{{start+1, start + opt.train_length}}]
  end

  -- Initialize the initial state of the network with zeros
  previous_h = torch.zeros(opt.batch_size, opt.rnn_size)
  zero_h = previous_h:clone():zero()

  -- Variables that we need to store during forward propagation, to enable doing back-propagation
  inputs = {}
  rnn_state = {[0] = previous_h}
  predictions = {}
  loss = 0
  
  -- This is a tricky part. We want to use the same network weights for the forward/backward propagations
  -- in all time steps. However, the nn modules not only store the parameters, but also the intermediate output
  -- of all layers in the network (this is necessary for the backward pass). 
  -- So, we would like to share the parameters, but do not share the state of the output of each forward pass.
  -- In this file we clone the RNN "train_length" times, so the parameters are the same, but the intermediate states
  -- are preserved. We just need to remember to collect (sum) all gradients in the end.
  rnn = {}
  for t = 1, opt.train_length do
    rnn[t] = n:clone()
  end

  for t = 1, opt.train_length do
    inputs[t] = {one_hot_encoding(x[{{},t}], dictionary_size), rnn_state[t-1]}
    out_i = rnn[t]:forward(inputs[t])
    rnn_state[t] = out_i[1]
    predictions[t] = out_i[2]
    
    loss = loss + criterion:forward(predictions[t], y[{{},t}])
  end

  loss = loss / opt.train_length

  d_rnn_state = {[opt.train_length+1] = zero_h}


  --Backward pass: 
  for t = opt.train_length, 1, -1 do
    de_dc = criterion:backward(predictions[t], y[{{},t}])
    de_dinputs = rnn[t]:backward(inputs[t], {d_rnn_state[t+1], de_dc} )
    d_rnn_state[t] = de_dinputs[2]  -- discard the gradients to the input x
  end

  for t = 1, opt.train_length do
    pt, grad_params_t = rnn[t]:getParameters()
    grads:add(grad_params_t)
  end
  previous_h = rnn_state[#rnn_state]
  
  return loss, grads

end

local optim_state = {learningRate = 1e-3}
for iter = 1,opt.max_iterations do
  
    local timer = torch.Timer()
  _, loss = optim.rmsprop(feval, params, optim_state)
  local time = timer:time().real
  print('Iteration ', iter, '; loss ', loss[1], time)
end

torch.save('models/model.t7', n)
  --Next steps
  -- cleanup + blog post
  