local torch = require 'torch'
local nn = require 'nn'
require 'optim'

temperature=0.8

dictionary = torch.load('data/dictionary.t7')
dataset = torch.load('data/input.t7')
model = torch.load('model.t7')
dictionary_size = 0

inverse_dictionary = {}
for k,v in pairs(dictionary) do 
  inverse_dictionary[v]=k
    dictionary_size = dictionary_size + 1 
end

function one_hot_encoding(x, n)
  local one_hot = torch.zeros(n)
  one_hot[x] = 1
  return one_hot
end


starttext = 'ROMEO:\n'

startvector = torch.zeros(#starttext)
for i = 1, #starttext do
  startvector[i] = dictionary[starttext:sub(i,i)]
end

a=1

prev_h = torch.zeros(256)

output = ""

for i = 1, #starttext do
  new_state = model:forward({one_hot_encoding(startvector[i], dictionary_size), prev_h})
  prev_h = new_state[1]
  logprobs = new_state[2]
  output = output .. inverse_dictionary[startvector[i]]
end

function sampleFromLogProbs(logprobs)
  logprobs:div(temperature)
  probs = torch.exp(logprobs)
  probs:div(probs:sum())
  return torch.multinomial(probs, 1)
end

for i=1,2000 do
  new_x = sampleFromLogProbs(logprobs)[1]
  new_state = model:forward({one_hot_encoding(new_x, dictionary_size), prev_h})
  prev_h = new_state[1]
  logprobs = new_state[2]
  output = output .. inverse_dictionary[new_x]
end

print(output)