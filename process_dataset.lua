-- Preprocess the dataset: 
--   1) Build a dictionary of all characters in the dataset (including special characters such as '\n')
--   2) Transform the text file into a single integer vector (consisted of indices to the characters in the dictionary)
function process_file(input_file)
  local f = assert(io.open(input_file, 'r'))

  -- Read the whole file into memory (only useful for small files)
  local t = f:read("*all")
  f:close()

  local dictionary = {}
  local cur_index = 1

  local dataset = torch.IntTensor(#t)
  for i=1,#t do
    cur_char = string.sub(t,i,i) -- Obtain the ith character in the file, that is: t[i]
    if not dictionary[cur_char] then
      dictionary[cur_char] = cur_index
      cur_index = cur_index + 1
    end
    dataset[i] = dictionary[cur_char]
  end
  
  return dataset, dictionary
end

dataset, dictionary = process_file('data/input.txt')
torch.save('data/input.t7', dataset)
torch.save('data/dictionary.t7', dictionary)