local LookupTable3D, parent = torch.class('nn.LookupTable3D', 'nn.Module')

LookupTable3D.__version = 1

--local didit

function LookupTable3D:__init(nIndex, size, inputFrameSize, dW)
   parent.__init(self)

   dW = dW or 1
   self.size = torch.LongStorage(2)
   self.size[1] = nIndex
   self.size[2] = size
   self.inputFrameSize = inputFrameSize
   self.dW = dW
   self.weight = torch.Tensor(self.size)
   self.gradWeight = torch.Tensor(self.size):zero()
   self.inputs = {}

   self:reset()
end

function LookupTable3D:reset(stdv)
   stdv = stdv or 1
   self.weight:apply(function()
                        return torch.normal(0, stdv)
                     end)
end

function LookupTable3D:updateOutput(input)

-- if not didit then 
--   didit=true
   --print("updateOutput")
--[[  
   local nIndex
   if input:size(1) <= self.inputFrameSize then
      nIndex=1
   else
      nIndex=input:size(1)-self.inputFrameSize
   end
   self.size[1] = nIndex
   self.output:resize(nIndex,self.size[2]*self.inputFrameSize)

   local i=1
   for k=1,nIndex do
      for j=1,self.inputFrameSize do
         self.output[{k,{(j-1)*self.size[2]+1,j*self.size[2]}}] = self.weight:select(1, input[(i-1)+j])
      end
      i = i+self.dW
   end
-- else
   --print("no updateOutput")
-- end
  return self.output
--]]
   input.nn.LookupTable3D_updateOutput(self, input)
   return self.output

end

function LookupTable3D:zeroGradParameters()
   for k,_ in pairs(self.inputs) do
      self.gradWeight:select(1, k):zero()
   end
   self.inputs = {}
end

function LookupTable3D:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
--if not didit then 
--   didit=true

--[[
   local i=1
   for k=1,self.size[1] do
      local grad = gradOutput:select(1,k):resize(self.inputFrameSize,self.size[2])
      for j=1,self.inputFrameSize do
        local l = input[(i-1)+j]
        self.inputs[l] = true
        self.gradWeight:select(1, l):add(scale, grad:select(1, j))
      end
      i = i+self.dW
   end
--end
--]]
   input.nn.LookupTable3D_accGradParameters(self, input, gradOutput, scale)
end

function LookupTable3D:updateParameters(learningRate)
   for k,_ in pairs(self.inputs) do
      self.weight:select(1, k):add(-learningRate, self.gradWeight:select(1, k))
   end
end

-- we do not need to accumulate parameters when sharing
LookupTable3D.sharedAccUpdateGradParameters = LookupTable3D.accUpdateGradParameters
