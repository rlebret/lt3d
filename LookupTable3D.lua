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
   input.nn.LookupTable3D_accGradParameters(self, input, gradOutput, scale)
end

function LookupTable3D:accUpdateGradParameters(input, gradOutput, lr)
   input.nn.LookupTable3D_accUpdateGradParameters(self, input, gradOutput, lr)
end

function LookupTable3D:updateParameters(learningRate)
   for k,_ in pairs(self.inputs) do
      self.weight:select(1, k):add(-learningRate, self.gradWeight:select(1, k))
   end
end

-- we do not need to accumulate parameters when sharing
LookupTable3D.sharedAccUpdateGradParameters = LookupTable3D.accUpdateGradParameters
