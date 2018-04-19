require 'nn'
require 'cudnn'
require 'cunn'

local ShakeDrop, parent = torch.class('nn.ShakeDrop', 'nn.Module')

function ShakeDrop:__init(deathRate)
    parent.__init(self)
    self.gradInput = torch.Tensor()
    self.gate = true
    self.train = true
    self.deathRate = deathRate or 0.5
end

function ShakeDrop:updateOutput(input)
    -- local skip_forward = self.skip:forward(input)
    self.gate = (torch.rand(1)[1] > self.deathRate)
    self.output:resizeAs(input)
    if self.train then
      if not self.gate then -- only compute convolutional output when gate is open
         self.output:uniform():mul(2):add(-1):cmul(input)
      else
         self.output:copy(input)
      end
    else
      self.output:copy(input):mul(1-(self.deathRate*1.0))
    end
    return self.output
end

function ShakeDrop:updateGradInput(input, gradOutput)
   if self.gradInput then
    self.gradInput:resizeAs(gradOutput)
    if not self.gate then
      self.gradInput:uniform():cmul(gradOutput)
    else
      self.gradInput:copy(gradOutput)
    end
    return self.gradInput
   end
end

function ShakeDrop:accGradParameters(input, gradOutput, scale)
   -- scale = scale or 1
   -- if self.gate then
   --    self.net:accGradParameters(input, gradOutput, scale)
   -- end
end