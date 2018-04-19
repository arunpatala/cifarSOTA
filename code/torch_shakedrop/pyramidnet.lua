local nn = require 'nn'
require 'cunn'
require 'models/ShakeDrop'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = 110
   local deathRateScale = 0.5
   local alpha = 270
   local iChannels

   -- The shortcut layer is either identity or 1x1 convolution
   local function shortcut(nInputPlane, nOutputPlane, stride)
      -- Strided, zero-padded identity shortcut
      local short = nn.Sequential()
      if stride == 2 then
         short:add(nn.SpatialAveragePooling(2, 2, 2, 2))
      end
      if nInputPlane ~= nOutputPlane then
         short:add(nn.Padding(1, (nOutputPlane - nInputPlane), 3))
      else
	       short:add(nn.Identity())
      end
      return short
   end

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, stride, deathRate)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      s:add(SBatchNorm(nInputPlane))
      s:add(Convolution(nInputPlane,n,3,3,stride,stride,1,1))
      s:add(SBatchNorm(n))
      s:add(ReLU(true))
      s:add(Convolution(n,n,3,3,1,1,1,1))
      s:add(SBatchNorm(n))
      s:add(nn.ShakeDrop(deathRate))
      return nn.Sequential()
         :add(nn.ConcatTable()
            :add(s)
            :add(shortcut(nInputPlane, n, stride)))
         :add(nn.CAddTable(true))
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, baseFeatures, addFeatures, blockNo, count, stride)
      local features = baseFeatures+(blockNo-1)*count*addFeatures
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features+i*addFeatures, i == 1 and stride or 1,
                     deathRateScale * (i+(blockNo-1)*count)/(count*3)
               ))
      end
      return s
   end

   local model = nn.Sequential()
   -- Model type specifies number of layers for CIFAR-10 model
   assert((depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110, 1202')
   local n = (depth - 2) / 6
   iChannels = 16
   addChannel = alpha/(3*n)
   print(' | PyramidNet(ShakeDrop)-' .. depth ..  ' alpha=' .. alpha .. ' ' .. opt.dataset)
 
   -- The ResNet CIFAR-10 model
   model:add(Convolution(3,16,3,3,1,1,1,1))
   model:add(SBatchNorm(16))
   -- model:add(ReLU(true))
   model:add(layer(basicblock, 16, addChannel, 1, n))
   model:add(layer(basicblock, 16, addChannel, 2, n, 2))
   model:add(layer(basicblock, 16, addChannel, 3, n, 2))
   model:add(nn.Copy(nil, nil, true))
   model:add(SBatchNorm(iChannels))
   model:add(ReLU(true))
   model:add(Avg(8, 8, 1, 1))
   model:add(nn.View(iChannels):setNumInputDims(3))
   if opt.dataset == 'cifar10' then
      model:add(nn.Linear(iChannels, 10))
   elseif opt.dataset == 'cifar100' then
     model:add(nn.Linear(iChannels, 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel