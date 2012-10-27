require 'torch'

local mytester = torch.Tester()

local precision = 1e-5
local expprecision = 1e-4
local dW = 5
local wsz = 5
local wfsz = 20
local nIndex = 100
local mbsz = 50

local lt3dtest = {}

function testIO(module,input, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval

   -- run module
   module:forward(input)
   local go = module.output:clone():copy(torch.rand(module.output:nElement()):mul(inrange):add(minval))
   module:zeroGradParameters()
   module:accGradParameters(input,go)

   local fo = module.output:clone()

   -- write module
   local f = torch.DiskFile('tmp.bin','w'):binary()
   f:writeObject(module)
   f:close()
   -- read module
   local m = torch.DiskFile('tmp.bin'):binary():readObject()
   m:forward(input)
   module:zeroGradParameters()
   m:accGradParameters(input,go)
   -- cleanup
   os.remove('tmp.bin')

   local fo2 = m.output:clone()

   local errf = fo - fo2

   return errf:abs():max()
end

function testJacobian(module, input)
   input:copy(torch.ceil(torch.rand(mbsz)*nIndex))
   local jac_fprop = nn.Jacobian.forward(module,input)
   local jac_bprop = backward(module,input)
   local error = jac_fprop-jac_bprop
   return error:abs():max()
end

function testJacobianParameters (module, input, param, dparam, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.ceil(torch.rand(mbsz)*nIndex))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local jac_bprop = nn.Jacobian.backward(module, input, param, dparam)
   local jac_fprop = nn.Jacobian.forward(module, input, param)
   local error = jac_fprop - jac_bprop
   return error:abs():max()
end

function testJacobianUpdateParameters (module, input, param, minval, maxval)
   minval = minval or -2
   maxval = maxval or 2
   local inrange = maxval - minval
   input:copy(torch.ceil(torch.rand(mbsz)*nIndex))
   param:copy(torch.rand(param:nElement()):mul(inrange):add(minval))
   local params_bprop = nn.Jacobian.backwardUpdate(module, input, param)
   local params_fprop = nn.Jacobian.forwardUpdate(module, input, param)

   local error = params_fprop - params_bprop
   return error:abs():max()
end

function lt3dtest.LookupTable3D()
   local input = torch.Tensor(mbsz):zero()
   local module = nn.LookupTable3D(nIndex,wfsz,wsz,dW)

--   local err = testJacobian(module,input)
--   mytester:assertlt(err,precision, 'error on state ')

   local err = testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err,precision, 'error on weight ')

   local err = testJacobianUpdateParameters(module, input, module.weight)
   mytester:assertlt(err,precision, 'error on weight [direct update] ')

   for t,err in pairs(nn.Jacobian.testAllUpdate(module, input, 'weight', 'gradWeight')) do
      mytester:assertlt(err, precision, string.format(
                         'error on weight [%s]', t))
   end

   -- IO
   local ferr = testIO(module,input)
   mytester:asserteq(ferr, 0, torch.typename(module) .. ' - i/o forward err ')
end

mytester:add(lt3dtest)


if not lt3d then
   require 'lt3d'
   mytester:run()
else
   mytester:run()
end