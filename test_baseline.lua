require 'torch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'
require 'image'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Sensorimotor Object Recognition')
cmd:text()
cmd:text('Options:')
------------ General options --------------------
cmd:option('-appearanceSamplesPath', '', 'SOR3D colorized object depthmaps')    
cmd:option('-manualSeed',         2, 'Manually set RNG seed')
cmd:option('-GPU',                1, 'Default preflossed GPU')
cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
cmd:option('-backend',      'cudnn', 'Default backend')
------------- Data options ------------------------
cmd:option('-imageSize',         300,    'Smallest side of the resized image')
cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
cmd:option('-cnnSize',           4096,    'size of FC6')
cmd:option('-nClasses',             14,     'number of classes')
------------- Test options --------------------
cmd:option('-batchSize',          32,    'mini-batch size (1 = pure stochastic)')
---------- Model options ----------------------------------
cmd:option('-netType',     'vgg_16_pretrained', 'alexnet_pretrained(pretrained caffe model) | vgg_16_pretrain(pretrained caffe model)')
cmd:option('-loadNet', '', 'load networks from directory')

opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU)
paths.dofile('util.lua')

if opt.backend == 'cudnn' then
  require 'cudnn'
end

--load model
local model = torch.load(opt.loadNet .. 'baseline.t7')
model:cuda()

--Mean & Std of SOR3D processed data
local meanstdApp = {mean = {0.016,0.01, 0}, std = {0.0423, 0.0284, 0}}

--GPU inputs (preallocate)
local inputsCudaAppearance = torch.CudaTensor()
local labelsCuda = torch.CudaTensor()

--Set timers
local timer = torch.Timer()
local dataTimer = torch.Timer()

function scaleImg(inputObject)
  local diff = opt.imageSize - opt.cropSize
  local iW = inputObject:size(3)
  local iH = inputObject:size(2)
  if iW > opt.imageSize or iH > opt.imageSize then
    inputObject = image.scale(inputObject, opt.imageSize, opt.imageSize, 'bilinear')
  end

  local outObject = image.scale(inputObject, opt.cropSize, opt.cropSize, 'bilinear')

  assert(outObject:size(3) == opt.cropSize)
  assert(outObject:size(2) == opt.cropSize)

  for c=1,3 do -- channels
    if meanApp then outObject[{{c},{},{}}]:add(-meanstdApp.mean[c]):div(meanstdApp.std[c]) end
  end

  return outObject
end

function LoadTestData()
  path1_test = opt.appearanceSamplesPath..'/test/'
  classes_test = paths.dir(path1_test)
  table.sort(classes_test,function(a,b)return a<b end)
  table.remove(classes_test,1) table.remove(classes_test,1)

  appearance_test={}
  numOfSubjectsInClass_test=torch.FloatTensor(#classes_test):fill(0)

  for i = 1, #classes_test do
    appearance_test[i] = paths.dir(path1_test .. classes_test[i])
    table.sort(appearance_test[i],function(a,b)return a<b end)
    table.remove(appearance_test[i],1) table.remove(appearance_test[i],1)
    numOfSubjectsInClass_test[i] = #appearance_test[i]
  end

  totalSamples_test = torch.sum(numOfSubjectsInClass_test)
  print('Total Test Samples: '..totalSamples_test)
  allSamplesIndex_test = torch.FloatTensor(totalSamples_test)

  cnt = 0
  appearanceSum_test = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
  for i = 1, #classes do
    for j = 1, numOfSubjectsInClass_test[i] do
      cnt = cnt+1
      allSamplesIndex_test[cnt] = i
    end
  end
end

function Test()
  print('==> doing evaluation on test data:')

  cutorch.synchronize()
  temporalAppearance = appearance_test

  model:evaluate()

  batchesInTotalSamples = torch.floor(totalSamples_test/opt.batchSize)
  remainingSamples = totalSamples_test - (batchesInTotalSamples*opt.batchSize)
  tm = torch.Timer()
  samples_cnt=0

  --Create batch
  appearanceBatch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
  labels = torch.Tensor(opt.batchSize):fill(0)
  for i=1,batchesInTotalSamples do
    for j=1, opt.batchSize do
      if samples_cnt>=totalSamples_test then break end
      samples_cnt=samples_cnt+1
      randomClass = allSamplesIndex_test[samples_cnt]
      numOfSubjectsInRandomClass = #temporalAppearance[randomClass]
      randomSubject = torch.randperm(numOfSubjectsInRandomClass)
      appearanceSample = path1_test .. classes_test[randomClass] .. '/' .. temporalAppearance[randomClass][randomSubject[1]] .. '/' .. '003.png'

      currAppearanceSample = image.load(appearanceSample)
      app = scaleImg(currAppearanceSample)
      table.remove(temporalAppearance[randomClass],randomSubject[1])

      appearanceBatch[j]:copy(app)
      labels[j] = randomClass
    end
    Forward(appearanceBatch, labels, false)
  end
  for j=1, opt.batchSize do
    if j <= remainingSamples then
      samples_cnt=samples_cnt+1
      randomClass = allSamplesIndex_test[samples_cnt]
      numOfSubjectsInRandomClass = #temporalAppearance[randomClass]
      randomSubject = torch.randperm(numOfSubjectsInRandomClass)
      appearanceSample = path1_test .. classes_test[randomClass] .. '/' .. temporalAppearance[randomClass][randomSubject[1]] .. '/' .. '003.png'

      currAppearanceSample = image.load(appearanceSample)
      app = scaleImg(currAppearanceSample)
      table.remove(temporalAppearance[randomClass],randomSubject[1])

      appearanceBatch[j]:copy(app)
      labels[j] = randomClass
    else
      appearanceBatch[j]:fill(0)
      labels[j] = 0
    end
  end
  Forward(appearanceBatch, labels)
  print('Samples evaluated: ' .. samples_cnt)
end

function Forward(inputsCPUappearance, labelsCPU)
  cutorch.synchronize()
  collectgarbage()
  local dataLoadingTime = dataTimer:time().real
  timer:reset()

  inputsCudaAppearance:resize(inputsCPUappearance:size()):copy(inputsCPUappearance)
  labelsCuda:resize(labelsCPU:size()):copy(labelsCPU)

  local N = opt.batchSize;

  out = model:forward(inputsCudaAppearance)

  local _,prediction_sorted = out:float():sort(2, true) -- descending
  for i=1,N do
    if labelsCPU[i] > 0 then
      conf:add(prediction_sorted[i][1], labelsCPU[i])
    end
  end

  dataTimer:reset()
end
print('\n')
path = opt.appearanceSamplesPath..'/test/'
classes = paths.dir(path)
table.sort(classes,function(a,b)return a<b end)
table.remove(classes,1) table.remove(classes,1)
for i = 1, #classes do
   print('Class' .. i .. ': ' .. classes[i])
end
print('\n')
  
LoadTestData()
conf = optim.ConfusionMatrix(#classes,{1,2,3,4,5,6,7,8,9,10,11,12,13,14})
conf:zero()
Test()

print(conf)
filename = paths.concat(opt.loadNet, 'confusionMatrix.csv')
print('=> saving confusion matrix to: ' .. opt.loadNet)
file = io.open(filename, 'w')
file:write(tostring(conf))
file:close()
image.display(conf:render())

