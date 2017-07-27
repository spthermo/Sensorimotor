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
cmd:option('-cache', './Results/SML/', 'subdirectory in which to save/log experiments')
cmd:option('-affordanceSamplesPath', '', 'SOR3D 3D flow magnitude')
cmd:option('-appearanceSamplesPath', '', 'SOR3D colorized object depthmaps')    
cmd:option('-GPU',                1, 'Default preflossed GPU')
cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
cmd:option('-backend',      'cudnn', 'Default backend')
------------- Data options ------------------------
cmd:option('-imageSize',         300,    'Smallest side of the resized image')
cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
cmd:option('-cnnSize',           4096,    'Size of FC6')
cmd:option('-augment',             0,     'Use data augmentation')
cmd:option('-nClasses',             14,     'number of classes')
------------- Training options --------------------
cmd:option('-nEpochs',             30,    'Number of total epochs to run')
cmd:option('-decayEpoch',         16,     'Epoch number to decay LR')
cmd:option('-batchSize',          32,    'mini-batch size (1 = pure stochastic)')
cmd:option('-verbose',             0, 'print per-batch training and test info')
---------- Optimization options ----------------------
cmd:option('-learningRate',    0.005, 'learning rate')
cmd:option('-learningRateDecay',    0.2, 'learning rate decay')
cmd:option('-momentum',         0.9, 'momentum')
cmd:option('-weightDecay',     5e-4, 'weight decay')
---------- Model options ----------------------------------
cmd:option('-netType',     'vgg_16_pretrained', 'alexnet_pretrained(pretrained caffe model) | vgg_16_pretrain(pretrained caffe model)')

opt = cmd:parse(arg or {})
opt.save = opt.cache
opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
print('Saving data to: ' .. opt.save)
os.execute('mkdir -p ' .. opt.save)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU)
paths.dofile('util.lua')

-- load VGG16 pretrained
paths.dofile('models/' .. opt.netType .. '.lua')
local vgg16 = createModel(opt.nGPU)
local model1 = vgg16:clone() --affordance stream
local model2 = vgg16:clone() --appearance stream


for i=40,31,-1 do
  model1:remove(i)
  model2:remove(i)
end

--normalize activations before the first fusion
model1:add(nn.SpatialBatchNormalization(512,1e-3))
model2:add(nn.SpatialBatchNormalization(512,1e-3))

local model4 = nn.Sequential() --appearance stream after the inclusion of the affordance features
  :add(nn.SpatialConvolution(1024, 512, 1, 1, 1, 1, 0, 0)) --1x1 conv used after feature maps fusion
  :add(nn.ReLU(true))
	:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
	:add(nn.View(-1):setNumInputDims(3))
  :add(nn.Dropout(0.5))
	:add(nn.Linear(25088, opt.cnnSize))
  :add(nn.BatchNormalization(opt.cnnSize)) --normalize activations before the second fusion
	:add(nn.ReLU(true))

local model3 = nn.Sequential() --affordance stream part 2
	:add(nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil())
	:add(nn.View(-1):setNumInputDims(3))
  :add(nn.Dropout(0.5))
	:add(nn.Linear(25088, opt.cnnSize))
  :add(nn.BatchNormalization(opt.cnnSize))--normalize activations before the second fusion
	:add(nn.ReLU(true))
local fusion = nn.Sequential() --FC network after the second fusion
	:add(nn.Linear(2*opt.cnnSize, opt.cnnSize))
	:add(nn.ReLU(true))
	:add(nn.Dropout(0.5))
	:add(nn.Linear(opt.cnnSize, opt.nClasses))
	:add(nn.LogSoftMax())

model4.modules[6].weight = vgg16.modules[33].weight
model3.modules[4].weight = vgg16.modules[33].weight
vgg16 = nil

model1:cuda()
model2:cuda()
model3:cuda()
model4:cuda()
fusion:cuda()

print(model2)
print(model4)
print(model1)
print(model3)
print(fusion)

if opt.backend == 'cudnn' then
	require 'cudnn'
	cudnn.convert(model1, cudnn)
	cudnn.convert(model2, cudnn)
	cudnn.convert(model3, cudnn)
	cudnn.convert(model4, cudnn)
	cudnn.convert(fusion, cudnn)
else
	lossor'Unsupported backend'
end

criterion = nn.ClassNLLCriterion()
criterion:cuda()

--Mean & Std of SOR3D processed data
local meanstdAff = {mean = {0.0207,0.0038, 0.0022}, std = {0.0543, 0.0236, 0.0165}}
local meanstdApp = {mean = {0.016,0.01, 0}, std = {0.0423, 0.0284, 0}}

--GPU inputs (preallocate)
local inputsCudaAffordance = torch.CudaTensor()
local inputsCudaAppearance = torch.CudaTensor()
local labelsCuda = torch.CudaTensor()

--Set timers
local timer = torch.Timer()
local dataTimer = torch.Timer()

--Get model parameters
local WeightsFusion, GradientsFusion = fusion:getParameters()
local WeightsModel1, GradientsModel1 = model1:getParameters()
local WeightsModel2, GradientsModel2 = model2:getParameters()
local WeightsModel3, GradientsModel3 = model3:getParameters()
local WeightsModel4, GradientsModel4 = model4:getParameters()
local totalParams = WeightsModel1:size(1) + WeightsModel2:size(1) + WeightsModel3:size(1) + WeightsModel4:size(1) + WeightsFusion:size(1)


local optimStateFusion = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}
local optimStateModel4 = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}
local optimStateModel3 = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}
local optimStateModel2 = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}
local optimStateModel1 = {
  learningRate = opt.learningRate,
  momentum = opt.momentum,
  dampening = 0.0,
  weightDecay = opt.weightDecay
}

--Create loggers.
lossLogger = optim.Logger(paths.concat(opt.save, 'loss.log'))
acclogger = optim.Logger(paths.concat(opt.save, 'accuracies.log'))
batchNumber = 0
top1_epoch = 0
loss_epoch = 0

--data augmentation
local function hflip(x1,x2)
  local y1, y2
  if torch.random(0,1) > 0.5 then
    y1 = image.hflip(x1) 
    y2 = image.hflip(x2)
    return y1, y2
  else
    return x1, x2
  end
end

local function randomCrop(inputObject, inputHand)
  local diff = opt.imageSize - opt.cropSize
  local iW = inputObject:size(3)
  local iH = inputObject:size(2)
  if iW > opt.imageSize or iH > opt.imageSize then
    inputObject = image.scale(inputObject, opt.imageSize, opt.imageSize, 'bilinear')
  end
  if inputHand:size(3) > opt.imageSize or inputHand:size(2) > opt.imageSize then
    inputHand = image.scale(inputHand, opt.imageSize, opt.imageSize, 'bilinear')
  end

  local w1 = math.ceil(torch.uniform(1e-2, diff))
  local h1 = math.ceil(torch.uniform(1e-2, diff))
  local outObject = image.crop(inputObject, w1, h1, w1 + opt.cropSize, h1 + opt.cropSize)
  local outHand = image.crop(inputHand, w1, h1, w1 + opt.cropSize, h1 + opt.cropSize)

  assert(outObject:size(3) == opt.cropSize)
  assert(outObject:size(2) == opt.cropSize)
  assert(outHand:size(3) == opt.cropSize)
  assert(outHand:size(2) == opt.cropSize)

  for c=1,3 do -- channels
    if meanApp then outObject[{{c},{},{}}]:add(-meanstdApp.mean[c]):div(meanstdApp.std[c]) end
    if meanAff then outHand[{{c},{},{}}]:add(-meanstdAff.mean[c]):div(meanstdAff.std[c]) end
  end

  return outObject, outHand
end

local function scaleImg(inputObject, inputHand)
  local diff = opt.imageSize - opt.cropSize
  local iW = inputObject:size(3)
  local iH = inputObject:size(2)
  if iW > opt.imageSize or iH > opt.imageSize then
    inputObject = image.scale(inputObject, opt.imageSize, opt.imageSize, 'bilinear')
  end
  if inputHand:size(3) > opt.imageSize or inputHand:size(2) > opt.imageSize then
    inputHand = image.scale(inputHand, opt.imageSize, opt.imageSize, 'bilinear')
  end

  local outObject = image.scale(inputObject, opt.cropSize, opt.cropSize, 'bilinear')
  local outHand = image.scale(inputHand, opt.cropSize, opt.cropSize, 'bilinear')

  assert(outObject:size(3) == opt.cropSize)
  assert(outObject:size(2) == opt.cropSize)
  assert(outHand:size(3) == opt.cropSize)
  assert(outHand:size(2) == opt.cropSize)

  for c=1,3 do -- channels
    if meanApp then outObject[{{c},{},{}}]:add(-meanstdApp.mean[c]):div(meanstdApp.std[c]) end
    if meanAff then outHand[{{c},{},{}}]:add(-meanstdAff.mean[c]):div(meanstdAff.std[c]) end
  end

  return outObject, outHand
end

function LoadTrainData()
  path1 = opt.affordanceSamplesPath..'/train/'
  path2 = opt.appearanceSamplesPath..'/train/'
  classes = paths.dir(path1)
  table.sort(classes,function(a,b)return a<b end)
  table.remove(classes,1) table.remove(classes,1)

	affordance={}
	appearance={}
	numOfSubjectsInClass=torch.FloatTensor(#classes):fill(0)

	for i = 1, #classes do
		affordance[i] = paths.dir(path1 .. classes[i])
		appearance[i] = paths.dir(path2 .. classes[i])
		table.sort(affordance[i],function(a,b)return a<b end)
		table.sort(appearance[i],function(a,b)return a<b end)
		table.remove(affordance[i],1) table.remove(affordance[i],1)
		table.remove(appearance[i],1) table.remove(appearance[i],1)
		numOfSubjectsInClass[i] = #affordance[i]
	end

	totalSamples = torch.sum(numOfSubjectsInClass)
	print('Total Train Samples: '..totalSamples)
	allSamplesIndex = torch.FloatTensor(totalSamples)

	cnt = 0
  affordanceSum = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
  appearanceSum = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
  for i = 1, #classes do
    for j = 1, numOfSubjectsInClass[i] do
      cnt = cnt+1
      allSamplesIndex[cnt] = i
    end
  end
end

function LoadTestData()
  path1_val = opt.affordanceSamplesPath..'/val/'
  path2_val = opt.appearanceSamplesPath..'/val/'
  classes_val = paths.dir(path1_val)
  table.sort(classes_val,function(a,b)return a<b end)
  table.remove(classes_val,1) table.remove(classes_val,1)

  affordance_val={}
  appearance_val={}
  numOfSubjectsInClass_val=torch.FloatTensor(#classes_val):fill(0)

  for i = 1, #classes_val do
    affordance_val[i] = paths.dir(path1_val .. classes_val[i])
    appearance_val[i] = paths.dir(path2_val .. classes_val[i])
    table.sort(affordance_val[i],function(a,b)return a<b end)
    table.sort(appearance_val[i],function(a,b)return a<b end)
    table.remove(affordance_val[i],1) table.remove(affordance_val[i],1)
    table.remove(appearance_val[i],1) table.remove(appearance_val[i],1)
    numOfSubjectsInClass_val[i] = #affordance_val[i]
  end

  totalSamples_val = torch.sum(numOfSubjectsInClass_val)
  print('Total Validation Samples: '..totalSamples_val)
  allSamplesIndex_val = torch.FloatTensor(totalSamples_val)

  cnt = 0
  affordanceSum_val = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
  appearanceSum_val = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
  for i = 1, #classes do
    for j = 1, numOfSubjectsInClass_val[i] do
      cnt = cnt+1
      allSamplesIndex_val[cnt] = i
    end
  end
end

function Train()
	print('==> doing epoch on training data:')
	print("==> online epoch # " .. epoch)

  cutorch.synchronize()
  temporalAffordance = affordance
  temporalAppearance = appearance

	model1:training()
	model2:training()
	model3:training()
	model4:training()
	fusion:training()
	
  tm = torch.Timer()
	top1_epoch = 0
	loss_epoch = 0
	batchNumber = 0
	rand_cnts=torch.randperm(totalSamples)
	samples_cnt=0
   
	--Create batch
	affordanceBatch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
	appearanceBatch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
	labels = torch.Tensor(opt.batchSize):fill(0)
	for i=1,epochSize do
    for j=1,opt.batchSize do
      if samples_cnt>=totalSamples then break end
    	samples_cnt=samples_cnt+1
    	randomClass = allSamplesIndex[rand_cnts[samples_cnt]]
    	numOfSubjectsInRandomClass = #temporalAffordance[randomClass]
    	randomSubject = torch.randperm(numOfSubjectsInRandomClass)
    	affordanceSample = path1 .. classes[randomClass] .. '/' .. temporalAffordance[randomClass][randomSubject[1]] .. '/' .. '001.png'
    	appearanceSample = path2 .. classes[randomClass] .. '/' .. temporalAffordance[randomClass][randomSubject[1]] .. '/' .. '003.png'

    	currAffordanceSample = image.load(affordanceSample)
    	currAppearanceSample = image.load(appearanceSample)

      if opt.augment == 1 then
        appFlipped, affFlipped = hflip(currAppearanceSample, currAffordanceSample)
        app, aff = randomCrop(appFlipped, affFlipped)
      else
        app, aff = scaleImg(currAppearanceSample, currAffordanceSample)
    	end
      table.remove(temporalAffordance[randomClass],randomSubject[1])
    	table.remove(temporalAppearance[randomClass],randomSubject[1])
    
    	affordanceBatch[j]:copy(aff)
    	appearanceBatch[j]:copy(app)
    	labels[j] = randomClass
    end
    Forward(affordanceBatch, appearanceBatch, labels, true)
   end

	cutorch.synchronize()

	top1_epoch = top1_epoch * 100 / (opt.batchSize * epochSize )
	loss_epoch = loss_epoch / epochSize

	print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                     .. 'average loss (per batch): %.2f \t '
                     .. 'accuracy(%%):\t top-1 %.2f\t',
                     epoch, tm:time().real, loss_epoch, top1_epoch))
	print('\n')
	
	lossLogger:add{ ['Loss'] = loss_epoch*100 }
	lossLogger:style{['Loss'] = '-'}
	lossLogger:plot()

	return top1_epoch
end

function Test()
  print('==> doing epoch on validation data:')
  print("==> online epoch # " .. epoch)

  cutorch.synchronize()
  temporalAffordance = affordance_val
  temporalAppearance = appearance_val

  model1:evaluate()
  model2:evaluate()
  model3:evaluate()
  model4:evaluate()
  fusion:evaluate()

  tm = torch.Timer()
  top1_epoch = 0
  loss_epoch = 0
  batchNumber = 0
  rand_cnts=torch.randperm(totalSamples_val)
  samples_cnt=0

  --Create batch
  affordanceBatch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
  appearanceBatch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
  labels = torch.Tensor(opt.batchSize):fill(0)
  for i=1,epochTestSize do
    for j=1,opt.batchSize do
      if samples_cnt>=totalSamples_val then break end
      samples_cnt=samples_cnt+1
      randomClass = allSamplesIndex_val[rand_cnts[samples_cnt]]
      numOfSubjectsInRandomClass = #temporalAffordance[randomClass]
      randomSubject = torch.randperm(numOfSubjectsInRandomClass)
      affordanceSample = path1_val .. classes_val[randomClass] .. '/' .. temporalAffordance[randomClass][randomSubject[1]] .. '/' .. '001.png'
      appearanceSample = path2_val .. classes_val[randomClass] .. '/' .. temporalAffordance[randomClass][randomSubject[1]] .. '/' .. '003.png'

      currAffordanceSample = image.load(affordanceSample)
      currAppearanceSample = image.load(appearanceSample)
      app, aff = scaleImg(currAppearanceSample, currAffordanceSample)
      table.remove(temporalAffordance[randomClass],randomSubject[1])
      table.remove(temporalAppearance[randomClass],randomSubject[1])

      affordanceBatch[j]:copy(aff)
      appearanceBatch[j]:copy(app)
      labels[j] = randomClass
    end
    Forward(affordanceBatch, appearanceBatch, labels, false)
  end
  cutorch.synchronize()

  top1_epoch = top1_epoch * 100 / (opt.batchSize * epochTestSize )

  print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                   .. 'accuracy(%%):\t top-1 %.2f\t',
                   epoch, tm:time().real, top1_epoch))
  print('\n')

  return top1_epoch
end

function Forward(inputsCPUaffordance, inputsCPUappearance, labelsCPU, trainFlag)
  cutorch.synchronize()
  collectgarbage()
  local dataLoadingTime = dataTimer:time().real
  timer:reset()

  inputsCudaAffordance:resize(inputsCPUaffordance:size()):copy(inputsCPUaffordance)
  inputsCudaAppearance:resize(inputsCPUappearance:size()):copy(inputsCPUappearance)
  labelsCuda:resize(labelsCPU:size()):copy(labelsCPU)
  local loss, feats1, feats2
  local N = opt.batchSize
  
  if trainFlag then
  	model1:zeroGradParameters()
  	model2:zeroGradParameters()
  	model3:zeroGradParameters()
  	model4:zeroGradParameters()
  	fusion:zeroGradParameters()
  end

  feats1 = model1:forward(inputsCudaAffordance)
  feats2 = model2:forward(inputsCudaAppearance)
  featsConcat = torch.cat(feats2, feats1, 2) --concatenate in depth before 1x1 conv, first fusion
  affordanceFeatures = model3:forward(feats1:cuda())
  appearanceFeatures = model4:forward(featsConcat:cuda())
  feats = torch.cat(appearanceFeatures, affordanceFeatures, 2) --concatenate the 2 feature vectors, second fusion
  out = fusion:forward(feats:cuda())
  if trainFlag then
    loss = criterion:forward(out, labelsCuda)
  	gradOut = criterion:backward(out, labelsCuda)
  	gradIn = fusion:backward(feats:cuda(), gradOut:cuda())
  	tempGrad = gradIn:split(4096, 2) --split gradients
    grad4 = model4:backward(featsConcat:cuda(), tempGrad[1]:contiguous())
  	grad3 = model3:backward(feats1:cuda(), tempGrad[2]:contiguous())
  	tempGrad4 = grad4:split(512, 2) --split gradients
  	model1:backward(inputsCudaAffordance, grad3)
  	model2:backward(inputsCudaAppearance, tempGrad4[1]:contiguous())
  	
  	fevalFusion = function()
  		return loss, GradientsFusion
  	end
     fevalModel4 = function()
        return loss, GradientsModel4
     end
     fevalModel3 = function()
        return loss, GradientsModel3
     end
     fevalModel2 = function()
        return loss, GradientsModel2
     end
  	fevalModel1 = function()
    	   return loss, GradientsModel1
  	end
  	
  	optim.sgd(fevalFusion, WeightsFusion, optimStateFusion)
    optim.sgd(fevalModel4, WeightsModel4, optimStateModel4)
    optim.sgd(fevalModel3, WeightsModel3, optimStateModel3)
    optim.sgd(fevalModel2, WeightsModel2, optimStateModel2)
  	optim.sgd(fevalModel1, WeightsModel1, optimStateModel1)
  end

  batchNumber = batchNumber + 1
  if trainFlag then loss_epoch = loss_epoch + loss end
  local top1 = 0
  local _,prediction_sorted = out:float():sort(2, true) -- descending
  for i=1,N do
    if prediction_sorted[i][1] == labelsCPU[i] then
      top1_epoch = top1_epoch + 1
      top1 = top1 + 1
    end
  end
  top1 = top1 * 100 / N

  if opt.verbose == 1 then
    if trainFlag then
       print(('Epoch: [%d][%d/%d] Time %.3f loss %.4f Top1-%%: %.2f(%.2f) learningRate %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, epochSize, timer:time().real, loss, top1, top1_epoch * 100 / (batchNumber * opt.batchSize ),
          optimStateModel1.learningRate, dataLoadingTime))
    else
       print(('Epoch: [%d][%d/%d] Time %.3f Top1-%%: %.2f(%.2f) learningRate %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, epochTestSize, timer:time().real, top1, top1_epoch * 100 / (batchNumber * opt.batchSize ),
          optimStateModel1.learningRate, dataLoadingTime))
    end
  end
  dataTimer:reset()
end

print('\n')
print('Model parameters: ' .. totalParams)
print('\n')

path = opt.appearanceSamplesPath..'/train/'
classes = paths.dir(path)
table.sort(classes,function(a,b)return a<b end)
table.remove(classes,1) table.remove(classes,1)
for i = 1, #classes do
   print('Class' .. i .. ': ' .. classes[i])
end
print('\n')

epoch = 1
for i=1,opt.nEpochs do
  LoadTrainData()
  epochSize = torch.floor(totalSamples / opt.batchSize)
  if epoch == opt.decayEpoch then --decay LR
    optimStateFusion.learningRate = opt.learningRate * opt.learningRateDecay
    optimStateModel4.learningRate = opt.learningRate * opt.learningRateDecay
    optimStateModel3.learningRate = opt.learningRate * opt.learningRateDecay
    optimStateModel2.learningRate = opt.learningRate * opt.learningRateDecay
    optimStateModel1.learningRate = opt.learningRate * opt.learningRateDecay
  end
	train_acc = Train()
	collectgarbage()

  LoadTestData()
  epochTestSize = torch.floor(totalSamples_val / opt.batchSize)
	test_acc = Test()
	collectgarbage()

	acclogger:add{['train acc.'] = train_acc*100,
		['validation acc.'] = test_acc*100}
	acclogger:style{ ['train acc.'] = '+-', ['validation acc.'] = '+-' }
	acclogger:plot()
	epoch = epoch + 1
end

--Clear the intermediate states in the model before saving to disk
model1:clearState()
model2:clearState()
model3:clearState()
model4:clearState()
fusion:clearState()
   
saveDataParallel(paths.concat(opt.save, 'app1.t7'), model2)
saveDataParallel(paths.concat(opt.save, 'app2.t7'), model4)
saveDataParallel(paths.concat(opt.save, 'aff1.t7'), model1)
saveDataParallel(paths.concat(opt.save, 'aff2.t7'), model3)
saveDataParallel(paths.concat(opt.save, 'fusion.t7'), fusion)
