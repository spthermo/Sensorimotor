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
cmd:option('-cache', './Results/Baseline/', 'subdirectory in which to save/log experiments')
cmd:option('-appearanceSamplesPath', '', 'SOR3D colorized object depthmaps')    
cmd:option('-GPU',                2, 'Default preflossed GPU')
cmd:option('-nGPU',               1, 'Number of GPUs to use by default')
cmd:option('-backend',      'cudnn', 'Default backend')
------------- Data options ------------------------
cmd:option('-imageSize',         300,    'Smallest side of the resized image')
cmd:option('-cropSize',          224,    'Height and Width of image crop to be used as input layer')
cmd:option('-cnnSize',           4096,    'size of FC6')
cmd:option('-augment',             0,     'use data augmentation')
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
local model = createModel(opt.nGPU)

for i=40,39,-1 do
  model:remove(i)
end
model:add(nn.Linear(opt.cnnSize, opt.nClasses))
model:add(nn.LogSoftMax())

model:cuda()
print(model)

if opt.backend == 'cudnn' then
	require 'cudnn'
	cudnn.convert(model, cudnn)
elseif opt.backend ~= 'nn' then
	lossor'Unsupported backend'
end

criterion = nn.ClassNLLCriterion()
criterion:cuda()

--Mean & Std of SOR3D processed data
local meanstdApp = {mean = {0.016,0.01, 0}, std = {0.0423, 0.0284, 0}}

--GPU inputs (preallocate)
local inputsCudaAppearance = torch.CudaTensor()
local labelsCuda = torch.CudaTensor()

--Set timers
local timer = torch.Timer()
local dataTimer = torch.Timer()

--Get model parameters
local Weights, Gradients = model:getParameters()
local totalParams = Weights:size(1)

local optimState = {
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
local function hflip(x1)
  local y1
  if torch.random(0,1) > 0.5 then
    y1 = image.hflip(x1) 
    return y1
  else
    return x1
  end
end

function randomCrop(inputObject)
  local diff = opt.imageSize - opt.cropSize
  local iW = inputObject:size(3)
  local iH = inputObject:size(2)
  if iW > opt.imageSize or iH > opt.imageSize then
    inputObject = image.scale(inputObject, opt.imageSize, opt.imageSize, 'bilinear')
  end

  local w1 = math.ceil(torch.uniform(1e-2, diff))
  local h1 = math.ceil(torch.uniform(1e-2, diff))
  local outObject = image.crop(inputObject, w1, h1, w1 + opt.cropSize, h1 + opt.cropSize)

  assert(outObject:size(3) == opt.cropSize)
  assert(outObject:size(2) == opt.cropSize)

  for c=1,3 do -- channels
    if meanApp then outObject[{{c},{},{}}]:add(-meanstdApp.mean[c]):div(meanstdApp.std[c]) end
  end

  return outObject
end

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

function LoadTrainData()
  path = opt.appearanceSamplesPath..'/train/'
  classes = paths.dir(path)
  table.sort(classes,function(a,b)return a<b end)
  table.remove(classes,1) table.remove(classes,1)

	appearance={}
	numOfSubjectsInClass=torch.FloatTensor(#classes):fill(0)

	for i = 1, #classes do
		appearance[i] = paths.dir(path .. classes[i])
		table.sort(appearance[i],function(a,b)return a<b end)
		table.remove(appearance[i],1) table.remove(appearance[i],1)
		numOfSubjectsInClass[i] = #appearance[i]
	end

	totalSamples = torch.sum(numOfSubjectsInClass)
	print('Total Train Samples: '..totalSamples)
	allSamplesIndex = torch.FloatTensor(totalSamples)

	cnt = 0
  appearanceSum = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
	for i = 1, #classes do
    for j = 1, numOfSubjectsInClass[i] do
    	cnt = cnt+1
    	allSamplesIndex[cnt] = i
    end
	end
end

function LoadTestData()
  path_val = opt.appearanceSamplesPath..'/val/'
  classes_val = paths.dir(path_val)
  table.sort(classes_val,function(a,b)return a<b end)
  table.remove(classes_val,1) table.remove(classes_val,1)

  appearance_val={}
  numOfSubjectsInClass_val=torch.FloatTensor(#classes_val):fill(0)

  for i = 1, #classes_val do
    appearance_val[i] = paths.dir(path_val .. classes_val[i])
    table.sort(appearance_val[i],function(a,b)return a<b end)
    table.remove(appearance_val[i],1) table.remove(appearance_val[i],1)
    numOfSubjectsInClass_val[i] = #appearance_val[i]
  end

  totalSamples_val = torch.sum(numOfSubjectsInClass_val)
  print('Total Validation Samples: '..totalSamples_val)
  allSamplesIndex_val = torch.FloatTensor(totalSamples_val)

  cnt = 0
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
  temporalAppearance = appearance

	model:training()

	tm = torch.Timer()
	top1_epoch = 0
	loss_epoch = 0
	batchNumber = 0
	rand_cnts=torch.randperm(totalSamples)
	samples_cnt=0
   
	--Create batch
	appearanceBatch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
	labels = torch.Tensor(opt.batchSize):fill(0)
	for i=1,epochSize do
      for j=1,opt.batchSize do
        	if samples_cnt>=totalSamples then break end
      	samples_cnt=samples_cnt+1
      	randomClass = allSamplesIndex[rand_cnts[samples_cnt]]
      	numOfSubjectsInRandomClass = #temporalAppearance[randomClass]
      	randomSubject = torch.randperm(numOfSubjectsInRandomClass)
      	appearanceSample = path .. classes[randomClass] .. '/' .. temporalAppearance[randomClass][randomSubject[1]] .. '/' .. '003.png'

      	currAppearanceSample = image.load(appearanceSample)
        
        if opt.augment == 1 then
          appFlipped = hflip(currAppearanceSample)
          app = randomCrop(appFlipped)
        else
          app = scaleImg(currAppearanceSample)
        end
      	table.remove(temporalAppearance[randomClass],randomSubject[1])
      
      	appearanceBatch[j]:copy(app)
      	labels[j] = randomClass
      end
      Forward(appearanceBatch, labels, true)
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
  temporalAppearance = appearance_val

  model:evaluate()

  tm = torch.Timer()
  top1_epoch = 0
  loss_epoch = 0
  batchNumber = 0
  rand_cnts=torch.randperm(totalSamples_val)
  samples_cnt=0

  --Create batch
  appearanceBatch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
  labels = torch.Tensor(opt.batchSize):fill(0)
  for i=1,epochTestSize do
    for j=1,opt.batchSize do
      if samples_cnt>=totalSamples then break end
      samples_cnt=samples_cnt+1
      randomClass = allSamplesIndex_val[rand_cnts[samples_cnt]]
      numOfSubjectsInRandomClass = #temporalAppearance[randomClass]
      randomSubject = torch.randperm(numOfSubjectsInRandomClass)
      appearanceSample = path_val .. classes_val[randomClass] .. '/' .. temporalAppearance[randomClass][randomSubject[1]] .. '/' .. '003.png'

      currAppearanceSample = image.load(appearanceSample)
      
      app = scaleImg(currAppearanceSample)
      table.remove(temporalAppearance[randomClass],randomSubject[1])

      appearanceBatch[j]:copy(app)
      labels[j] = randomClass
    end
    Forward(appearanceBatch, labels, false)
  end

  cutorch.synchronize()

  top1_epoch = top1_epoch * 100 / (opt.batchSize * epochTestSize )

  print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                   .. 'accuracy(%%):\t top-1 %.2f\t',
                   epoch, tm:time().real, top1_epoch))
  print('\n')

  return top1_epoch
end

function Forward(inputsCPUappearance, labelsCPU, trainFlag)
  cutorch.synchronize()
  collectgarbage()
  local dataLoadingTime = dataTimer:time().real
  timer:reset()

  inputsCudaAppearance:resize(inputsCPUappearance:size()):copy(inputsCPUappearance)
  labelsCuda:resize(labelsCPU:size()):copy(labelsCPU)
  local loss
  local N = opt.batchSize;
  if trainFlag then
  	model:zeroGradParameters()
  end
  out = model:forward(inputsCudaAppearance)

  if trainFlag then
    loss = criterion:forward(out, labelsCuda)
  	gradOut = criterion:backward(out, labelsCuda)
  	model:backward(inputsCudaAppearance, gradOut:cuda())
  	
  	feval = function()
  		return loss, Gradients
  	end    	
  	optim.sgd(feval, Weights, optimState)
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
        optimState.learningRate, dataLoadingTime))
    else
      print(('Epoch: [%d][%d/%d] Time %.3f Top1-%%: %.2f(%.2f) learningRate %.0e DataLoadingTime %.3f'):format(
        epoch, batchNumber, epochTestSize, timer:time().real, top1, top1_epoch * 100 / (batchNumber * opt.batchSize ),
        optimState.learningRate, dataLoadingTime))
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
    optimState.learningRate = opt.learningRate * opt.learningRateDecay
  end
	train_acc = Train()
	collectgarbage()

  LoadTestData()
  epochTestSize = torch.floor(totalSamples_val / opt.batchSize)
	test_acc = Test()
	collectgarbage()
	acclogger:add{['train acc.'] = train_acc*100,
		['test acc.'] = test_acc*100}
	acclogger:style{ ['train acc.'] = '+-', ['test acc.'] = '+-' }
	acclogger:plot()
	epoch = epoch + 1
end

--Clear the intermediate states in the model before saving to disk
model:clearState()
saveDataParallel(paths.concat(opt.save, 'baseline.t7'), model)

