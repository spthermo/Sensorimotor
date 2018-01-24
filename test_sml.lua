require 'torch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'nngraph'

local opt = {
	loadNet = 'path/to/saved/best/model',
	affordance_sample_pathsPath = 'path/to/affordance/samples',
	appearance_sample_pathsPath = 'path/to/appearance/samples',
	GPU = 1,
	nGPU = 1,
	backend = 'cudnn',
	imageSize = 300,
	cropSize = 224,
	nClasses = 14,
	batchSize = 32,
}
opt = xlua.envparams(opt)

torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU)
paths.dofile('util.lua')

net = torch.load(opt.loadNet .. '/load/best/model.t7')
net:cuda()
cudnn.convert(net, cudnn)

--GPU inputs (preallocate)
affordance_gpu = torch.CudaTensor()
appearance_gpu = torch.CudaTensor()
gt_gpu = torch.CudaTensor()

--set timers
timer = torch.Timer()
dataTimer = torch.Timer()

--image scaling
function scaleImg(inputObject, inputHand)
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

	return outObject, outHand
end

--load test data, all variables are global
function load_test_set()
	path1_test = opt.affordance_sample_pathsPath..'/test/'
	path2_test = opt.appearance_sample_pathsPath..'/test/'
	classes_test = paths.dir(path1_test)
	table.sort(classes_test,function(a,b)return a<b end)
	table.remove(classes_test,1) table.remove(classes_test,1)

	affordance_test={}
	appearance_test={}
	numOfSubjectsInClass_test=torch.FloatTensor(#classes_test):fill(0)

	for i = 1, #classes_test do
		affordance_test[i] = paths.dir(path1_test .. classes_test[i])
		appearance_test[i] = paths.dir(path2_test .. classes_test[i])
		
		table.sort(affordance_test[i],function(a,b)return a<b end)
		table.sort(appearance_test[i],function(a,b)return a<b end)
		
		table.remove(affordance_test[i],1) table.remove(affordance_test[i],1)
		table.remove(appearance_test[i],1) table.remove(appearance_test[i],1)
		
		numOfSubjectsInClass_test[i] = #affordance_test[i]
	end

	totalSamples_test = torch.sum(numOfSubjectsInClass_test)
	print('Total Test Samples: '..totalSamples_test)
	allSamplesIndex_test = torch.FloatTensor(totalSamples_test)

	cnt = 0
	affordanceSum_test = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
	appearanceSum_test = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
	for i = 1, #classes do
			 for j = 1, numOfSubjectsInClass_test[i] do
					cnt = cnt+1
					allSamplesIndex_test[cnt] = i --keeps the class id for all subjects
			 end
	end
end

--runs once sets up the test batches
function test()
	print('==> doing evaluation on test data:')

	tm = torch.Timer()
	temp_affordance = affordance_test
	temp_appearance = appearance_test

	net:evaluate()
	
	samples_cnt=0
	total_batches = torch.floor(totalSamples_test/opt.batchSize)
	remainingSamples = totalSamples_test - (total_batches*opt.batchSize)

	--create batch
	affordance_batch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
	appearance_batch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
	gt = torch.Tensor(opt.batchSize):fill(0)
	for i=1,total_batches do
		for j=1, opt.batchSize do
			if samples_cnt>=totalSamples_test then break end --check set bounds
			samples_cnt=samples_cnt+1
			
			--select class & subject from shuffled data (optional)
			rand_cls = allSamplesIndex_test[samples_cnt]
			subjects_in_cls = #temp_affordance[rand_cls]
			rand_subj = torch.randperm(subjects_in_cls)
			
			--load paths
			affordance_sample_path = path1_test .. classes_test[rand_cls] .. '/' .. temp_affordance[rand_cls][rand_subj[1]] .. '/' .. '001.png'
			appearance_sample_path = path2_test .. classes_test[rand_cls] .. '/' .. temp_affordance[rand_cls][rand_subj[1]] .. '/' .. '003.png'

			--load images
			curr_aff_sample = image.load(affordance_sample_path)
			curr_app_sample = image.load(appearance_sample_path)
			
			--image scaling
			app, aff = scaleImg(curr_app_sample, curr_aff_sample)
			table.remove(temp_affordance[rand_cls],rand_subj[1])
			table.remove(temp_appearance[rand_cls],rand_subj[1])

			affordance_batch[j]:copy(aff)
			appearance_batch[j]:copy(app)
			gt[j] = rand_cls
		end
		Forward(affordance_batch, appearance_batch, gt, false)
	end
	--prepare the last batch
	for j=1, opt.batchSize do
		if j <= remainingSamples then
			samples_cnt=samples_cnt+1
			
			rand_cls = allSamplesIndex_test[samples_cnt]
			subjects_in_cls = #temp_affordance[rand_cls]
			rand_subj = torch.randperm(subjects_in_cls)
			
			affordance_sample_path = path1_test .. classes_test[rand_cls] .. '/' .. temp_affordance[rand_cls][rand_subj[1]] .. '/' .. '001.png'
			appearance_sample_path = path2_test .. classes_test[rand_cls] .. '/' .. temp_affordance[rand_cls][rand_subj[1]] .. '/' .. '003.png'

			curr_aff_sample = image.load(affordance_sample_path)
			curr_app_sample = image.load(appearance_sample_path)
			
			app, aff = scaleImg(curr_app_sample, curr_aff_sample)
			table.remove(temp_affordance[rand_cls],rand_subj[1])
			table.remove(temp_appearance[rand_cls],rand_subj[1])

			affordance_batch[j]:copy(aff)
			appearance_batch[j]:copy(app)
			gt[j] = rand_cls
		else
			affordance_batch[j]:fill(0)
			appearance_batch[j]:fill(0)
			gt[j] = 0
		end
	end
	forward_batch(affordance_batch, appearance_batch, gt)
	print('Samples evaluated: ' .. samples_cnt)
end

--handles the forward pass, accuracy and fills the confusion matrix
function forward_batch(affordance_cpu, appearance_cpu, gt_cpu)
	local prediction_sorted
	affordance_gpu:resize(affordance_cpu:size()):copy(affordance_cpu)
	appearance_gpu:resize(appearance_cpu:size()):copy(appearance_cpu)
	gt_gpu:resize(gt_cpu:size()):copy(gt_cpu)

	out = net:forward({affordance_gpu, appearance_gpu})

	_,prediction_sorted = out:float():sort(2, true)
	for i=1, opt.batchSize do
		if gt_cpu[i] > 0 then
			conf:add(prediction_sorted[i][1], gt_cpu[i])
		end
	end
end

--print class names
path = opt.appearance_sample_pathsPath..'/test/'
classes = paths.dir(path)
table.sort(classes,function(a,b)return a<b end)
table.remove(classes,1) table.remove(classes,1)
for i = 1, #classes do
	 print('Class' .. i .. ': ' .. classes[i])
end
print('\n')

--init confusion matrix	
conf = optim.ConfusionMatrix(#classes,{1,2,3,4,5,6,7,8,9,10,11,12,13,14})
conf:zero()

--test
load_test_set()
test()

--print and save results
print(conf)
filename = paths.concat(opt.loadNet, 'confusionMatrix.csv')
print('=> saving confusion matrix to: ' .. opt.loadNet)
file = io.open(filename, 'w')
file:write(tostring(conf))
file:close()
image.display(conf:render())