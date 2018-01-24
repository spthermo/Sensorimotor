require 'torch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'cunn'
require 'cudnn'
require 'image'
require 'nngraph'
local json = require 'cjson'

local opt = {
    save = 'logs/',
    affordance_sample_pathsPath = 'path/to/affordance/samples',
    appearance_sample_pathsPath = 'path/to/appearance/samples',
    GPU = 1,
    nGPU = 1,
    backend = 'cudnn',
    netType = 'vgg_16_pretrained',
    cnnSize = 4096,
    imageSize = 300,
    cropSize = 224,
    augment = 0, 
    nClasses = 14,
    nEpochs = 30,
    batchSize = 32,
    verbose = 1, --print batch loss, acc
    learningRate = 0.005,
    learningRateDecay = 0.2,
    momentum = 0.9,
    weightDecay = 5e-4,
    lambda = 0.0005,
}
opt = xlua.envparams(opt)

print('Will save at ' .. opt.save)
paths.mkdir(opt.save)
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(opt.GPU)
paths.dofile('util.lua')

-- load VGG16 pretrained
paths.dofile('models/' .. opt.netType .. '.lua')
local vgg16 = createModel(opt.nGPU, opt.backend)
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
    :add(nn.JoinTable(2))
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
    :add(nn.JoinTable(2))
    :add(nn.Linear(2*opt.cnnSize, opt.cnnSize))
    :add(nn.ReLU(true))
    :add(nn.Dropout(0.5))
    :add(nn.Linear(opt.cnnSize, opt.nClasses))
    :add(nn.LogSoftMax())

vgg16 = nil

-- build graph
in_app = nn.Identity()()
in_aff = nn.Identity()()

aff_1 = in_aff
    - model1

app_1 = in_app
    - model2

aff_2 = aff_1
    - model3

app_2 = {aff_1, app_1}
    - model4

fus = {aff_2, app_2}
    - fusion

net = nn.gModule({in_aff, in_app}, {fus})
net:cuda()
cudnn.convert(net, cudnn)
weights, grads = net:getParameters()
print('Model parameters: ' .. weights:size(1))

local optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay,
}

criterion = nn.ClassNLLCriterion()
criterion:cuda()

--GPU inputs (preallocate)
affordance_gpu = torch.CudaTensor()
appearance_gpu = torch.CudaTensor()
gt_gpu = torch.CudaTensor()

--set timers
timer = torch.Timer()
dataTimer = torch.Timer()

--create logger.
function log(t) print('json_stats: ' .. json.encode(tablex.merge(t,opt,true))) end

loss_epoch = 0

--data augmentation
function hflip(x1,x2)
    local y1, y2
    if torch.random(0,1) > 0.5 then
        y1 = image.hflip(x1) 
        y2 = image.hflip(x2)
        return y1, y2
    else
        return x1, x2
    end
end

--random cropping
function randomCrop(inputObject, inputHand)
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

    return outObject, outHand
end

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

--handles the LR based on validation accuracy memory
function ReduceLRonPlateau(acc)
    if ((acc < val_acc_list[2]) and (val_acc_list[2] < val_acc_list[1])) then
        optimState.learningRate = optimState.learningRate * opt.learningRateDecay
    end
    val_acc_list[1] = val_acc_list[2]
    val_acc_list[2] = acc
end

--clear the intermediate states in the model before saving to disk
function Save_best_model(acc, epoch)
    if (acc < best_val_acc) then
        best_val_acc = acc
        torch.save(paths.concat(opt.save, 'net_best_e' .. epoch .. '.t7'), net:clearState())
    end
end

--load training data, all variables are global
function load_train_set()
    path1 = opt.affordance_sample_pathsPath..'/train/'
    path2 = opt.appearance_sample_pathsPath..'/train/'
    classes = paths.dir(path1)
    table.sort(classes,function(a,b)return a<b end)
    table.remove(classes,1) table.remove(classes,1)

    affordance = {}
    appearance = {}
    totalSamples = 0
    numOfSubjectsInClass = torch.Tensor(#classes):fill(0)

    for i = 1, #classes do
        affordance[i] = paths.dir(path1 .. classes[i])
        appearance[i] = paths.dir(path2 .. classes[i])
        
        table.sort(affordance[i],function(a,b)return a<b end)
        table.sort(appearance[i],function(a,b)return a<b end)
        
        table.remove(affordance[i],1) table.remove(affordance[i],1)
        table.remove(appearance[i],1) table.remove(appearance[i],1)
        
        numOfSubjectsInClass[i] = #appearance[i]
        totalSamples = totalSamples + #appearance[i]
    end

    print('Total Train Samples: '..totalSamples)
    allSamplesIndex = torch.FloatTensor(totalSamples)

    cnt = 0
    affordanceSum = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
    appearanceSum = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
    for i = 1, #classes do
        for j = 1, numOfSubjectsInClass[i] do
            cnt = cnt+1
            allSamplesIndex[cnt] = i --keeps the class id for all subjects
        end
    end
end

--load validation data, all variables are global
function load_val_set()
    path1_val = opt.affordance_sample_pathsPath..'/val/'
    path2_val = opt.appearance_sample_pathsPath..'/val/'
    classes_val = paths.dir(path1_val)
    table.sort(classes_val,function(a,b)return a<b end)
    table.remove(classes_val,1) table.remove(classes_val,1)

    affordance_val={}
    appearance_val={}
    totalSamples_val = 0
    numOfSubjectsInClass_val=torch.Tensor(#classes_val):fill(0)

    for i = 1, #classes_val do
        affordance_val[i] = paths.dir(path1_val .. classes_val[i])
        appearance_val[i] = paths.dir(path2_val .. classes_val[i])
        
        table.sort(affordance_val[i],function(a,b)return a<b end)
        table.sort(appearance_val[i],function(a,b)return a<b end)
        
        table.remove(affordance_val[i],1) table.remove(affordance_val[i],1)
        table.remove(appearance_val[i],1) table.remove(appearance_val[i],1)
        
        numOfSubjectsInClass_val[i] = #affordance_val[i]
        totalSamples_val = totalSamples_val + #appearance_val[i]
    end

    print('Total Validation Samples: '..totalSamples_val)
    allSamplesIndex_val = torch.FloatTensor(totalSamples_val)

    cnt = 0
    affordanceSum_val = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
    appearanceSum_val = torch.Tensor(3, opt.imageSize, opt.imageSize):fill(0)
    for i = 1, #classes do
        for j = 1, numOfSubjectsInClass_val[i] do
            cnt = cnt+1
            allSamplesIndex_val[cnt] = i --keeps the class id for all subjects
        end
    end
end

--runs for each epoch, sets up the training batches
function train(epoch)
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch)

    tm = torch.Timer()
    temp_affordance = affordance --keep a temp image of the loaded data (paths)
    temp_appearance = appearance

    net:training()
    
    top1_epoch, batchNumber, loss_epoch, samples_cnt = 0, 0, 0, 0

    rand_cnts=torch.randperm(totalSamples)
     
    --Create batch
    affordance_batch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
    appearance_batch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
    gt = torch.Tensor(opt.batchSize):fill(0)
    for i=1,epochSize do
        for j=1,opt.batchSize do
            if samples_cnt>=totalSamples then break end --check set bounds
            samples_cnt=samples_cnt+1
            
            --select class & subject from shuffled data
            rand_cls = allSamplesIndex[rand_cnts[samples_cnt]]
            subjects_in_cls = #temp_affordance[rand_cls]
            rand_subj = torch.randperm(subjects_in_cls)
            
            --load paths
            affordance_sample_path = path1 .. classes[rand_cls] .. '/' .. temp_affordance[rand_cls][rand_subj[1]] .. '/' .. '001.png'
            appearance_sample_path = path2 .. classes[rand_cls] .. '/' .. temp_affordance[rand_cls][rand_subj[1]] .. '/' .. '003.png'

            --load images
            curr_aff_sample = image.load(affordance_sample_path)
            curr_app_sample = image.load(appearance_sample_path)

            --augmentation (optional)
            if opt.augment == 1 then
                appFlipped, affFlipped = hflip(curr_app_sample, curr_aff_sample)
                app, aff = randomCrop(appFlipped, affFlipped)
            else
                app, aff = scaleImg(curr_app_sample, curr_aff_sample)
            end
            table.remove(temp_affordance[rand_cls],rand_subj[1])
            table.remove(temp_appearance[rand_cls],rand_subj[1])
        
            affordance_batch[j]:copy(aff)
            appearance_batch[j]:copy(app)
            gt[j] = rand_cls
        end
        forward_batch(affordance_batch, appearance_batch, gt, epoch, true)
     end

    cutorch.synchronize()

    top1_epoch = top1_epoch * 100 / (opt.batchSize * epochSize )
    loss_epoch = loss_epoch / epochSize

    print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                                         .. 'average loss (per batch): %.2f \t '
                                         .. 'accuracy(%%):\t top-1 %.2f\t',
                                         epoch, tm:time().real, loss_epoch, top1_epoch))
    print('\n')

    return top1_epoch
end

--runs for each epoch, sets up the validation batches
function test(epoch)
    print('==> doing epoch on validation data:')
    print("==> online epoch # " .. epoch)

    tm = torch.Timer()
    temp_affordance = affordance_val --keep a temp image of the loaded data (paths)
    temp_appearance = appearance_val

    net:evaluate()
    
    top1_epoch, batchNumber, samples_cnt = 0, 0, 0
    rand_cnts=torch.randperm(totalSamples_val)

    --create batch
    affordance_batch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
    appearance_batch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
    gt = torch.Tensor(opt.batchSize):fill(0)
    for i=1,epochTestSize do
        for j=1,opt.batchSize do
            if samples_cnt>=totalSamples_val then break end --check set bounds
            samples_cnt=samples_cnt+1
            
            --select class & subject from shuffled data
            rand_cls = allSamplesIndex_val[rand_cnts[samples_cnt]]
            subjects_in_cls = #temp_affordance[rand_cls]
            rand_subj = torch.randperm(subjects_in_cls)
            
            --load paths
            affordance_sample_path = path1_val .. classes_val[rand_cls] .. '/' .. temp_affordance[rand_cls][rand_subj[1]] .. '/' .. '001.png'
            appearance_sample_path = path2_val .. classes_val[rand_cls] .. '/' .. temp_affordance[rand_cls][rand_subj[1]] .. '/' .. '003.png'

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
        forward_batch(affordance_batch, appearance_batch, gt, epoch, false)
    end

    cutorch.synchronize()
    top1_epoch = top1_epoch * 100 / (opt.batchSize * epochTestSize )
    print(string.format('Epoch: [%d][VALIDATION SUMMARY] Total Time(s): %.2f\t'
                                     .. 'accuracy(%%):\t top-1 %.2f\t',
                                     epoch, tm:time().real, top1_epoch))
    print('\n')

    return top1_epoch
end

--handles the forward/backward pass, gradients, accuracies etc (train or val based on flag)
function forward_batch(affordance_cpu, appearance_cpu, gt_cpu, epoch, trainFlag)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    local loss, top1, prediction_sorted

    timer:reset()

    affordance_gpu:resize(affordance_cpu:size()):copy(affordance_cpu)
    appearance_gpu:resize(appearance_cpu:size()):copy(appearance_cpu)
    gt_gpu:resize(gt_cpu:size()):copy(gt_cpu)
    
    if trainFlag then grads:zero() end

    out = net:forward({affordance_gpu, appearance_gpu}) --forward pass
    if trainFlag then
        feval = function(weights)
            loss = criterion:forward(out, gt_gpu)
            gradOut = criterion:backward(out, gt_gpu)
            net:backward({affordance_gpu, appearance_gpu}, gradOut) 
            return loss, grads
        end
        optim.sgd(feval, weights, optimState)
    end

    batchNumber = batchNumber + 1
    if trainFlag then loss_epoch = loss_epoch + loss end
    
    top1 = 0
    _,prediction_sorted = out:float():sort(2, true) -- descending
    for i=1, opt.batchSize do
        if prediction_sorted[i][1] == gt_cpu[i] then
            top1_epoch = top1_epoch + 1
            top1 = top1 + 1
        end
    end
    top1 = top1 * 100 / opt.batchSize

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

--print class names
path = opt.appearance_sample_pathsPath..'/train/'
classes = paths.dir(path)
table.sort(classes,function(a,b)return a<b end)
table.remove(classes,1) table.remove(classes,1)
for i = 1, #classes do
     print('Class' .. i .. ': ' .. classes[i])
end
print('\n')

--inits
val_acc_list = {100,100}
best_val_acc = 100

--main loop
for i=1,opt.nEpochs do
    
    --train
    load_train_set()
    epochSize = torch.floor(totalSamples / opt.batchSize)
    train_acc = train(i)
    collectgarbage()

    --val
    load_val_set(i)
    epochTestSize = torch.floor(totalSamples_val / opt.batchSize)
    test_acc = test(i)
    collectgarbage()

    --keep log
    log
    {
        loss = loss_epoch,
        train_loss = loss_epoch,
        train_acc = train_acc,
        epoch = i,
        test_acc = test_acc,
        lr = optimState.learningRate,
        train_time = 0,
        test_time = 0,
        n_parameters = weights:size(1),
    }

    ReduceLRonPlateau(test_acc) --handles lr
    Save_best_model(test_acc, i) --save best model
end
