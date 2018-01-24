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
    appearance_sample_paths = 'path/to/appearance/samples',
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
    batchSize = 64,
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
net = vgg16:clone()

net:remove(40)
net:remove(39)

net:add(nn.Linear(opt.cnnSize, opt.nClasses))
net:add(nn.LogSoftMax())

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
appearance_gpu = torch.CudaTensor()
gt_gpu = torch.CudaTensor()

--set timers
timer = torch.Timer()
dataTimer = torch.Timer()

--create logger.
function log(t) print('json_stats: ' .. json.encode(tablex.merge(t,opt,true))) end

--data augmentation
function hflip(x)
    local y
    if torch.random(0,1) > 0.5 then
        y = image.hflip(x)
        return y
    else
        return x
    end
end

--random cropping
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

    return outObject
end

--image scaling
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

    return outObject
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
    path = opt.appearance_sample_paths..'/train/'
    classes = paths.dir(path)
    table.sort(classes,function(a,b)return a<b end)
    table.remove(classes,1) table.remove(classes,1)

    appearance = {}
    totalSamples = 0
    numOfSubjectsInClass = torch.Tensor(#classes):fill(0)

    for i = 1, #classes do
        appearance[i] = paths.dir(path .. classes[i])
        
        table.sort(appearance[i],function(a,b)return a<b end)
        table.remove(appearance[i],1) table.remove(appearance[i],1)
        
        numOfSubjectsInClass[i] = #appearance[i]
        totalSamples = totalSamples + #appearance[i]
    end

    print('Total Train Samples: '..totalSamples)
    allSamplesIndex = torch.FloatTensor(totalSamples)

    cnt = 0
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
    path_val = opt.appearance_sample_paths..'/val/'
    classes_val = paths.dir(path_val)
    table.sort(classes_val,function(a,b)return a<b end)
    table.remove(classes_val,1) table.remove(classes_val,1)

    appearance_val={}
    totalSamples_val = 0
    numOfSubjectsInClass_val=torch.Tensor(#classes_val):fill(0)

    for i = 1, #classes_val do
        appearance_val[i] = paths.dir(path_val .. classes_val[i])
        
        table.sort(appearance_val[i],function(a,b)return a<b end)
        table.remove(appearance_val[i],1) table.remove(appearance_val[i],1)
        
        numOfSubjectsInClass_val[i] = #appearance_val[i]
        totalSamples_val = totalSamples_val + #appearance_val[i]
    end

    print('Total Validation Samples: '..totalSamples_val)
    allSamplesIndex_val = torch.FloatTensor(totalSamples_val)

    cnt = 0
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
    temp_appearance = appearance --keep a temp image of the loaded data (paths)

    net:training()
    
    top1_epoch, batchNumber, loss_epoch, samples_cnt = 0, 0, 0, 0

    rand_cnts=torch.randperm(totalSamples)
     
    --Create batch
    appearance_batch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
    gt = torch.Tensor(opt.batchSize):fill(0)
    for i=1,epochSize do
        for j=1,opt.batchSize do
            if samples_cnt>=totalSamples then break end --check set bounds
            samples_cnt=samples_cnt+1
            
            --select class & subject from shuffled data
            rand_cls = allSamplesIndex[rand_cnts[samples_cnt]]
            subjects_in_cls = #temp_appearance[rand_cls]
            rand_subj = torch.randperm(subjects_in_cls)
            
            --load paths
            appearance_sample_path = path .. classes[rand_cls] .. '/' .. temp_appearance[rand_cls][rand_subj[1]] .. '/' .. '003.png' --can choose random frame from each session.

            --load images
            curr_app_sample = image.load(appearance_sample_path)

            --augmentation (optional)
            if opt.augment == 1 then
                appFlipped = hflip(curr_app_sample)
                app = randomCrop(appFlipped)
            else
                app = scaleImg(curr_app_sample)
            end
            table.remove(temp_appearance[rand_cls],rand_subj[1])
        
            appearance_batch[j]:copy(app)
            gt[j] = rand_cls
        end
        forward_batch(appearance_batch, gt, epoch, true)
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
    temp_appearance = appearance_val --keep a temp image of the loaded data (paths)

    net:evaluate()
    
    top1_epoch, batchNumber, samples_cnt = 0, 0, 0
    rand_cnts=torch.randperm(totalSamples_val)

    --create batch
    appearance_batch = torch.Tensor(opt.batchSize, 3, opt.cropSize, opt.cropSize):fill(0)
    gt = torch.Tensor(opt.batchSize):fill(0)
    for i=1,epochTestSize do
        for j=1,opt.batchSize do
            if samples_cnt>=totalSamples_val then break end --check set bounds
            samples_cnt=samples_cnt+1
            
            --select class & subject from shuffled data
            rand_cls = allSamplesIndex_val[rand_cnts[samples_cnt]]
            subjects_in_cls = #temp_appearance[rand_cls]
            rand_subj = torch.randperm(subjects_in_cls)
            
            --load paths
            appearance_sample_path = path_val .. classes_val[rand_cls] .. '/' .. temp_appearance[rand_cls][rand_subj[1]] .. '/' .. '003.png'

            --load images
            curr_app_sample = image.load(appearance_sample_path)

            --image scaling
            app = scaleImg(curr_app_sample)
            table.remove(temp_appearance[rand_cls],rand_subj[1])

            appearance_batch[j]:copy(app)
            gt[j] = rand_cls
        end
        forward_batch(appearance_batch, gt, epoch, false)
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
function forward_batch(appearance_cpu, gt_cpu, epoch, trainFlag)
    cutorch.synchronize()
    collectgarbage()
    local dataLoadingTime = dataTimer:time().real
    local loss, top1, prediction_sorted

    timer:reset()

    appearance_gpu:resize(appearance_cpu:size()):copy(appearance_cpu)
    gt_gpu:resize(gt_cpu:size()):copy(gt_cpu)
    
    if trainFlag then grads:zero() end

    out = net:forward(appearance_gpu) --forward pass
    if trainFlag then
        feval = function(weights)
            loss = criterion:forward(out, gt_gpu)
            gradOut = criterion:backward(out, gt_gpu)
            net:backward(appearance_gpu, gradOut) 
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
path = opt.appearance_sample_paths..'/train/'
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