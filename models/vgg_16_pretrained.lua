require 'loadcaffe'

function createModel(nGPU,backend)
   if not paths.dirp('models/VGG_16') then
      print('=> Downloading VGG ILSVRC-2014 16-layer model weights')
     os.execute('mkdir models/VGG_16')
     local caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
     local proto_url = 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt'
     os.execute('wget --output-document models/VGG_16/VGG_ILSVRC_16_layers.caffemodel ' .. caffemodel_url)
     os.execute('wget --output-document models/VGG_16/deploy.prototxt '              .. proto_url)
   end
   
   local proto      = 'models/VGG_16/deploy.prototxt'
   local caffemodel = 'models/VGG_16/VGG_ILSVRC_16_layers.caffemodel'
   
   if backend == 'cudnn' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cudnn')   
   elseif backend == 'cunn' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cunn')   
   elseif backend == 'cnn2' then
      pretrain = loadcaffe.load(proto, caffemodel, 'cnn2')
   else
      print 'not supported module'
      exit(1)
   end
  
   return pretrain
end
