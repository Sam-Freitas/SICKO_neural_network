import torch
import os
import matplotlib.pyplot as plt
from backbones_unet.model.unet import Unet
from utils import *
from network.CMUNeXt import CMUNeXt# cmunext, cmunext_s, cmunext_l

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    if torch.backends.mps.is_available():
        device = torch.device('mps')

img_size = 128

def get_this_model():
    # model = get_model(model_size=50,device=device,freeze_layers=None, weights=False)    
    # model = Unet(in_channels=1, num_classes=1, backbone='convnext_base', activation=torch.nn.GELU).to(device)
    # model = CMUNeXt(input_channel = 1, num_classes = 1).to(device) # base
    # model = CMUNeXt(input_channel = 1, num_classes = 1,dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9]).to(device) ## small
    model = CMUNeXt(input_channel=1,num_classes=1,dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7]).to(device) ## large
    return model

def proc_time(b_sz,img_size, model, n_iter=5):
    time.sleep(1)
    x = torch.rand(b_sz, 1, img_size, img_size).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iter):
        model(x)
    torch.cuda.synchronize()
    end = time.time() - start
    throughput = b_sz * n_iter / end
    print(f"Batch: {b_sz} \t {throughput} samples/sec")
    return (b_sz, throughput, )

# model = get_model(model_size=50,device=device,freeze_layers=None)   
# model = get_model(model_size=101,device=device,freeze_layers=None)  
# model = Unet(in_channels=1, num_classes=1, backbone='convnext_base', activation=torch.nn.ReLU).to(device) 
model = get_this_model()

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))

memory_stats = torch.cuda.memory_stats()

# Calculate available GPU memory
total_memory = torch.cuda.get_device_properties(0).total_memory
available_memory = total_memory - memory_stats["allocated_bytes.all.current"]
print(f"Available GPU memory: {available_memory / 1024**3:.2f} GB")

results = []
for batch_size in range(0,256,2):
    a = proc_time(batch_size,img_size,model=model,n_iter=10)
    results.append(np.asarray(a))

    r = np.asarray(results)
    
    plt.plot(r[:,0],r[:,1],'b')
    plt.plot(r[:,0][np.argmax(r[:,1])],r[:,1][np.argmax(r[:,1])], 'go')#, markersize = 15)
    plt.title(str([r[:,0][np.argmax(r[:,1])],r[:,1][np.argmax(r[:,1])]]))
    plt.xlabel('batch size')
    plt.ylabel('throughput samples/sec')
    plt.savefig('output_throughput.png')
    plt.close('all')

    torch.cuda.empty_cache()

print('eof')