import gc
import torch

def model_test(model,test_loader):
 model.eval()
 images = []
 predictions = []   
 treshold = 0.5
 correct =0
 target_sum=0
 for i ,(img, target, img_name) in enumerate(test_loader):
    images.append(img_name)
    img = img.float()
    img = img.cuda()
    target = target.float()
    target = target.cuda()
    with torch.no_grad():
      output = torch.sigmoid(model(img)).float()
    
    output = torch.where(output > treshold, 1,0)
    predictions.append(output)
    #pdb.set_trace
    res = output==target
    
    for tensor in res:
        if False in tensor:
            continue
        else:
            correct += 1
    
    
    target_sum += len(target)
    avg_acc = correct/((target_sum))
    del img
    del output
    
    gc.collect() 
    torch.cuda.empty_cache()
    
    
 return images, predictions, avg_acc
