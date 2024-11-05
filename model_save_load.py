import torch

def save_model(model,optimizer,epoch,train_loss,val_loss,train_acc,val_acc,path):
  torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc
            }, path)
  

def load_model(path,model,optimizer):
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  train_loss = checkpoint['train_loss']
  val_loss = checkpoint['val_loss']
  train_acc = checkpoint['train_acc']
  val_acc = checkpoint['val_acc']
  return train_loss,val_loss,train_acc,val_acc,epoch
