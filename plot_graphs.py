import matplotlib.pyplot as plt


def plot_graphs(train_loss,val_loss,train_acc,val_acc,epcohs, fig_path):
  plt.figure(figsize=(30,10))

  plt.subplot(1,2,1)
  plt.title("Loss")
  plt.plot(list(range(0,epcohs)),train_loss, label='Train')
  plt.plot(list(range(0,epcohs)), val_loss, label='Validation')
  plt.xlabel('Epochs')
  plt.ylabel('Rate')
  plt.legend()
    
  plt.subplot(1,2,2)
  plt.title("Accuracy")
  plt.plot(list(range(0,epcohs)),train_acc, label='Train')
  plt.plot(list(range(0,epcohs)), val_acc, label='Validation')
  plt.xlabel('Epochs')
  plt.ylabel('Rate')
  plt.legend()
  plt.savefig(fig_path)
