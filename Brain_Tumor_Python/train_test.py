import torch 

def train_step(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optim: torch.optim.Optimizer,
          device: torch.device):
  # put model in train mode
  model.train()

  # setup loss and accuracy
  train_loss, train_acc = 0, 0

  for batch, (X, y) in enumerate(dataloader):
    # send data to target device
    X, y = X.to(device), y.to(device)

    # forward pass
    y_pred = model(X)

    # calculate loss
    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    # optimizer zero grad
    optim.zero_grad()

    # loss backward
    loss.backward()

    # optimizer step
    optim.step()

    # calculate accuracy
    y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")
  return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
  # put model in eval mode
  model.eval()

  # setup loss and accuracy
  test_loss, test_acc = 0, 0

  with torch.inference_mode():
    for batch, (X,y) in enumerate(dataloader):
      # send data to target device
      X, y = X.to(device), y.to(device)

      # forward pass
      test_pred = model(X)

      # calculate loss
      loss = loss_fn(test_pred, y)
      test_loss += loss.item()

      # calc acc
      test_pred_label = test_pred.argmax(dim=1)
      test_acc += (test_pred_label == y).sum().item()/len(test_pred_label)

    # Adjust metrics
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
    return test_loss, test_acc