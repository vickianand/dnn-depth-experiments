
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def validation(model, criterion, validloader, device):
    model.eval()
    n = len(validloader)
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(validloader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # forward
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss_sum += loss.cpu().numpy()
            total += labels.size(0)
            correct += (predicted == labels).float().sum().cpu().numpy()
    accuracy = 100 * correct / total
    return loss_sum / n, accuracy


def train(model, optimizer, criterion, trainloader, device):
    model.train()
    n = len(trainloader)
    loss_sum = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = False
        labels.requires_grad = False

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.detach(), 1)
        loss.backward()
        optimizer.step()
        loss_sum = loss_sum + loss.detach().cpu().numpy()
        total += labels.detach().shape[0]
        correct += (predicted == labels).detach().float().sum().cpu().numpy()
    accuracy = 100 * correct / total
    return loss_sum / n, accuracy


def test(model, testloader, device, exp_name):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        "Accuracy of the model on the 10000 test images: %d %%"
        % (100 * correct / total)
    )
    with open(f'./results/{exp_name}/test.txt', 'w') as f:
        f.write("{:.4f}".format(100. * correct / total))


def train_loop(model, optimizer, criterion, trainloader, validloader, device, n_epoch, exp_name):
    lambda1 = lambda epoch: 0.95 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    
    trainLoss = []
    trainAccuracy = []
    valLoss = []
    valAccuracy = []
    best_val_accuracy = 0.0
    for epoch in range(n_epoch):  # loop over the dataset multiple times
        scheduler.step(epoch)
        train_loss, train_accuracy = train(
            model, optimizer, criterion, trainloader, device
        )
        print(
            "epoch %d Train loss: %.6f, Accuracy: %.2f"
            % (epoch, train_loss, train_accuracy)
        )
        val_loss, val_accuracy = validation(model, criterion, validloader, device)
        print(
            "epoch %d Validation loss: %.6f, Accuracy: %.2f"
            % (epoch, val_loss, val_accuracy)
        )
        trainLoss.append(train_loss)
        trainAccuracy.append(train_accuracy)
        valLoss.append(val_loss)
        valAccuracy.append(val_accuracy)
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f'./models/{exp_name}/best.net')

    x_axis = np.arange(0, n_epoch, 1)

    plt.figure()
    plt.plot(x_axis, trainLoss, label="Train Loss")
    plt.plot(x_axis, valLoss, label="Validation Loss")
    plt.title("{exp_name} Average Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.savefig(f'./results/{exp_name}/loss.png')
    plt.close()

    plt.figure()
    plt.plot(x_axis, trainAccuracy, label="Train Accuracy")
    plt.plot(x_axis, valAccuracy, label="Validation Accuracy")
    plt.title("{exp_name} Learning curves")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f'./results/{exp_name}/accuracy.png')

    plt.close()

