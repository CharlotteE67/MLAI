from LeNet5 import LeNet5
import numpy as np
import os
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 256
    train_dataset = mnist.MNIST(root='./train', train=True, transform=ToTensor(), download=True)
    test_dataset = mnist.MNIST(root='./test', train=False, transform=ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    model = LeNet5().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = CrossEntropyLoss()
    all_epoch = 100
    # prev_acc = 0
    for current_epoch in range(all_epoch):
        model.train()
        for idx, (train_x, train_label) in enumerate(train_loader):
            print(train_x[0].shape)
            exit(0)
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            predict_y = model(train_x.float())
            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            optimizer.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()

        for idx, (test_x, test_label) in enumerate(test_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y = torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num
        # print('Epoch accuracy: {:.3f}'.format(acc), flush=True)
        print('Epoch %d/%d: Loss=%.6f, Accuracy=%.6f' % (current_epoch, all_epoch, loss.item(), acc))
        if not os.path.isdir("models"):
            os.mkdir("models")
        torch.save(model, 'models/mnist_{:.3f}.pkl'.format(acc))
        # if np.abs(acc - prev_acc) < 1e-4:
        #     break
        # prev_acc = acc
    print("Model finished training")


if __name__ == '__main__':
    main()


'''
D:\Anaconda3\envs\mlai\python.exe D:/SUSTech/2023Spring·S2/ML/hw4/train.py
Epoch 0/100: Loss=1.121254, Accuracy=0.714300
Epoch 1/100: Loss=0.534921, Accuracy=0.867000
Epoch 2/100: Loss=0.363975, Accuracy=0.906600
Epoch 3/100: Loss=0.282882, Accuracy=0.924400
Epoch 4/100: Loss=0.243548, Accuracy=0.936900
Epoch 5/100: Loss=0.223768, Accuracy=0.945600
Epoch 6/100: Loss=0.211894, Accuracy=0.950500
Epoch 7/100: Loss=0.202599, Accuracy=0.954900
Epoch 8/100: Loss=0.194752, Accuracy=0.959800
Epoch 9/100: Loss=0.188196, Accuracy=0.964100
Epoch 10/100: Loss=0.182561, Accuracy=0.967200
Epoch 11/100: Loss=0.177729, Accuracy=0.970100
Epoch 12/100: Loss=0.173792, Accuracy=0.972300
Epoch 13/100: Loss=0.170785, Accuracy=0.974000
Epoch 14/100: Loss=0.168585, Accuracy=0.974700
Epoch 15/100: Loss=0.166974, Accuracy=0.975700
Epoch 16/100: Loss=0.165758, Accuracy=0.976900
Epoch 17/100: Loss=0.164783, Accuracy=0.977200
Epoch 18/100: Loss=0.163923, Accuracy=0.977600
Epoch 19/100: Loss=0.163085, Accuracy=0.977800
Epoch 20/100: Loss=0.162204, Accuracy=0.977600
Epoch 21/100: Loss=0.161240, Accuracy=0.977700
Epoch 22/100: Loss=0.160168, Accuracy=0.978000
Epoch 23/100: Loss=0.158961, Accuracy=0.978300
Epoch 24/100: Loss=0.157595, Accuracy=0.978800
Epoch 25/100: Loss=0.156040, Accuracy=0.979100
Epoch 26/100: Loss=0.154273, Accuracy=0.979100
Epoch 27/100: Loss=0.152280, Accuracy=0.979300
Epoch 28/100: Loss=0.150063, Accuracy=0.979400
Epoch 29/100: Loss=0.147646, Accuracy=0.979700
Epoch 30/100: Loss=0.145072, Accuracy=0.979900
Epoch 31/100: Loss=0.142400, Accuracy=0.980600
Epoch 32/100: Loss=0.139696, Accuracy=0.980900
Epoch 33/100: Loss=0.137009, Accuracy=0.981200
Epoch 34/100: Loss=0.134360, Accuracy=0.981600
Epoch 35/100: Loss=0.131741, Accuracy=0.981800
Epoch 36/100: Loss=0.129124, Accuracy=0.982700
Epoch 37/100: Loss=0.126476, Accuracy=0.983000
Epoch 38/100: Loss=0.123771, Accuracy=0.983900
Epoch 39/100: Loss=0.121004, Accuracy=0.984300
Epoch 40/100: Loss=0.118180, Accuracy=0.984500
Epoch 41/100: Loss=0.115310, Accuracy=0.984600
Epoch 42/100: Loss=0.112394, Accuracy=0.984600
Epoch 43/100: Loss=0.109427, Accuracy=0.984400
Epoch 44/100: Loss=0.106391, Accuracy=0.984600
Epoch 45/100: Loss=0.103253, Accuracy=0.984800
Epoch 46/100: Loss=0.099974, Accuracy=0.984800
Epoch 47/100: Loss=0.096504, Accuracy=0.984800
Epoch 48/100: Loss=0.092796, Accuracy=0.985400
Epoch 49/100: Loss=0.088819, Accuracy=0.985800
Epoch 50/100: Loss=0.084570, Accuracy=0.986000
Epoch 51/100: Loss=0.080080, Accuracy=0.986000
Epoch 52/100: Loss=0.075421, Accuracy=0.985900
Epoch 53/100: Loss=0.070780, Accuracy=0.986200
Epoch 54/100: Loss=0.066529, Accuracy=0.986800
Epoch 55/100: Loss=0.063027, Accuracy=0.986900
Epoch 56/100: Loss=0.060407, Accuracy=0.986800
Epoch 57/100: Loss=0.058669, Accuracy=0.987300
Epoch 58/100: Loss=0.057490, Accuracy=0.987100
Epoch 59/100: Loss=0.056235, Accuracy=0.987300
Epoch 60/100: Loss=0.054430, Accuracy=0.987100
Epoch 61/100: Loss=0.051928, Accuracy=0.987100
Epoch 62/100: Loss=0.048793, Accuracy=0.987000
Epoch 63/100: Loss=0.045188, Accuracy=0.987200
Epoch 64/100: Loss=0.041316, Accuracy=0.987500
Epoch 65/100: Loss=0.037361, Accuracy=0.987500
Epoch 66/100: Loss=0.033383, Accuracy=0.987800
Epoch 67/100: Loss=0.029439, Accuracy=0.987800
Epoch 68/100: Loss=0.025671, Accuracy=0.987800
Epoch 69/100: Loss=0.022152, Accuracy=0.987700
Epoch 70/100: Loss=0.018640, Accuracy=0.987500
Epoch 71/100: Loss=0.015239, Accuracy=0.987400
Epoch 72/100: Loss=0.012617, Accuracy=0.987500
Epoch 73/100: Loss=0.011005, Accuracy=0.987300
Epoch 74/100: Loss=0.010051, Accuracy=0.987300
Epoch 75/100: Loss=0.009467, Accuracy=0.987600
Epoch 76/100: Loss=0.008409, Accuracy=0.987400
Epoch 77/100: Loss=0.009198, Accuracy=0.987400
Epoch 78/100: Loss=0.007047, Accuracy=0.986900
Epoch 79/100: Loss=0.004253, Accuracy=0.986700
Epoch 80/100: Loss=0.006757, Accuracy=0.987400
Epoch 81/100: Loss=0.008071, Accuracy=0.984800
Epoch 82/100: Loss=0.007426, Accuracy=0.987200
Epoch 83/100: Loss=0.012092, Accuracy=0.986400
Epoch 84/100: Loss=0.004271, Accuracy=0.986900
Epoch 85/100: Loss=0.008373, Accuracy=0.987600
Epoch 86/100: Loss=0.005055, Accuracy=0.987900
Epoch 87/100: Loss=0.010343, Accuracy=0.987600
Epoch 88/100: Loss=0.005408, Accuracy=0.987300
Epoch 89/100: Loss=0.002541, Accuracy=0.987900
Epoch 90/100: Loss=0.003596, Accuracy=0.988300
Epoch 91/100: Loss=0.002441, Accuracy=0.987700
Epoch 92/100: Loss=0.004149, Accuracy=0.987400
Epoch 93/100: Loss=0.001839, Accuracy=0.987500
Epoch 94/100: Loss=0.003418, Accuracy=0.988000
Epoch 95/100: Loss=0.002449, Accuracy=0.987200
Epoch 96/100: Loss=0.003165, Accuracy=0.987800
Epoch 97/100: Loss=0.019424, Accuracy=0.987600
Epoch 98/100: Loss=0.002952, Accuracy=0.983700
Epoch 99/100: Loss=0.001626, Accuracy=0.984500
Model finished training

Process finished with exit code 0

'''