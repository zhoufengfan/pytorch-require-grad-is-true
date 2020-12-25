import torch
import torch.nn as nn


class Network5(nn.Module):
    def __init__(self, input_dim=20, output_dim=7):
        super(Network5, self).__init__()
        self.dis = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.dis(x)
        return x


if __name__ == '__main__':
    epoch = 1000
    # It's better not to use `reshape` to change the shape of the tensor. Because it'll will change the tensor to
    # non-leaf.
    t1 = torch.tensor([[0.0, 1.0, 2.0]], requires_grad=True)
    t2 = torch.tensor([[1.0, 1.0, 1.0]])
    l3 = [t1]
    optimizer = torch.optim.Adam(l3, lr=0.0001)
    criterion = nn.KLDivLoss(reduction="batchmean")
    for i in range(epoch):
        loss = criterion(t1, t2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("t1 is{}".format(t1))
    print("t2 is{}".format(t2))
