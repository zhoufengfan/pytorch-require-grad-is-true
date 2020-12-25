import torch
import torch.nn as nn

if __name__ == '__main__':
    epoch = 5000
    # It's better not to use `reshape` to change the shape of the tensor. Because it'll will change the tensor to
    # non-leaf.
    t1 = torch.tensor([[20.0, 1.0, 30.0]], requires_grad=True)
    t2 = torch.tensor([[1.0, 1.0, 1.0]])
    l3 = [t1]
    optimizer = torch.optim.Adam(l3, lr=0.01)
    criterion = nn.KLDivLoss(reduction="batchmean")
    for i in range(epoch):
        loss = criterion(torch.log(t2), t1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("t1 is{}".format(t1))
    print("t2 is{}".format(t2))
