import torch

device = torch.device('cuda:0')

x_data = [[0.05, 0.1]]
y_data = [[0.01, 0.99]]
w1 = [[0.15, 0.25], [0.2, 0.3]]
w2 = [[0.4, 0.5], [0.45, 0.55]]
b1 = 0.35
b2 = 0.6


x = torch.tensor(x_data, device=device)
y = torch.tensor(y_data, device=device)
w1 = torch.tensor(w1, requires_grad=True, device=device)
w2 = torch.tensor(w2, requires_grad=True, device=device)

learning_rate = 0.5

for t in range(1):
    print("Epoch： ", t + 1)
    H = (x.mm(w1) + b1).sigmoid()
    y_pred = (H.mm(w2) + b2).sigmoid()
    loss = (y_pred - y).pow(2).sum()/2
    print("loss: ", loss.data)
    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
        print("updated weights on input layer: ", w1.data)
        print("updated weights on hidden layer: ", w2.data)

