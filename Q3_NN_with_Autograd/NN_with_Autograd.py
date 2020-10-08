import torch

# set device using first gpu to do calculation
device = torch.device('cuda:0')

# initialize the input, weights, bias, output, and learning rate
learning_rate = 0.5

x = torch.tensor([[0.05, 0.1]], device=device)
y = torch.tensor([[0.01, 0.99]], device=device)
w1 = torch.tensor([[0.15, 0.25], [0.2, 0.3]], requires_grad=True, device=device)
w2 = torch.tensor([[0.4, 0.5], [0.45, 0.55]], requires_grad=True, device=device)
b1 = torch.tensor(0.35, requires_grad=True, device=device)
b2 = torch.tensor(0.6, requires_grad=True, device=device)

# in this task, we only try with one Epoch update
for t in range(1):
    print("Epoch： ", t + 1)
    # forward propagation with two layers
    H = (x.mm(w1) + b1).sigmoid()
    y_pred = (H.mm(w2) + b2).sigmoid()
    loss = (y_pred - y).pow(2).sum()/2
    print("loss: ", loss.data)

    # compute gradient of loss with respect to w1 and w2
    loss.backward()

    # do not need to build computational graph for this
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        b1 -= learning_rate * b1.grad
        b2 -= learning_rate * b2.grad

        # make sure reset gradient to zero
        w1.grad.zero_()
        w2.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()

        print("updated weights on input layer: ", w1.data)
        print("updated weights on hidden layer: ", w2.data)
        print("updated bias on input layer: ", b1.data)
        print("updated bias on hidden layer: ", b2.data)

