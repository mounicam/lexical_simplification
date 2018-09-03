import torch
from torch.autograd import Variable


class NRR:
    def __init__(self, x, y, drop_out=0.0):
        self.train_x = Variable(torch.FloatTensor(x))
        self.train_y = Variable(torch.FloatTensor(y), requires_grad=False)
        d_in, h, d_out = len(x[0]), 8, 1

        self.model = torch.nn.Sequential(
                torch.nn.Linear(d_in, h),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=drop_out),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=drop_out),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=drop_out),
                torch.nn.Linear(h, h),
                torch.nn.Tanh(),
                torch.nn.Dropout(p=drop_out),
                torch.nn.Linear(h, d_out)
        )
        self.loss_fn = torch.nn.MSELoss()

    def train(self, epochs, lr):
        self.model.training = True
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            y_pred = self.model(self.train_x)
            loss = self.loss_fn(y_pred, self.train_y)
            if epoch % 20 == 0:
                loss_val = loss.data.cpu().numpy().tolist()[0]
                print("MSE loss after %d iterations: %f" % (epoch, loss_val))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def set_testing(self):
        self.model.eval()

    def predict(self, test_x):
        test_x = Variable(torch.FloatTensor(test_x))
        return self.model(test_x)
