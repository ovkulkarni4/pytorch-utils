import torch

from sklearn.model_selection import train_test_split

def get_optimizer(model, optimizer_type: str, lr):
    if optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr=lr)

    if optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)

    raise ValueError(f"Unrecognized type={optimizer_type}")

def get_linear_data():
    # Create some data (same as notebook 01)
    weight = 0.7
    bias = 0.3
    start = 0
    end = 1
    step = 0.01

    # Create data
    X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
    y_regression = weight * X_regression + bias # linear regression formula

    # Check the data
    print(len(X_regression))
    X_regression[:5], y_regression[:5]

    return X_regression, y_regression

def split_to_tensor(X, y, test_size=0.2, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return (
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_train),
        torch.FloatTensor(y_test)
    )

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc
