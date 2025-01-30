# For simple accuracy
def simple_accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# For multiclass accuracy
def multiclass_accuracy_fn(y_true, logits):
    y_pred = torch.argmax(logits, dim=1)
    return (y_true == y_pred).float().mean().item() * 100

# For pipeline with softmax
def pipeline_with_softmax_accuracy_fn(y_true, logits):
    y_pred = torch.softmax(logits, dim=1).argmax(dim=1)
    return (y_true == y_pred).float().mean().item() * 100

# device handling and dtype checking
def device_and_dtype_accuracy_fn(y_true, y_pred):
    # Move to same device and type
    y_true = y_true.to(y_pred.device)
    y_true = y_true.long()
    
    correct = (y_true == y_pred).float().sum()
    acc = correct / len(y_true) * 100
    return acc.item()
