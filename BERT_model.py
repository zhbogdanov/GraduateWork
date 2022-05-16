import torch
import numpy as np
import torch.nn as nn
from IPython import display
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import accuracy_score, f1_score
from transformers import AdamW, AutoTokenizer, AutoModel


device = 'cpu'


class Classifier(nn.Module):
    def __init__(self, n_outputs=2):
        super(Classifier, self).__init__()
        self.bert = AutoModel.from_pretrained("models/rubert-base-cased")
        # self.bert = AutoModel.from_pretrained("models/rubert-base-cased-sentence")
        # self.bert = AutoModel.from_pretrained("models/distilbert-base-uncased")
        self.nout = n_outputs
        self.config = self.bert.config
        self.fc = nn.Linear(self.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, n_outputs)
        self.drop = nn.Dropout(0.5)
        self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base")
        # self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-sentence")
        # self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/distilbert-base-uncased")
        self.tokenizer.model_max_length = 130

    def forward(self, *args, **kwargs):
        x = self.bert(**kwargs).pooler_output
        x = self.drop(x)
        x = self.fc(x)
        x = nn.ReLU()(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = self.bert(**kwargs)
        x = nn.functional.softmax(x.logits, dim=1)
        return x


def todevice(d):
    for k in d:
        d[k] = d[k].to(device)


def apply_model(model, features):
    tokenizer = model.tokenizer
    if type(features[0]) is str:
        input_data = tokenizer(features, padding=True, return_tensors='pt', truncation=True)
        other_features = None
    else:
        text_features = [t for t in features if type(t[0]) is str]
        other_features = [t.to(device) for t in features if type(t[0]) is not str]
        input_data = tokenizer(*text_features, padding=True, return_tensors='pt', truncation=True)
    todevice(input_data)
    output = model(other_features, **input_data)
    return output


def train_epoch(model, optimizer, scheduler, train_dl):
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_history = []
    for i, (texts, labels) in enumerate(train_dl):
        optimizer.zero_grad()
        output = apply_model(model, texts)
        loss = criterion(output, labels.to(device).view(-1))
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_history.append(loss.item())
    return loss_history


def predict(model, dl):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    y_true = []
    probs = []
    mean_loss = 0
    count = 0

    with torch.no_grad():
        for texts, labels in dl:
            output = apply_model(model, texts)
            probs.extend(nn.Softmax(dim=1)(output).detach().tolist())

            loss = criterion(output, labels.to(device).view(-1))
            mean_loss += loss.item()
            count += 1

            y_true.extend((labels.tolist()))

    mean_loss /= count
    return mean_loss, y_true, probs


def train(model, optimizer, scheduler, train_dl, epochs, test_dl=None):
    model.train()
    avg = 'binary' if model.nout <= 2 else 'macro'
    for epoch in range(epochs):
        epoch_history = train_epoch(model, optimizer, scheduler, train_dl)
        if test_dl is not None:
            test_loss, y_true, probs = predict(model, test_dl)
            y_pred = np.argmax(probs, axis=1)
            scores = {'F1': f1_score(y_true, y_pred, average=avg) * 100,
                      'acc': accuracy_score(y_true, y_pred) * 100}
            yield epoch_history, test_loss, scores, y_true, probs
        else:
            yield epoch_history


def plot_history(loss_history, validation, F1, acc, sizes, title="", clear_output=False):
    if clear_output:
        display.clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7))
    ax1.plot(np.arange(1, sizes[-1] + 1), loss_history, zorder=-1, label="Обучение")
    ax1.scatter(sizes, validation, marker='*', c='red', zorder=1, s=50, label='Валидация')
    ax1.grid()
    ax1.set_xticks([0] + sizes)
    ax1.set_xlabel('Итерация')
    ax1.set_ylabel('Кросс-энтропия')
    ax1.legend()
    ax2.scatter(np.arange(0, len(F1)), F1)
    ax2.scatter(np.arange(0, len(F1)), acc)
    print(acc[-1], F1[-1])
    ax2.plot(F1, label="F1")
    ax2.plot(acc, label="Accuracy")
    ax2.grid()
    ax2.set_yticks(np.linspace(min(F1), max(acc), 20))
    ax2.set_xticks(np.arange(0, len(sizes), 1))
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Метрика (%)')
    ax2.legend()
    fig.suptitle(title)
    plt.show()
    return fig


def train_model(model, optimizer, params, train_ds, test_ds):
    torch.manual_seed(7)
    BS = params['BS']
    EPOCHS = params['n_epochs']
    if not optimizer:
        optimizer = AdamW(model.parameters(), lr=params['lr'], weight_decay=params['w_decay'])

    scheduler = None if not params['shed'] \
        else OneCycleLR(optimizer, max_lr=params['max_lr'], steps_per_epoch=ceil(len(train_ds) / BS),
                        epochs=EPOCHS, anneal_strategy='linear')

    train_dl = DataLoader(train_ds, batch_size=BS, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BS, shuffle=False)

    loss_history = []
    validation_history = []
    sizes = []
    f1_history = []
    acc_history = []

    y_history = []
    epoch = 0
    best_f1 = 0
    for epoch_history, test_loss, test_scores, y_true, probs in train(model, optimizer, scheduler, train_dl, EPOCHS,
                                                                      test_dl):
        f1 = test_scores['F1']
        if f1 > best_f1:
            print(f'epoch #{epoch}:\treached better score ({f1:4.2f} vs {best_f1:4.2f})')
            best_f1 = f1
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
            }, f'models/tmp/dem_qa_sent_weights.pt')

        loss_history.extend(epoch_history)
        validation_history.append(test_loss)
        sizes.append(len(loss_history))
        f1_history.append(test_scores['F1'])
        acc_history.append(test_scores['acc'])
        y_history.append((y_true, probs))

        epoch += 1

    best_epoch_num = np.argmax(f1_history)

    return y_history[best_epoch_num]
