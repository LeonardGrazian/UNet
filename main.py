
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
torch.manual_seed(94)
from torch import nn

from data_utils import get_dataloaders, load_model_state
from modules import UNet


# constants
from config import MODEL_FILENAME, EPOCHS, LR, INSPECT, N_INSPECT_IMAGES
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, Y) in enumerate(train_dataloader):
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        Y_ = model(X)
        loss = loss_fn(Y_, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 20 == 0:
            print('Working on batch {} (loss={:.4})...'.format(batch, loss))


def test(test_dataloader, model, loss_fn):
    model.eval()

    total_batches = len(test_dataloader)
    total_loss = 0
    with torch.no_grad():
        for X, Y in test_dataloader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)

            Y_ = model(X)
            loss = loss_fn(Y_, Y)
            total_loss += loss

    print('Evaluation loss={:.4}'.format(total_loss / total_batches))


def inspect_output(test_dataloader, model):
    softmax = nn.Softmax(dim=1)
    model.eval()
    inspected_images = 0
    with torch.no_grad():
        for X, Y in test_dataloader:
            Y_ = softmax(model(X))

            for x, y, y_ in zip(X, Y, Y_):
                x  = np.moveaxis(x.numpy(), 0, 2)
                y  = np.moveaxis(y.numpy(), 0, 2)
                y_ = np.moveaxis(y_.numpy(), 0, 2)

                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

                ax1.set_title('Image')
                ax1.imshow(x)
                ax2.set_title('True Segmentation')
                ax2.imshow(y)
                ax3.set_title('Pred Segmentation')
                ax3.imshow(y_)
                plt.show()

                inspected_images += 1
                if inspected_images >= N_INSPECT_IMAGES:
                    break
            else:
                continue # proceed if "break" is not executed in inner loop
            break # end inspection if "break" is executed in inner loop


def create_model():
    train_dataloader, val_dataloader, _ = get_dataloaders()

    model = UNet(
        [64, 128, 256, 512],
        1024,
        [512, 256, 128, 64]
    ).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(EPOCHS):
        print('Working on epoch {}...'.format(ep))
        train(train_dataloader, model, loss_fn, optimizer)
        test(val_dataloader, model, loss_fn)
        print()
    print('Done.')

    return model


def main():
    model_file = Path(MODEL_FILENAME)
    if model_file.exists():
        print('Using saved model at {}...'.format(MODEL_FILENAME))
        model = UNet(
            [64, 128, 256, 512],
            1024,
            [512, 256, 128, 64]
        ).to(DEVICE)
        load_model_state(model, MODEL_FILENAME, DEVICE)
    else:
        print('Creating new model...')
        model = create_model()
        torch.save(model.state_dict(), MODEL_FILENAME)

    print('Evaluating model on test set...')
    _, _, test_dataloader = get_dataloaders()
    loss_fn = nn.CrossEntropyLoss()
    test(test_dataloader, model, loss_fn)

    if INSPECT:
        print('Inspecting model outputs...')
        inspect_output(test_dataloader, model)


if __name__ == '__main__':
    main()
