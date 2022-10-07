from model import Transformer
import torch
import logging
from plot import *
from helpers import *
from joblib import load
import math, random


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)


def flip_from_probability(p):
    return True if random.random() < p else False


def transformer(dataloader, EPOCH, k, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):
    device = torch.device(device)
    print("---device---", device)
    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    best_model = ""
    min_train_loss = float('inf')

    for epoch in range(EPOCH + 1):
        train_loss = 0

        model.train()
        for index_in, index_tar, _input, target, sensor_number in dataloader:
            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            optimizer.zero_grad()
            src = _input.permute(1, 0, 2).double().to(device)[:-1, :, :]    # torch.Size([24, 1, 7])
            target = _input.permute(1, 0, 2).double().to(device)[1:, :, :]  # src shifted by 1.
            # choose which value: real or prediction
            sampled_src = src[:1, :, :]  # t0 torch.Size([1, 1, 7])

            for i in range(len(target) - 1):
                # len(target) - 1 == 46

                prediction = model(sampled_src, device)  # torch.Size([1xw, 1, 1])

                """
                # to update model at every step
                # loss = criterion(prediction, target[:i+1,:,:1])
                # loss.backward()
                # optimizer.step()
                """

                # One day, enough data to make inferences about cycles
                if i < 24:
                    prob_true_val = True
                else:
                    # coin flip
                    # probability of heads/tails depends on the epoch, evolves with time.
                    v = k / (k + math.exp(epoch / k))
                    # starts with over 95 % probability of true val for each flip in epoch 0.
                    prob_true_val = flip_from_probability(v)

                # Using true value as next value
                if prob_true_val:
                    sampled_src = torch.cat((sampled_src.detach(), src[i + 1, :, :].unsqueeze(0).detach()))
                # using prediction as new value
                else:
                    # only using the humidity of prediction, the position is true value
                    positional_encodings_new_val = src[i + 1, :, 1:].unsqueeze(0)
                    predicted_humidity = torch.cat((prediction[-1, :, :].unsqueeze(0), positional_encodings_new_val), dim=2)
                    # sampled_src shape: torch.Size([1, 1, 7])
                    sampled_src = torch.cat((sampled_src.detach(), predicted_humidity.detach()))

            """To update model after each sequence"""
            # prediction shape: torch.Size([46, 1, 1])
            loss = criterion(target[:-1, :, 0].unsqueeze(-1), prediction)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach().item()

        if train_loss < min_train_loss:
            if epoch % 20 == 0:
                torch.save(model.state_dict(), path_to_save_model + f"best_train_{epoch}.pth")
                torch.save(optimizer.state_dict(), path_to_save_model + f"optimizer_{epoch}.pth")
                best_model = f"best_train_{epoch}.pth"
            min_train_loss = train_loss

        if epoch % 20 == 0:  # Plot 20-Step Predictions
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            scaler = load('scalar_item.joblib')
            sampled_src_humidity = scaler.inverse_transform(sampled_src[:, :, 0].cpu())  # torch.Size([47, 1, 7])
            src_humidity = scaler.inverse_transform(src[:, :, 0].cpu())  # torch.Size([47, 1, 7])
            prediction_humidity = scaler.inverse_transform(prediction[:, :, 0].detach().cpu().numpy())  # torch.Size([47, 1, 7])
            plot_training_3(epoch, path_to_save_predictions, src_humidity, sampled_src_humidity, prediction_humidity, sensor_number, index_in, index_tar)

        train_loss /= len(dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)

    plot_loss(path_to_save_loss, train=True)
    return best_model
