import os
from models.pointnet import PointNetFeat, PointNetFeedForward
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm


from dataset.rich_dataset import RICHDataset, combine_datset
from dataset.data_loader import data_loader
from utils.helpers import get_config, get_logger

logger = get_logger()


def trainer(
    feature_extractor,
    model,
    criterion,
    optimizer,
    scheduler,
    trainloader,
    validloader,
    epochs=5,
    device="cpu",
    verbose=True,
):
    """Simple training wrapper for PyTorch network."""

    train_loss, valid_loss, valid_accuracy = [], [], []
    for epoch in range(epochs):
        scheduler.step()

        # for each epoch
        train_batch_loss = 0
        valid_batch_loss = 0
        valid_batch_acc = 0

        # Training
        for X, y, p in trainloader:
            # GPU
            X = X.transpose(2, 1).to(device)
            y = y.long().to(device)
            p = p.to(device)

            optimizer.zero_grad()  # Zero all the gradients w.r.t. parameters
            X = feature_extractor(X).to(device)  # Forward pass to get output
            y_hat = model(X, p).to(device)
            loss = criterion(y_hat, y)  # Calculate loss based on output
            loss.backward()  # Calculate gradients w.r.t. parameters
            optimizer.step()  # Update parameters
            train_batch_loss += loss.item()  # Add loss for this batch to running total
        train_loss.append(train_batch_loss / len(trainloader))

        # Validation
        model.eval()
        with torch.no_grad():  # this stops pytorch doing computational graph stuff under-the-hood and saves memory and time
            for X, y, p in validloader:
                # GPU
                X = X.transpose(2, 1).to(device)
                y = y.long().to(device)
                p = p.to(device)

                X = feature_extractor(X).to(device)  # Forward pass to get output
                y_hat = model(X, p).to(device)
                _, y_hat_labels = torch.softmax(y_hat, dim=1).topk(1, dim=1)
                loss = criterion(y_hat, y)
                valid_batch_loss += loss.item()
                valid_batch_acc += (
                    (y_hat_labels.squeeze() == y).type(torch.float32).mean().item()
                )
        valid_loss.append(valid_batch_loss / len(validloader))
        valid_accuracy.append(valid_batch_acc / len(validloader))  # accuracy

        model.train()

        # Print progress
        if verbose:
            logger.info(
                f"""Epoch {epoch + 1}
                Train Loss: {train_loss[-1]:.3f}
                Valid Loss: {valid_loss[-1]:.3f}
                Valid Accuracy: {valid_accuracy[-1]:.2f}
                """
            )

    results = {
        "train_loss": train_loss,
        "valid_loss": valid_loss,
        "valid_accuracy": valid_accuracy,
    }
    return results


def train_combined(reload_model=True):
    """Train the model on combined dataset"""

    logger.info(f"Using gpus: {get_config('gpu')}")

    # define feature extractor
    feature_extractor = PointNetFeat(k=get_config("model.pointnet.num_classes"))

    # Fully connected model
    model = PointNetFeedForward(257, 3)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        feature_extractor = torch.nn.DataParallel(
            feature_extractor, device_ids=get_config("gpu")
        )
        model = torch.nn.DataParallel(model, device_ids=get_config("gpu"))
        device = f"cuda:{feature_extractor.device_ids[0]}"

    feature_extractor.to(device)
    model.to(device)

    criterion = F.nll_loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # model path
    model_path = get_config("model.pointnet.saved_model")

    logger.info(
        f"""
    Device: {device},
    model_path: {model_path}
    """
    )

    for file_, dataset in combine_datset("train", val_split=0.2).items():

        logger.info(f"Training for {file_}")

        trainloader, validloader, _ = data_loader(dataset)

        if reload_model and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            logger.info(
                f"model reloaded from existing path {model_path}, continue training"
            )

        # start training
        trainer(
            feature_extractor,
            model,
            criterion,
            optimizer,
            scheduler,
            trainloader,
            validloader,
            epochs=get_config("model.pointnet.epochs"),
            device=device,
            verbose=True,
        )

        logger.info(f"Training completed with file: {file_}")
        logger.info(f"Saving trained model to {model_path}")

        # save model
        torch.save(model.state_dict(), model_path)
        logger.info(f"model successfully saved to {model_path}")


if __name__ == "__main__":
    train_combined()
    # path = os.path.join(
    #     get_config("dataset.base_dir"),
    #     "A",
    #     "Run008548.EOSlist.CTRL.p.v2.0.4-01_f.v2.0.4-01.h5",
    # )
    # dataset = RICHDataset(dset_path=path, val_split=0.1, test_split=0.1)
    # num_classes = get_config("model.pointnet.num_classes")
    # batch_size = get_config("data_loader.batch_size")
    # model = PointNetFeat(k=num_classes).to(device)
    # criterion = F.nll_loss
    # optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # trainloader, validloader, testloader = data_loader(dataset)

    # # start training
    # trainer(
    #     model,
    #     criterion,
    #     optimizer,
    #     scheduler,
    #     trainloader,
    #     validloader,
    #     epochs=5,
    #     verbose=True,
    # )

    # # Save model
    # PATH = "pointnet.pt"
    # torch.save(model.state_dict(), PATH)  # save model at PATH

    # # Load model
    # model = PointNetFeat(k=num_classes)  # create an instance of the model
    # model.load_state_dict(torch.load(PATH))  # load model from PATH
