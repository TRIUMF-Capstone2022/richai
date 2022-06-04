
import time
import os
import torch
import pandas as pd
import torch.nn.functional as F
from utils.helpers import get_config, get_logger
from models.pointnet import PointNetFc
from dataset.rich_dataset import RICHDataset
from dataset.data_loader import data_loader

logger = get_logger()


def trainer(
    model,
    optimizer,
    train_loader,
    val_loader=None,
    criterion=None,
    epochs=5,
    scheduler=None,
    device="cuda",
    results=False,
    operating_point=0.5,
    show_results=2500,
):
    """Trainer for pointnet"""

    logger.info(f"Starting training...")
    training_start = time.time()

    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []

    model.to(device)

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}...")
        epoch_start = time.time()

        running_loss = 0.0
        running_total, running_correct = 0, 0
        total_pions, total_muons = 0, 0
        true_pion_preds, false_neg_muons = 0, 0

        # --------------------------------------- TRAINING ---------------------------------------
        logger.info(f"Starting training phase of epoch {epoch+1}...")
        model.train()

        for i, (X, y, p, r) in enumerate(train_loader, 0):
            # X needs to be reshaped for knn calculation to work
            # (batch_size, PMTs, points) -> (batch_size, points, PMTs)
            X = X.float().to(device)

            # label, momentum, radius are OK with shape (1, batch_size)
            y = y.long().to(device)
            p = p.float().to(device)
            r = r.float().to(device)

            # zero parameter gradients
            optimizer.zero_grad()

            # forward pass and calculate loss
            logits = model(X, p, r).flatten()
            loss = criterion(logits, y.type(torch.float32))
            running_loss += loss.item()

            # calculate predictions and accuracy
            preds = torch.sigmoid(logits) > operating_point
            running_total += X.size(0)
            running_correct += preds.eq(y).sum().item()
            acc = running_correct / running_total

            # total pions and muons this batch
            total_pions += torch.sum(y == 1).item()
            total_muons += torch.sum(y == 0).item()

            # true positive pions and false negative muons
            true_pion_preds += torch.sum((preds == 1) & (y == 1)).item()
            false_neg_muons += torch.sum((preds == 0) & (y == 1)).item()

            # pion efficiency and muon misclassification as pion
            pion_efficiency = true_pion_preds / total_pions
            muon_misclass = false_neg_muons / total_muons

            # back prop and update weights
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            # log results every 100 batches or upon final batch
            if (i + 1) % show_results == 0 or i == len(train_loader) - 1:
                outstr = (
                    "(T)E: %d (B: %4d/%4d), Loss: %.4f, Acc: %.4f, Pion eff: %.4f, Muon misclass: %.4f"
                    % (
                        epoch + 1,
                        i + 1,
                        len(train_loader),
                        running_loss / running_total,
                        acc,
                        pion_efficiency,
                        muon_misclass,
                    )
                )

                logger.info(outstr)

        train_losses.append(running_loss / len(train_loader))
        train_accs.append(acc)

        logger.info(f"Completed training phase of epoch {epoch+1}!")

        # --------------------------------------- VALIDATION -------------------------------------
        if val_loader:
            logger.info(f"Starting validation phase of epoch {epoch+1}...")
            model.eval()

            running_loss = 0.0
            running_total, running_correct = 0, 0
            total_pions, total_muons = 0, 0
            true_pion_preds, false_neg_muons = 0, 0

            with torch.no_grad():
                for i, (X, y, p, r) in enumerate(val_loader, 0):
                    X = X.float().to(device)
                    y = y.long().to(device)
                    p = p.float().to(device)
                    r = r.float().to(device)

                    logits = model(X, p, r).flatten()
                    loss = criterion(logits, y.type(torch.float32))
                    running_loss += loss.item()

                    # preds = logits.max(dim=1)[1]
                    preds = torch.sigmoid(logits) > operating_point
                    running_total += X.size(0)
                    running_correct += preds.eq(y).sum().item()
                    acc = running_correct / running_total

                    # total pions and muons this batch
                    total_pions += torch.sum(y == 1).item()
                    total_muons += torch.sum(y == 0).item()

                    # true positive pions and false negative muons
                    true_pion_preds += torch.sum((preds == 1) & (y == 1)).item()
                    false_neg_muons += torch.sum((preds == 0) & (y == 1)).item()

                    # pion efficiency and muon misclassification as pion
                    pion_efficiency = true_pion_preds / total_pions
                    muon_misclass = false_neg_muons / total_muons

                    # log results every 100 batches or upon final batch
                    if (i + 1) % show_results == 0 or i == len(val_loader) - 1:
                        outstr = (
                            "(V)E: %d (B: %4d/%4d), Loss: %.4f, Acc: %.4f, Pion eff: %.4f, Muon misclass: %.4f"
                            % (
                                epoch + 1,
                                i + 1,
                                len(val_loader),
                                running_loss / running_total,
                                acc,
                                pion_efficiency,
                                muon_misclass,
                            )
                        )

            valid_losses.append(running_loss / len(val_loader))
            valid_accs.append(acc)

            logger.info(outstr)

        logger.info(f"Completed validation phase of epoch {epoch+1}!")

        # log epoch elapsed time
        epoch_time_elapsed = time.time() - epoch_start

        outstr = ("Epoch %d/%d completed in %.0fm %.0fs") % (
            epoch + 1,
            epochs,
            epoch_time_elapsed // 60,
            epoch_time_elapsed % 60,
        )

        logger.info(outstr)

    # log total elapsed training time
    total_time_elapsed = time.time() - training_start
    outstr = "Training completed in {:.0f}m {:.0f}s".format(
        total_time_elapsed // 60, total_time_elapsed % 60
    )

    logger.info(outstr)

    if results:
        data = {
            "train_loss": train_losses,
            "train_acc": train_accs,
            "val_loss": valid_losses,
            "val_acc": valid_accs,
        }
        results = pd.DataFrame(data)

        results.to_csv("pointnet_training_results.csv")


def train_combined(reload_model=True):
    """Combined training for all the files"""
    

    # model parameters
    num_classes = get_config("model.pointnet.num_classes")
    model_path = get_config("model.pointnet.saved_model")
    epochs = get_config("model.pointnet.epochs")

    # dataset parameters
    dset_path = get_config("dataset.pointnet.dataset")
    sample_file = get_config("dataset.pointnet.sample_file")
    val_split = get_config("dataset.pointnet.val")
    test_split = get_config("dataset.pointnet.test")
    seed = get_config("seed")
    delta = get_config("model.pointnet.delta")
    
    # log training information
    logger.info(
        f"""
    TRAINING SESSION INFORMATION
    Main dataset: {dset_path}
    Sample file: {sample_file}
    Model save path: {model_path}
    Total epochs: {epochs}
    Train/val/test: {1-val_split-test_split:.2f}/{val_split}/{test_split}
    Seed: {seed}
    Time delta: {delta}
    Final output layer nodes: {num_classes}
    """
    )

    model = PointNetFc(
        num_classes=get_config('model.pointnet.num_classes'),
        momentum=True, radius=True
    )

    # enable multi GPUs
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=get_config('gpu'))
        device = f'cuda:{model.device_ids[0]}'
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # get the dataset, for now this is C since it's been updated
    dataset = RICHDataset(
        dset_path=dset_path,
        sample_file=sample_file,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
        delta=delta,
        # data_augmentation=True
    )

    # get the data loaders
    train_loader, val_loader, _ = data_loader(dataset)
    
    if reload_model and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        logger.info(
            f"model reloaded from existing path {model_path}, continue training"
        )


    # train the model
    trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        epochs=epochs,
        device=device,
        show_results=2500,
    )

    torch.save(model.state_dict(), model_path)
    logger.info(f"Model successfully saved to {model_path}")


if __name__ == "__main__":
    train_combined()