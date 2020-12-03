import os
import numpy as np
import torch
import torchvision
import argparse

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

from model import load_optimizer, save_model
from modules import SimCLR, NT_Xent, get_resnet
from modules.transformations import TransformsSimCLR
from modules.sync_batchnorm import convert_model
from utils import yaml_config_hook

from my_algorithm.decoding import decoder_step_loss_func, Decoder, train_autoencoder
from my_algorithm.metric_learning import penalty_for_fake, penalty_for_random


def train(args, train_loader, model, decoder, criterion, optimizer, \
          optimizer_decoder, writer, random_fake=False, scatter_radius=1.0):
    loss_epoch = 0
    loss_epoch_decoder = 0
    penalty_epoch = 0
    for step, ((x_i, x_j), _) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        loss_decoder = decoder_step_loss_func(model, decoder, x_i, x_j, optimizer_decoder)

        model.train()
        # positive pair, with encoding
        h_i, h_j, z_i, z_j = model(x_i, x_j)

        loss = criterion(z_i, z_j)

        coeff_lambda = args.random_coefficient
        random_fake = True# step % 2 == 0
        if not random_fake:
            penalty = penalty_for_fake(model, decoder, x_i, x_j, z_i, z_j, scatter_radius=scatter_radius)
        else:
            penalty = penalty_for_random(model, decoder, x_i, x_j)

        (loss + coeff_lambda * penalty).backward()
        optimizer.step()

        penalty_epoch += penalty.item()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        if args.nr == 0 and step % 50 == 0:
            print(f"Step [{step}/{len(train_loader)}]\t Loss: {loss.item()}")

        if args.nr == 0:
            writer.add_scalar("Loss/train_epoch", loss.item(), args.global_step)
            args.global_step += 1

        loss_epoch += loss.item()
        loss_epoch_decoder += loss_decoder
    return loss_epoch, loss_epoch_decoder, penalty_epoch


def main(gpu, args):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size),
        )
    else:
        raise NotImplementedError

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )

    # initialize ResNet
    encoder = get_resnet(args.resnet, pretrained=False)
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    # initialize model
    model = SimCLR(args, encoder, n_features)
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.epoch_num)
        )
        print(model_fp)
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler = load_optimizer(args, model)
    criterion = NT_Xent(args.batch_size, args.temperature, args.device, args.world_size)

    # DDP / DP
    if args.dataparallel:
        model = convert_model(model)
        model = DataParallel(model)
    else:
        if args.nodes > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)

    writer = None
    if args.nr == 0:
        writer = SummaryWriter()

    #added by @IvanKruzhilov
    decoder = Decoder(3, 3, args.image_size)
    optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=0.001)
    #decoder.load_state_dict(torch.load('save/decoder_my_algorithm_augmented.pt'))
    decoder = decoder.to(args.device)


    args.global_step = 0
    args.current_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        scatter_radius = 0.2
        random_fake = None#set in train fucntion now

        loss_epoch, loss_epoch_decoder, penalty_epoch = \
            train(args, train_loader, model, decoder, criterion, optimizer, \
            optimizer_decoder, writer, random_fake, scatter_radius)

        loss_mean, bce_mean = train_autoencoder(model, decoder, train_loader, None, \
                                                optimizer_decoder, freeze_encoder=True)

        if args.nr == 0 and scheduler:
            scheduler.step()

        if args.nr == 0 and epoch % 5 == 0:
            save_model(args, model, optimizer)
            torch.save(decoder.state_dict(), os.path.join(args.model_path,'decoder{0}.pt'.format(epoch)))

        if epoch % 10 == 0:
            decoder = Decoder(3, 3, args.image_size)
            optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=0.001)
            decoder = decoder.to(args.device)

        if args.nr == 0:
            writer.add_scalar("Loss/train", loss_epoch / len(train_loader), epoch)
            writer.add_scalar("Misc/learning_rate", lr, epoch)
            mean_loss = loss_epoch / len(train_loader)
            mean_loss_decoder = loss_epoch_decoder / len(train_loader)
            mean_penalty = penalty_epoch / len(train_loader)
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {mean_loss}\t decoder loss: {mean_loss_decoder}\t \
                penalty: {mean_penalty}\t lr: {round(lr, 5)}"
            )
            print('loss: ',loss_mean, 'mse: ', bce_mean)
            args.current_epoch += 1

    ## end training
    save_model(args, model, optimizer)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SimCLR")
    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8000"

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes

    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
