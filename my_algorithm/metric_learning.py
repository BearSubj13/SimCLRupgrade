import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.transformations import TransformsSimCLR
import torchvision

def triplet_step(encoder, pos_batch1, pos_batch2, neg_batch1, neg_batch2, margin=0.1):
    encoder.eval()
    triplet_loss = torch.nn.TripletMarginLoss(margin=margin, p=2)
    neg_batch1 = neg_batch1.detach()
    neg_batch2 = neg_batch2.detach()

    _, _, positive1, positive2 = encoder(pos_batch1, pos_batch2)
    _, _, negative1, negative2 = encoder(neg_batch1, neg_batch2)

    loss111 = triplet_loss(positive1, positive2, negative1)
    loss211 = triplet_loss(positive2, positive1, negative1)
    loss122 = triplet_loss(positive1, positive2, negative2)
    loss212 = triplet_loss(positive2, positive1, negative2)

    # print(1-F.cosine_similarity(positive1, positive2))
    # # print(1 - F.cosine_similarity(positive1, negative1))
    # # print(1 - F.cosine_similarity(positive1, negative2))
    # # print(1 - F.cosine_similarity(positive2, negative1))
    # print(1 - F.cosine_similarity(positive2, negative2))
    # print(loss111.item(), loss211.item(), loss122.item(), loss212.item())
    # exit()

    loss = loss111 + loss211 + loss122 + loss212
    return loss


def image_locality(z, decoder, radius=1.0, min_proximity=0.2):
    decoder.eval()
    eps = radius*torch.randn_like(z)
    min_distance = radius*min_proximity
    if torch.any(torch.abs(eps) < min_distance):
        eps[torch.abs(eps) < min_distance] = min_distance*torch.sign(eps[torch.abs(eps) < min_distance])
    else:
        eps = torch.zeros_like(z)
    z = z + eps
    with torch.no_grad():
        decoded = decoder(z)
    return decoded


def penalty_for_fake(encoder, decoder, pos_batch1, pos_batch2, neg_z1, neg_z2, scatter_radius=1.0):
    neg_z1 = neg_z1.detach()
    neg_z2 = neg_z2.detach()
    neg_batch1 = image_locality(neg_z1, decoder, radius=scatter_radius, min_proximity=0.1)
    neg_batch2 = image_locality(neg_z2, decoder, radius=scatter_radius, min_proximity=0.1)
    penalty = triplet_step(encoder, pos_batch1, pos_batch2, neg_batch1, neg_batch2, margin=0.5)
    return penalty


def penalty_for_random(encoder, decoder, pos_batch1, pos_batch2):
    negative_z1 = torch.randn([pos_batch1.shape[0], decoder.latent_size], requires_grad=False)
    negative_z1 = negative_z1.to(pos_batch1.device)
    negative_z2 = torch.randn([pos_batch1.shape[0], decoder.latent_size], requires_grad=False)
    negative_z2 = negative_z2.to(pos_batch1.device)
    negative_z1 = F.normalize(negative_z1, p=2, dim=1)
    negative_z2 = F.normalize(negative_z2, p=2, dim=1)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        neg_batch1 = decoder(negative_z1)
        neg_batch2 = decoder(negative_z2)

        # color_jitter = torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        # train_transform = torch.nn.Sequential(
        #         torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
        #         torchvision.transforms.RandomApply([color_jitter], p=0.8),
        #         torchvision.transforms.RandomGrayscale(p=0.2),
        # )
        # neg_batch1 = train_transform(neg_batch1.cpu().clone())
        # neg_batch1 = neg_batch1.to(pos_batch1.device)

    penalty = triplet_step(encoder, pos_batch1, pos_batch2, neg_batch1, neg_batch2, margin=1.0)
    return penalty

