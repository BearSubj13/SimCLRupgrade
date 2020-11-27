import torch
import torch.nn as nn


def triplet_step(encoder, pos_batch1, pos_batch2, neg_batch1, neg_batch2):
    triplet_loss = torch.nn.TripletMarginLoss(margin=0.01, p=2)
    neg_batch1 = neg_batch1.detach()
    neg_batch2 = neg_batch2.detach()

    _, _, positive1, positive2 = encoder(pos_batch1, pos_batch2)
    _, _, negative1, negative2 = encoder(neg_batch1, neg_batch2)

    loss111 = triplet_loss(positive1, positive2, negative1)
    loss211 = triplet_loss(positive2, positive1, negative1)
    loss122 = triplet_loss(positive1, positive2, negative2)
    loss212 = triplet_loss(positive2, positive1, negative2)

    loss = loss111 + loss211 + loss122 + loss212
    return loss


def image_locality(z, decoder, radius=1.0, min_proximity=0.2):
    eps = radius*torch.randn_like(z)
    min_distance = radius*min_proximity
    eps[torch.abs(eps) < min_distance] = min_distance*torch.sign(eps[torch.abs(eps) < min_distance])
    z = z + eps
    with torch.no_grad():
        decoded = decoder(z)
    return decoded


def penalty_for_fake(encoder, decoder, pos_batch1, pos_batch2, neg_z1, neg_z2, scatter_radius=1.0):
    neg_z1 = neg_z1.detach()
    neg_z2 = neg_z2.detach()
    neg_batch1 = image_locality(neg_z1, decoder, radius=scatter_radius)
    neg_batch2 = image_locality(neg_z2, decoder, radius=scatter_radius)
    penalty = triplet_step(encoder, pos_batch1, pos_batch2, neg_batch1, neg_batch2)
    return penalty

