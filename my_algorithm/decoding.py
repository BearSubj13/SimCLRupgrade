import torch
import torch.nn as nn
import copy
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_channels, dec_channels, latent_size):
        #nn.Module.__init__(self)
        super().__init__()

        self.in_channels = in_channels
        self.dec_channels = dec_channels
        self.latent_size = latent_size
        self.input_image_size = 64

        ###############
        # DECODER
        ##############

        self.d_fc_1 = nn.Linear(latent_size, dec_channels * 16 * 2 * 2)
        self.d_bn_0 = nn.BatchNorm1d(dec_channels * 16 * 2 * 2)
        self.d_fc_2 = nn.Linear(dec_channels * 16 * 2 * 2, dec_channels * 16 * 2 * 2)

        self.d_conv_1 = nn.Conv2d(dec_channels * 16, dec_channels * 8,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_1 = nn.BatchNorm2d(dec_channels * 8)

        self.d_conv_2 = nn.Conv2d(dec_channels * 8, dec_channels * 4,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_2 = nn.BatchNorm2d(dec_channels * 4)

        self.d_conv_3 = nn.Conv2d(dec_channels * 4, dec_channels * 2,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_3 = nn.BatchNorm2d(dec_channels * 2)

        self.d_conv_4 = nn.Conv2d(dec_channels * 2, dec_channels,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)
        self.d_bn_4 = nn.BatchNorm2d(dec_channels)

        self.d_conv_5 = nn.Conv2d(dec_channels, in_channels,
                                  kernel_size=(4, 4), stride=(1, 1), padding=0)

        # Reinitialize weights using He initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()
            elif isinstance(m, torch.nn.Linear):
                nn.init.kaiming_normal_(m.weight.detach())
                m.bias.detach().zero_()


    def decode(self, x):

        # h1
        # x = x.view(-1, self.latent_size, 1, 1)
        x = self.d_fc_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_0(x)

        x = self.d_fc_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = x.view(-1, self.dec_channels * 16, 2, 2)

        # h2
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_1(x)

        # h3
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_2(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_2(x)

        # h4
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_3(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_3(x)

        # h5
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_4(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
        x = self.d_bn_4(x)

        # out
        x = F.interpolate(x, scale_factor=2)
        x = F.pad(x, pad=(2, 1, 2, 1), mode='replicate')
        x = self.d_conv_5(x)
        x = torch.sigmoid(x)

        return x


    def forward(self, z):
        decoded = self.decode(z)
        return decoded


def return_loss_function(encoder):
    def loss_f(x_est, x_gt):
        _, _, latent_estim, _ = encoder(x_est, x_est)
        _, _, latent_gt, _ = encoder(x_gt, x_gt)
        loss = 1 - nn.functional.cosine_similarity(latent_estim, latent_gt)
        loss = loss.mean()
        return loss

    return loss_f


def decoder_step_loss_func(encoder, decoder, batch1, batch2, optimizer):
    encoder_copy = copy.deepcopy(encoder)
    encoder_copy.eval()
    for parameter in encoder_copy.parameters():
        parameter.requires_grad = False
    criterion = return_loss_function(encoder_copy)

    loss, image_decoded1, image_decoded2, z1, z2 = decoder_step(encoder, decoder, batch1, batch2, optimizer, criterion)
    return loss


def decoder_step(encoder, decoder, batch1, batch2, optimizer_decoder, criterion, freeze_encoder=True, optimizer_encoder=None):
    assert freeze_encoder or optimizer_encoder is not None

    decoder.train()
    optimizer_decoder.zero_grad()

    if freeze_encoder:
        encoder.eval()
        with torch.no_grad():
            _, _, z1, z2 = encoder(batch1, batch2)
    else:
        optimizer_encoder.zero_grad()
        encoder.train()
        _, _, z1, z2 = encoder(batch1, batch2)


    z1 = z1.detach()
    z2 = z2.detach()
    image_decoded1 = decoder(z1)
    image_decoded2 = decoder(z2)

    loss1 = criterion(image_decoded1, batch1)
    loss2 = criterion(image_decoded2, batch2)
    (loss1+loss2).backward()
    optimizer_decoder.step()
    if not freeze_encoder:
        optimizer_encoder.step()
    loss = (loss1.item() + loss2.item()) / 2
    return loss, image_decoded1, image_decoded2, z1, z2


def train_autoencoder(encoder, decoder, data_loader, optimizer_encoder, optimizer_decoder, freeze_encoder=True):
    loss_sum = 0
    bce_sum = 0

    encoder_copy = copy.deepcopy(encoder)
    encoder_copy.eval()
    for parameter in encoder_copy.parameters():
        parameter.requires_grad = False
    criterion = return_loss_function(encoder_copy)

    for step, ((x_i, x_j), _) in enumerate(data_loader):
        x_i = x_i.cuda(non_blocking=True)
        x_j = x_j.cuda(non_blocking=True)

        loss, image_decoded1, image_decoded2, _, _ = decoder_step(encoder, decoder, x_i, x_j, \
                             optimizer_decoder, criterion, freeze_encoder, optimizer_encoder)
        bce_loss = nn.MSELoss()
        image_decoded1 = image_decoded1.detach()
        image_decoded2 = image_decoded2.detach()
        bce = (bce_loss(image_decoded1, x_i.detach()) + bce_loss(image_decoded2, x_j.detach()))/2
        bce_sum += bce.item()
        loss_sum += loss
    bce_mean = bce_sum / len(data_loader)
    loss_mean = loss_sum / len(data_loader)
    return loss_mean, bce_mean
