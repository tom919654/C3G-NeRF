from im2scene.discriminator import conv

discriminator_dict = {
    'dc': conv.DCDiscriminator,
    'dc_cond': conv.DCDiscriminatorCond,
    'resnet': conv.DiscriminatorResnet,
    'resnet_cond': conv.DiscriminatorResnetCond,
}
