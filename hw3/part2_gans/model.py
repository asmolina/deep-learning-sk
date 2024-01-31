import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm


###################################################################################################################
###################################################################################################################
######################################      AdaptiveBatchNorm     #################################################
###################################################################################################################
###################################################################################################################

class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer (=C in expected input shape (N,C,H,W)
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        # TODO
        self.num_features = num_features
        self.embed_features = embed_features
        self.batchnorm = nn.BatchNorm2d(self.num_features, affine=False)
        
        # Let's use nn.Linear:
        self.f = spectral_norm(nn.Linear(self.embed_features, self.num_features, bias=False))
        self.g = spectral_norm(nn.Linear(self.embed_features, self.num_features, bias=False))

        ##############################################################################
    def forward(self, inputs, embeds):
        # gamma = ... # TODO 
        # bias = ... # TODO
        gamma = self.f(embeds)
        bias = self.g(embeds)

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        # outputs = ... # TODO: apply batchnorm
        outputs = self.batchnorm(inputs)
        
        return outputs * gamma[..., None, None] + bias[..., None, None]




###################################################################################################################
###################################################################################################################
#########################################      PreActResBlock     #################################################
###################################################################################################################
###################################################################################################################
class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!

        self.upsample = upsample
        self.downsample = downsample
        self.batchnorm = batchnorm
        self.embed_channels = embed_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
            
        # Define pre-activation residual block
        # * With AdaptiveBatchNorm
        if self.batchnorm and self.embed_channels is not None:
            self.adaptiveBN1 = AdaptiveBatchNorm(num_features=self.in_channels, embed_features=self.embed_channels) # adaBN
            self.relu1 = nn.ReLU(inplace=False)
            self.conv1 = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            self.adaptiveBN2 = AdaptiveBatchNorm(num_features=self.out_channels, embed_features=self.embed_channels) # adaBN
            self.relu2 = nn.ReLU(inplace=False)
            self.conv2 = spectral_norm(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        # * With plain BatchNorm
        elif self.batchnorm and self.embed_channels is None:
            self.BN1 = nn.BatchNorm2d(self.in_channels)
            self.relu1 = nn.ReLU(inplace=False)
            self.conv1 = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False))
            self.BN2 = nn.BatchNorm2d(self.out_channels)
            self.relu2 = nn.ReLU(inplace=False)
            self.conv2 = spectral_norm(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        # * Without any BatchNorm
        else:
            self.relu1 = nn.ReLU(inplace=False)
            self.conv1 = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True))
            self.relu2 = nn.ReLU(inplace=False)
            self.conv2 = spectral_norm(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=True))
            
        # Define conv in skip connection for the case in_channels != out_channels:
        if self.in_channels != self.out_channels:
            self.skip_connection = spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True))
        
        # Define downsampling (if needed)
        # For downsampling by the factor of 2, we need k=4, p=1, s=2 (indeed, H_out = [(H_in + 2p - k)/s + 1] = H_in/2).
        if self.downsample:
            self.downsample_layer = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        
            
    ##############################################################################            
    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm
        #pass # TODO
        
        # Apply upsampling by factor of 2 (if needed):
        if self.upsample:
            x = F.interpolate(inputs, size=(inputs.shape[2] * 2, inputs.shape[3] * 2), mode='nearest')
        else:
            x = inputs
        
        # Apply residual block:
        if self.batchnorm and self.embed_channels is not None:
            residual = self.adaptiveBN1(x, embeds)
            residual = self.relu1(residual)
            residual = self.conv1(residual)
            residual = self.adaptiveBN2(residual, embeds)
            residual = self.relu2(residual)
            residual = self.conv2(residual)

        elif self.batchnorm and self.embed_channels is None:
            residual = self.BN1(x)
            residual = self.relu1(residual)
            residual = self.conv1(residual)
            residual = self.BN2(residual)
            residual = self.relu2(residual)
            residual = self.conv2(residual)

        else:
            residual = self.relu1(x)
            residual = self.conv1(residual)
            residual = self.relu2(residual)
            residual = self.conv2(residual)
        
        # Apply conv in skip connection (if needed):
        if self.in_channels != self.out_channels:
            x = self.skip_connection(x)
        
        # Sum up the residual and skip connection
        outputs = x + residual

        # Apply downsampling (if needed):
        if self.downsample:
            outputs = self.downsample_layer(outputs)

        return outputs




###################################################################################################################
###################################################################################################################
#########################################      GENERATOR     ######################################################
###################################################################################################################
###################################################################################################################

class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      + Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      + Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      + Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      + Forward an input tensor through a convolutional part,
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      + Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      ? At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      + Apply spectral norm to all conv and linear layers (not the embedding layer)

      ? Use Sigmoid at the end to map the outputs into an image

    Notes:

      + The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      + Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks
        # TODO
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.noise_channels = noise_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_class_condition = use_class_condition


        if self.use_class_condition:
            # Embeddings for each available class
            self.class_embeddings = nn.Embedding(num_classes, noise_channels)
            # if use_class_condition=True => we will use AdaptiveBatchNorm
            self.embed_channels = 2 * self.noise_channels
            self.linear_map = spectral_norm(
                nn.Linear(in_features=2 * self.noise_channels, out_features=self.max_channels * 16, bias=False), eps=1e-8)

        else:
            # use_class_condition=False => we will not use AdaptiveBatchNorm (but just plain BatchNorm)
            self.embed_channels = None
            self.linear_map = spectral_norm(
                nn.Linear(in_features=self.noise_channels, out_features=self.max_channels * 16, bias=False), eps=1e-8)


        n_channels = self.max_channels
        for i in range(1, self.num_blocks + 1):
            upsamp = PreActResBlock(in_channels=n_channels, out_channels=n_channels // 2,
                                    embed_channels=self.embed_channels, batchnorm=True, upsample=True, downsample=False)
            setattr(self, "PreActResBlock_upsamp" + str(i), upsamp)
            n_channels = n_channels // 2

        self.image_prediction = nn.Sequential(
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=False),
            spectral_norm(nn.Conv2d(n_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)),
            nn.Sigmoid() 
            )

    ##############################################################################
    def forward(self, noise, labels):
        # TODO

        # Concatenate input noise with class embeddings:
        if self.use_class_condition:
            input_embeddings =  torch.cat((noise, self.class_embeddings(labels)), dim=1)
        else:
            input_embeddings = noise

        # Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4
        x = self.linear_map(input_embeddings)
        x = torch.reshape(x, shape=(noise.shape[0], self.max_channels, 4, 4))

        # Forward an input tensor through a convolutional part, which consists of num_blocks PreActResBlocks.
        # Each PreActResBlock is conditioned on the input embeddings (since we set batchnorm=True in PreActResBlock).
        if self.use_class_condition:
            for i in range(1, self.num_blocks + 1):
                x = getattr(self, "PreActResBlock_upsamp" + str(i))(x, embeds=input_embeddings)
        else:
            # use_class_condition=False => we will not use AdaptiveBatchNorm (but just plain BatchNorm), because
            # after series of experiments it turns out that AdaptiveBatchNorm only make worse the results when it is
            # conditioned only on noise (the results are just some colorful rectangles, without any hint on flowers).
            # But with conditioning on torch.cat(class_embeds, noise) it significantly improves the generated images.
            # So let's use plain BN instead of AdaBN in this case:
            for i in range(1, self.num_blocks + 1):
                x = getattr(self, "PreActResBlock_upsamp" + str(i))(x)

        # At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head.
        # Use Sigmoid at the end to map the outputs into an image (we added it to self.image_prediction)
        outputs = self.image_prediction(x)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


###################################################################################################################
###################################################################################################################
#########################################    DISCRIMINATOR   ######################################################
###################################################################################################################
###################################################################################################################

class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
      + Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      + At the end of the convolutional part apply ReLU and sum pooling
    
    TODO:
      + implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    Scheme: materials/prgan.png
    Notation:
      - phi is a convolutional part of the discriminator
      - psi is a vector
      - y is a class embedding
    
    + Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    + Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        # TODO
        self.min_channels = min_channels
        self.max_channels = max_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.use_projection_head = use_projection_head

        # Convolution from 3 to self.min_channels(=32) channels
        n_channels = self.min_channels
        self.image_prediction_inv = nn.Sequential(
            spectral_norm(nn.Conv2d(3, n_channels, kernel_size=1, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(inplace=False),
        )

        # Convolutional part: perform downsampling by the factor of 2 so that,
        # after all self.num_blocks blocks number of channels is equal to self.max_channels.
        # (i.e. at some steps the number of channels doubles and (probably) at some steps does not)
        n_channels = self.max_channels
        for i in range(self.num_blocks, 0, -1):
            if n_channels // 2 >= self.min_channels:
                downsamp = PreActResBlock(in_channels=n_channels // 2, out_channels=n_channels,
                                          embed_channels=None, batchnorm=False,
                                          upsample=False, downsample=True)
                setattr(self, "PreActResBlock_downsamp" + str(i), downsamp)
                n_channels = n_channels // 2
            else:
                downsamp = PreActResBlock(in_channels=n_channels, out_channels=n_channels,
                                          embed_channels=None, batchnorm=False,
                                          upsample=False, downsample=True)
                setattr(self, "PreActResBlock_downsamp" + str(i), downsamp)


        # ReLU at the end of the conv part:
        self.end_relu = nn.ReLU(inplace=False)

        # psi: linear layer which maps a vector to a single digit
        self.psi = spectral_norm(nn.Linear(in_features=self.max_channels, out_features=1))

        # Embeddings (if projection is used as a conditioning method)
        if self.use_projection_head:
            self.class_embeddings = spectral_norm(nn.Embedding(self.num_classes, self.max_channels), eps=1e-8) 


    ###################################################################################################################
    def forward(self, inputs, labels):
        # pass # TODO

        # Conv at the begging to get (b, 3, h, w) -> (b, min_channels, h, w)
        x = self.image_prediction_inv(inputs)

        # Convolutional part
        for i in range(1, self.num_blocks + 1):
            x = getattr(self, "PreActResBlock_downsamp" + str(i))(x)

        # ReLU & sum pooling to get (b, c, h, w) -> (b, c, 1, 1) -> (b, c).
        # `phi` is the output from the convolutional part.
        x = self.end_relu(x)
        x = F.lp_pool2d(x, norm_type=1, kernel_size=(x.shape[2], x.shape[3]))
        phi = torch.reshape(x, shape=(x.shape[0], x.shape[1]))

        # Linear layer which maps a vector to a single digit.
        psi = self.psi(phi)

        if self.use_projection_head:
            # Obtain trainable class embeddings:
            y = self.class_embeddings(labels)

            # Calculate inner product for y and output of convolutional part
            multiplication = torch.matmul(y, torch.transpose(phi, dim0=0, dim1=1))
            inner_product = torch.reshape(torch.diag(multiplication, diagonal=0), shape=(-1, 1))

            scores = torch.reshape(psi + inner_product, shape=(psi.shape[0],))
        else:
            scores = torch.reshape(psi, shape=(psi.shape[0],))


        # scores = ... # TODO
        assert scores.shape == (inputs.shape[0],)
        return scores