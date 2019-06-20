import torch
from torchvision import transforms


def postp(tensor):  # to clip results in the range [0,1]
    """
    :param tensor: torch.Tensor
    :return: PIL.Image
    """

    # what's this do?
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])

    # what's this do?
    postpb = transforms.Compose([transforms.ToPILImage()])

    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img


def image_to_tensor(image, do_cuda):
    """
    :param [PIL.Image]
    :return: [torch.Tensor]
    """
    # pre and post processing for images
    img_size = 512

    tforms = transforms.Compose([transforms.Resize(img_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])

    tensor = tforms(image)

    if do_cuda:
        return tensor.unsqueeze(0).cuda()
    else:
        return tensor.unsqueeze(0)