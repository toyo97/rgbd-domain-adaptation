import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numbers


class RGBDCompose(object):
    """Composes several transforms together considering the multimodality of the input.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, rgb, depth):
        newlabel = None
        for t in self.transforms:
            if isinstance(t, CoupledRotation):
                rgb, depth, newlabel = t(rgb, depth)
                continue
            try:
                rgb = t(rgb)
                depth = t(depth)
            except TypeError:
                rgb, depth = t(rgb, depth)
        return rgb, depth, newlabel

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class CoupledRandomCrop(object):
    """Multi-modality implementation.
    Crop the given PIL Images at a random location.

    Args:
        size (sequence or int): Desired output size of the crops. If size is an
            int instead of sequence like (h, w), square crops (size, size) are
            made.
        padding_rgb (int or sequence, optional): Optional padding on each border
            of the rgb image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        padding_depth (int or sequence, optional): Optional padding on each border
            of the depth image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively. 
        pad_if_needed (boolean): It will pad the images if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, size, padding_rgb=None, padding_depth=None, pad_if_needed=False, fill=0,
                 padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding_rgb = padding_rgb
        self.padding_depth = padding_depth
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(rgb, depth, output_size: tuple):
        """Get parameters for ``crop`` for a random crop.

        Args:
            rgb (PIL Image): RGB Image to be cropped.
            depth (PIL Image): depth Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w_rgb, h_rgb = transforms.transforms._get_image_size(rgb)
        w_depth, h_depth = transforms.transforms._get_image_size(depth)

        th, tw = output_size
        if w_rgb == tw and h_rgb == th and w_depth == tw and h_depth == th:
            return 0, 0, th, tw

        i = random.randint(0, min(h_rgb, h_depth) - th)
        j = random.randint(0, min(w_rgb, w_depth) - tw)
        return i, j, th, tw

    def __call__(self, rgb, depth):
        """
        Args:
            rgb (PIL Image): RGB Image to be cropped.
            depth (PIL Image): depth Image to be cropped.

        Returns:
            rgb (PIL Image): Cropped RGB image.
            depth (PIL Image): Cropped depth image.
        """
        if self.padding_rgb is not None:
            rgb = F.pad(rgb, self.padding, self.fill, self.padding_mode)
        if self.padding_depth is not None:
            depth = F.pad(depth, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and rgb.size[0] < self.size[1]:
            rgb = F.pad(rgb, (self.size[1] - rgb.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and depth.size[0] < self.size[1]:
            depth = F.pad(depth, (self.size[1] - depth.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and rgb.size[1] < self.size[0]:
            rgb = F.pad(rgb, (0, self.size[0] - rgb.size[1]), self.fill, self.padding_mode)
        if self.pad_if_needed and depth.size[1] < self.size[0]:
            depth = F.pad(depth, (0, self.size[0] - depth.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(rgb, depth, self.size)

        return F.crop(rgb, i, j, h, w), F.crop(depth, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class CoupledRandomHorizontalFlip(object):
    """Horizontally flip the couple of PIL Images randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, rgb, depth):
        """
        Args:
            rgb (PIL Image): RGB Image to be flipped.
            depth (PIL Image): depth Image to be flipped.

        Returns:
            rgb (PIL Image): Randomly flipped RGB image.
            depth (PIL Image): Randomly flipped depth image.
        """
        if random.random() < self.p:
            return F.hflip(rgb), F.hflip(depth)
        return rgb, depth

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class CoupledRotation(object):
    """Rotate the two PIL Images.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self):
        pass

    def __call__(self, rgb, depth):
        """
        Args:
            rgb (PIL Image): RGB Image to be rotated.
            depth (PIL Image): depth Image to be rotated.

        Returns:
            rgb (PIL Image): Randomly rotated RGB image.
            depth (PIL Image): Randomly rotated depth image.
            z: relative rotation.
        """
        j = random.randint(0, 3)
        k = random.randint(0, 3)

        z = (k - j) % 4

        # Note: TF.rotate is counter-clockwise
        rgb = F.rotate(rgb, 270 * j)
        depth = F.rotate(depth, 270 * k)

        return rgb, depth, z

    def __repr__(self):
        return self.__class__.__name__
