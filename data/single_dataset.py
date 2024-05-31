from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
from data.ffhq_dataset import complex_imgaug

class SingleDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.A_paths = sorted(make_dataset(opt.src_dir, opt.max_dataset_size))
        input_nc = self.opt.output_nc 
        # self.transform = get_transform(opt, grayscale=(input_nc == 1))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.opt = opt
 
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        A_path = self.A_paths[index]
        hr_img = Image.open(A_path).convert('RGB')
        # A_img = A_img.resize((256, 256), Image.BICUBIC)
        # A_img = A_img.resize((256, 256), Image.BILINEAR)
        hr_img = hr_img.resize((256, 256))
        # hr_img = A_img.resize((256, 256))
        # hr_img = random_gray(hr_img, p=0.3)
        # scale_size = np.random.randint(32, 128)
        # lr_img = complex_imgaug(hr_img, self.img_size, scale_size)

        lr_img = complex_imgaug(hr_img, 256, scale_size = 64)

        # A = self.transform(A_img)
        A = self.transform(lr_img)
        hr_img = self.transform(hr_img)
        return {'LR': A, 'LR_paths': A_path, 'HR':hr_img}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.A_paths)
