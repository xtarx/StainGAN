# StainGAN
StainGAN implementation based on Cycle-Consistency Concept

For more information visit website.

## Structure
 * Stain-Transfer Model
 * Pre-processing.
 * Post-processing.
 * Evaluation 

## Datasets

The evaluation was done using the Camelyon16 challenge (https://camelyon16.grand-challenge.org/) consisting of 400 whole-slide images collected
in two different labs in Radboud University Medical Center (lab 1) and University
Medical Center Utrecht (lab 2). Otsu thresholding was used to remove the
background, Afterwards, 40, 000 256 × 256 patches were generated on the x40
magnification level, 30, 000 were used for training and 10, 000 used for validation
from lab 1 and 10, 000 patches were generated for testing from lab 2.

Patches can be found here: https://campowncloud.in.tum.de/index.php/s/iGgQ9vdHiMZsFJB?path=%2FStainGAN_camelyon16 

**Any use of the dataset or anypart of the code should be cited**


## Citation
If you use this code for your research, please cite our papers.
```
@article{shaban2018staingan,
  title={StainGAN: Stain Style Transfer for Digital Histological Images},
  author={Shaban, M Tarek and Baur, Christoph and Navab, Nassir and Albarqouni, Shadi},
  journal={arXiv preprint arXiv:1804.01601},
  year={2018}
}
```


## Todo
- [x] Submit Matlab Image Similarity code
- [ ] Submit Trained Model and sample images
- [ ] Update Readme with more examples and explanation
## Acknowledgments
Code is inspired by [pytorch-DCGAN](https://github.com/pytorch/examples/tree/master/dcgan) and [CycleGAN](https://github.com/junyanz/CycleGAN).

Contact <a href="mailto:shadi.albarqouni@tum.de?Subject=StainGAN" target="_top">Shadi Albarqouni</a> 
 
