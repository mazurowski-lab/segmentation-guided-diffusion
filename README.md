# Easy Segmentation-Guided Diffusion Models

*Have images with corresponding segmentation masks? Easily train a diffusion model to learn to generate images conditioned on the masks.*

#### By [Nicholas Konz](https://nickk124.github.io/), [Yuwen Chen](https://scholar.google.com/citations?user=61s49p0AAAAJ&hl=en), [Haoyu Dong](https://scholar.google.com/citations?user=eZVEUCIAAAAJ&hl=en) and [Maciej Mazurowski](https://sites.duke.edu/mazurowski/).

<img src='https://github.com/mazurowski-lab/segmentation-guided-diffusion/blob/main/figs/teaser.png' width='100%'>

This is the code for our paper **"Anatomically-Controllable Medical Image Generation with Segmentation-Guided Diffusion Models"**, where we introduce a simple yet powerful training procedure for conditioning image-generating diffusion models on (possibly incomplete) multiclass segmentation masks. 

Our method is considerably simpler and faster to train than existing segmentation-guided models (like the latent diffusion model ControlNet), and has much more precise anatomical conditioning due to completely operating in image space, especially for medical image datasets that are OOD from natural images. In our paper, we show that this results in significantly better anatomical control and realism in generated images, especially for medical images with complex and detailed anatomical structures (such as fibroglandular tissue in breast MRI).

**Using this code, you can:**
1. Train a segmentation-guided (or standard unconditional) diffusion model on your own dataset, with a wide range of options.
2. Generate images from these models (or using our provided pre-trained models).

Please follow the steps outlined below to do these. 

Also, check out our accompanying "**Synthetic Paired Breast MRI Dataset Release**" below!

Thank you to Hugging Face's awesome [Diffusers](https://github.com/huggingface/diffusers) library for providing a helpful backbone for our code!

## Citation

Please cite our paper if you use our code or reference our work:

## 1) Package Installation
First, run `pip3 install -r requirements.txt` to install the required packages.

## 2a) Use Pre-Trained Models

We provide pre-trained model checkpoints from our paper for the [Duke Breast MRI](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/) and [CT Organ](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=61080890) datasets, [here](https://drive.google.com/drive/folders/1OaOGBLfpUFe_tmpvZGEe2Mv2gow32Y8u). These include:

1. Segmentation-Conditional Models, trained without mask ablation.
2. Segmentation-Conditional Models, trained with mask ablation.
3. Unconditional (standard) Models.

Once you've downloaded your model of choice, please put it in a directory called `{NAME}/unet`, where `NAME` is the model checkpoint's filename without the `.safetensors` ending, to use it with our evaluation code. Next, you can proceed to the **Evaluation/Sampling** section below to generate images from these models.

## 2b) Train Your Own Models

### Data Preparation

Please put your training images in some dataset directory `DATA_FOLDER`, organized into train, validation and test split subdirectories. The images should be in a format that PIL can read (e.g. `.png`, `.jpg`, etc.). For example:

``` 
DATA_FOLDER
├── train
│   ├── tr_1.png
│   ├── tr_2.png
│   └── ...
├── val
│   ├── val_1.png
│   ├── val_2.png
│   └── ...
└── test
    ├── ts_1.png
    ├── ts_2.png
    └── ...
```

If you have segmentation masks, please put them in a similar directory structure in a separate folder `MASK_FOLDER`, with a subdirectory `all` that contains the split subfolders, as shown below. **Each segmentation mask should have the same filename as its corresponding image in `DATA_FOLDER`, and should be saved with integer values starting at zero of 0, 1, 2, ...

If you don't want to train a segmentation-guided model, you can skip this step.

``` 
MASK_FOLDER
├── all
│   ├── train
│   │   ├── tr_1.png
│   │   ├── tr_2.png
│   │   └── ...
│   ├── val
│   │   ├── val_1.png
│   │   ├── val_2.png
│   │   └── ...
│   └── test
│       ├── ts_1.png
│       ├── ts_2.png
│       └── ...
```

### Training

The basic command for training a standard unconditional diffusion model is
```bash
CUDA_VISIBLE_DEVICES={DEVICES} python3 main.py \
    --mode train \
    --model_type DDIM \
    --img_size 256 \
    --dataset {DATASET_NAME} \
    --img_dir {DATA_FOLDER} \
    --train_batch_size 16 \
    --eval_batch_size 8 \
    --num_epochs 400
```

where:
- `DEVICES` is a comma-separated list of GPU device indices to use (e.g. `0,1,2,3`).
- `model_type` specifies the type of diffusion model sampling algorithm to evaluate the model with, and can be `DDIM` or `DDPM`.
- `DATASET_NAME` is some name for your dataset (e.g. `breast_mri`).
- `DATA_FOLDER` is the path to your dataset directory, as outlined in the previous section.

#### Adding segmentation guidance, mask-ablated training, and other options

To train your model with mask guidance, simply add the options:
```bash
    --seg_dir {MASK_FOLDER} \
    --segmentation_guided \
    --num_segmentation_classes {N_SEGMENTATION_CLASSES} \
```

where:
- `MASK_FOLDER` is the path to your segmentation mask directory, as outlined in the previous section.
- `N_SEGMENTATION_CLASSES` is the number of classes in your segmentation masks, **including the background (0) class**.

To also train your model with mask ablation (randomly removing classes from the masks to each the model to condition on masks with missing classes; see our paper for details), simply also add the option `--use_ablated_segmentations`.

## 3) Evaluation/Sampling

Sampling images with a trained model is run similarly to training. For example, 100 samples from an unconditional model can be generated with the command:
```bash
CUDA_VISIBLE_DEVICES={DEVICES} python3 main.py \
    --mode eval_many \
    --model_type DDIM \
    --img_size 256 \
    --dataset {DATASET_NAME} \
    --img_dir {DATA_FOLDER} \
    --eval_batch_size 8 \
    --eval_sample_size 100
```

Note that the code will automatically use the checkpoint from the training run, and will save the generated images to a directory called `samples` in the model's output directory. To sample from a model with segmentation guidance, simply add the options:
```bash
    --seg_dir {MASK_FOLDER} \
    --segmentation_guided \
    --num_segmentation_classes {N_SEGMENTATION_CLASSES} \
```
This will generate images conditioned on the segmentation masks in `MASK_FOLDER/all/test`.

## Additional Options/Config
Our code has further options for training and evaluation; run `python3 main.py --help` for more information. Further settings still can be changed under `class TrainingConfig:` in `training.py` (some of which are exposed as command-line options for `main.py`, and some of which are not).

## Synthetic Paired Breast MRI Dataset Release

<img src='https://github.com/mazurowski-lab/segmentation-guided-diffusion/blob/main/figs/teaser_data.png' width='100%'>


We also release synthetic 2D breast MRI slice images that are paired/"pre-registered" in terms of blood vessels and fibroglandular tissue to real image counterparts, generated by our segmentation-guided model. These were created by applying our mask-ablated-trained segmentation-guided model to the existing segmentation masks of the held-out training set and test sets of our paper (30 patients total or about ~5000 2D slice images; see the paper for more information), but with the breast mask removed. Because of this, each of these synthetic images have blood vessels and fibroglandular tissues that are registered to/spatially match those of a real image counterpart, but have different surrounding breast tissue and shape. This paired data enables potential applications such as training some breast MRI registration model, self-supervised learning, etc.

The data can be downloaded [here](https://drive.google.com/file/d/1yaLLdzMhAjWUEzdkTa5FjbX3d7fKvQxM/view).

### Filename convention/dataset organization
The generated images are stored in `synthetic_data` in the downloadable `.zip` file above, with the filename convention `condon_Breast_MRI_{PATIENTNUMBER}_slice_{SLICENUMBER}.png`, where `PATIENTNUMBER` is the original dataset's patient number that the image comes from, and `SLICE_NUMBER` is the $z$-axis index of the original 3D MRI image that the 2D slice image was taken from. The corresponding real slice image, found in `real_data`, is then named `Breast_MRI_{PATIENTNUMBER}_slice_{SLICENUMBER}.png`. For example:
``` 
synthetic_data
│   ├── condon_Breast_MRI_002_slice_0.png
│   ├── condon_Breast_MRI_002_slice_1.png
│   └── ...
real_data
│   ├── Breast_MRI_002_slice_0.png
│   ├── Breast_MRI_002_slice_1.png
│   └── ...
```

### Dataset Citation
If you use this data, please cite both our paper (see **Citation** above) and the original breast MRI dataset:
```bib
@misc{sahadata,
  author={Saha, Ashirbani and Harowicz, Michael R and Grimm, Lars J and Kim, Connie E and Ghate, Sujata V and Walsh, Ruth and Mazurowski, Maciej A},
  title = {Dynamic contrast-enhanced magnetic resonance images of breast cancer patients with tumor locations [Data set]},
  year = {2021},
  publisher = {The Cancer Imaging Archive},
  howpublished = {\url{https://doi.org/10.7937/TCIA.e3sv-re93}},
}
```
