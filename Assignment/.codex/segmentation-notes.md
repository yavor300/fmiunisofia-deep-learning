
# Week 06 - Segmentation

## Setting the context around the task of Segmentation

<details>
<summary>What is image segmentation?</summary>

- Partitioning an image into multiple segments on the pixel level.
- Each pixel in an image is assigned to a particular segment.

</details>

<details>
<summary>Give three examples of image segmentation algorithms we've seen so far?</summary>

Image binarization (either manual or via a filter (for example, Otsu)):

![w06_pure_segmenation.png](assets/w06_pure_segmenation.png "w06_pure_segmenation.png")

Superpixel segmentation (using KMeans):

![w06_superpixel.png](assets/w06_superpixel.png "w06_superpixel.png")

Selective search does image segmentation as a first step:

![w06_ss_paper_example.png](assets/w06_ss_paper_example.png "w06_ss_paper_example.png")

</details>

<details>
<summary>What is semantic segmentation?</summary>

- Each pixel is classified into a predefined class / category.
- All pixels belonging to the same class are treated equally.
- No distinction is made between different instances of the same class.

</details>

<details>
<summary>What is the main difference between image segmentation and semantic segmentation?</summary>

An object might be part of multiple segments when we use image segmentation, but is part of only segment in semantic segmentation (hence the name).

</details>

<details>
<summary>Having said this what do you think are the most popular use cases for semantic segmentation?</summary>

In general, all use cases in which there is a need for **scene understanding**:

- Autonomous navigation. <- most popular
- Assisting the partially sighted.
- Medical diagnosis.
- Image editing.

![w06_sem_seg.png](assets/w06_sem_seg.png "w06_sem_seg.png")

</details>

<details>
<summary>What would semantic segmentation produce for the above image?</summary>

![w06_sem_seg_result.png](assets/w06_sem_seg_result.png "w06_sem_seg_result.png")

</details>

Last week we introduced the R-CNN model family and the YOLO model family. Open the [paper](https://arxiv.org/pdf/1311.2524) that discusses the first iteration of R-CNN and answer the following question.

<details>
<summary>Which are the two main datasets used to evaluate the model?</summary>

- [PASCAL VOC 2010](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html).
- [ILSVRC2013 detection dataset](https://www.image-net.org/challenges/LSVRC/2013/index.php).

</details>

<details>
<summary>Look through the first dataset - does it support a segmentation task?</summary>

Yep - we can use it to create, train and evaluate semantic segmentation models:

![w06_pascal_voc_segmentation.png](assets/w06_pascal_voc_segmentation.png "w06_pascal_voc_segmentation.png")

</details>

<details>
<summary>Do you know any other popular datasets for semantic segmentation?</summary>

- Microsoft's [COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) dataset ([homepage](https://cocodataset.org/#home)).
- The [Cityscapes](https://www.cityscapes-dataset.com/) Dataset.

</details>

<details>
<summary>What other types of segmentation tasks are there?</summary>

- Instance segmentation.
- Panoptic segmentation.

</details>

<details>
<summary>What is instance segmentation?</summary>

- Distinguishes between different instances of the same class.
- Background is not segmented (i.e. classified as belonging to class `0`).

![w06_sem_seg.png](assets/w06_sem_seg.png "w06_sem_seg.png")

<details>
<summary>What would instance segmentation produce for the above image?</summary>

![w06_inst_seg_result.png](assets/w06_inst_seg_result.png "w06_inst_seg_result.png")

</details>

</details>

<details>
<summary>What is panoptic segmentation?</summary>

- Combines semantic segmentation results with instance segmentation results.
- Assigns unique label to each instance of an object.
- Classifies background at pixel level.

![w06_sem_seg.png](assets/w06_sem_seg.png "w06_sem_seg.png")

<details>
<summary>What would panoptic segmentation produce for the above image?</summary>

![w06_pan_seg_result.png](assets/w06_pan_seg_result.png "w06_pan_seg_result.png")

</details>

</details>

The latest YOLO model as of today is [YOLO26](https://docs.ultralytics.com/models/yolo26/). Open its documentation via the link and answer the following question.

<details>
<summary>What kind of segmentation task does it support?</summary>

It supports instance segmentation - we can see how it performs on COCO in the table here (select the tab "Segmentation"): <https://docs.ultralytics.com/models/yolo26/#performance-metrics>.

To see how you can use YOLO26 on instance segmentation tasks see the [documentation here](https://docs.ultralytics.com/tasks/segment/).

</details>

## Semantic segmentation

### Data annotations

<details>
<summary>Would would our training set look like when we do semantic segmentation?</summary>

If we refer to a training example as having an input and a target output then:

- the input would be an image (2D or 3D);
- the output would actually also be a image. Because it's the segmentation output we'll refer the target output as a `mask`.

So then for our total training set, we'd have:

- A set of all images (2D or 3D).
- A set of the masks for each image.

</details>

<details>
<summary>What values would the mask hold?</summary>

- It's a 2D matrix with the same size of the image it's annotating.
- Each value in the mask is the class/category of each pixel in the image.

</details>

In our `DATA` folder you'll find the dataset `segmentation_cats_dogs`. It holds pictures of cats and dogs alongside their masks. The pixel annotations are as follows:

- `1`: Object / Foreground.
- `2`: Background.
- `3`: Not classified.

Let's load one image from this dataset:

```python
import os

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

image = Image.open(os.path.join('DATA', 'segmentation_cats_dogs', 'images', 'British_Shorthair_36.jpg'))
mask = Image.open(os.path.join('DATA', 'segmentation_cats_dogs', 'annotations', 'British_Shorthair_36.png'))

plt.axis('off')
plt.imshow(image)
plt.show()

transform = transforms.Compose([
  transforms.ToTensor()
])
image_tensor = transform(image)
mask_tensor = transform(mask)

print(f'Image shape: {image_tensor.shape}')
print(f'Mask shape: {mask_tensor.shape}')
```

![w06_example_cat.png](assets/w06_example_cat.png "w06_example_cat.png")

```console
Image shape: torch.Size([3, 333, 500])
Mask shape: torch.Size([1, 333, 500])
```

<details>
<summary>How would we print the unique values in "mask_tensor"?</summary>

Since it's a `PyTorch` tensor that is based on `numpy`, we can just call its method `.unique`:

```python
mask_tensor.unique()
```

```console
tensor([0.0039, 0.0078, 0.0118])
```

<details>
<summary>Wait a minute - that doesn't make sense! Why are we not seeing "1", "2" and "3"?</summary>

Because the transformation `ToTensor` also normalizes the values:

- `1 / 255 = 0.0039` - object / foreground.
- `2 / 255 = 0.0078` - background.
- `3 / 255 = 0.0118` - unclassified.

</details>

</details>

Let's say that we want to remove the third class, so as to perform binary image segmentation like so:

![w06_example_cat_bin_mask.png](assets/w06_example_cat_bin_mask.png "w06_example_cat_bin_mask.png")

<details>
<summary>How can we create the above binary mask?</summary>

We can use `torch.where`:

```python
binary_mask = torch.where(
  mask_tensor == 1/255,
  torch.tensor(1.0),
  torch.tensor(0.0),
)

mask = transforms.ToPILImage()(binary_mask)
plt.imshow(mask)
```

![w06_example_cat_bin_mask.png](assets/w06_example_cat_bin_mask.png "w06_example_cat_bin_mask.png")

</details>

<details>
<summary>How can we then crop out the cat in the original image using this new mask?</summary>

We multiply the image with the mask:

```python
to_pil_image = transforms.ToPILImage()(image_tensor * binary_mask)
plt.imshow(to_pil_image)
```

![w06_example_cat_bin_mask_segmented_out.png](assets/w06_example_cat_bin_mask_segmented_out.png "w06_example_cat_bin_mask_segmented_out.png")

</details>

### Fully Convolutional Networks for Semantic Segmentation (FCN)

This family of models is the first to successfully address the task of semantic segmentation. Let's examine the solution.

Paper: <https://arxiv.org/abs/1411.4038>.

<details>
<summary>Open the paper and find the key insight the authors use to distinguish FCNs from traditional convolutional neural networks.</summary>

The FCN network is `fully convolutional`.

![w06_fcn_paper_p1.png](assets/w06_fcn_paper_p1.png "w06_fcn_paper_p1.png")

</details>

<details>
<summary>This is kind of vague and abstract though. Which figure describes visually the process of "convolutionalization"?</summary>

Linear layers get substituted by convolutional layers.

![w06_fcn_paper_p2.png](assets/w06_fcn_paper_p2.png "w06_fcn_paper_p2.png")

</details>

<details>
<summary>What is the added value of this?</summary>

By doing this we only care about the filters and we do not depend on the input size of the images that come into our network.

![w06_fcn_paper_p3.png](assets/w06_fcn_paper_p3.png "w06_fcn_paper_p3.png")

</details>

<details>
<summary>But wait - why are traditional CNNs bound to a particular input size?</summary>

- It's exactly because we have `Linear` layers as part of the classification.
- Recall, we have two parts in the traditional CNNs:
  1. Feature extraction.
  2. Classification.
- The final step in the feature extraction is to `flatten` the feature maps into a `feature vector` which is then fed into at least `1` `Linear` layer.
  - The first `Linear` layer always has an input size that is **dependent** on the width and height of the images it has been **trained** on.
- See how we have `64 * 16 * 16` in the below example:

![w06_traditional_cnn_problem.png](assets/w06_traditional_cnn_problem.png "w06_traditional_cnn_problem.png")

<details>
<summary>What does the number 64 mean in the linear layer?</summary>

This is the number of output channels produced by the last convolutional layer.

</details>

<details>
<summary>What does the shape 16x16 mean in the linear layer?</summary>

This is the final shape of each channel. This is coupled with / determined by the input image size. If you have `k` pooling layers with stride `2`, the spatial size of the final feature maps becomes:

$$\frac{\text{input\_size}}{2^k}$$

In our case since we have `16x16` that means that the input had since `64` or in more detail:

| Stage   | Shape            |
| ------- | ---------------- |
| Input   | 3x64x64      |
| Conv1   | 32x64x64     |
| MaxPool | 32x32x32     |
| Conv2   | 64x32x32     |
| MaxPool | **64x16x16** |
| Flatten | **16384**        |

</details>

In short, **if we train a CNN on a particular set of images, at inference time we must use the same size as the training images**.

</details>

<details>
<summary>Ok, so that solves something, but it does not really help us do semantic segmentation - which figure lays out the "backend" architecture of an FCN network?</summary>

![w06_fcn_paper_p4.png](assets/w06_fcn_paper_p4.png "w06_fcn_paper_p4.png")

</details>

<details>
<summary>What does this tell us about the final layers - what is their width?</summary>

Their width is actually `1`.

</details>

<details>
<summary>What is their height?</summary>

Their height is also `1`!

</details>

<details>
<summary>So what does this "21" refer to?</summary>

This is the depth / the number of channels / the number of feature maps.

</details>

<details>
<summary>By why exactly "21" - why not "42"?</summary>

The final layer is just another representation of the final `Linear` layer - it corresponds to the number of output classes.

So, in essence, we get a `1 x 1 x num_classes`.

</details>

<details>
<summary>But wait, wait - to get "1" for the final width and heigth don't you have to know the input image size?</summary>

Yep, that is a perfectly valid question! The idea is that the authors pick input sizes such that during training they collapse to `1x1`.

However, that does not stop us from passing a larger image. We would not get `1x1` but instead - something larger. That would not be a problem since that is not the end of the network. We still have one additional part which is not shown in the above picture.

</details>

<details>
<summary>What part do we not see in the above picture?</summary>

We're still missing the part for going from `1 x 1 x num_classes` to the original height and width that would become our prediction mask.

</details>

<details>
<summary>What is the output shape of the model?</summary>

`original_height x original_width x num_classes`

</details>

<details>
<summary>What is the meaning of the last coordinate?</summary>

The model returns a mask for each class of objects. Each 2D matrix is responsible for 1 class.

</details>

<details>
<summary>What would the loss function aim to produce then?</summary>

The objective is to make each mask be binarized.

</details>

<details>
<summary>Which paragraph in the paper describes the entire architecture, including the final part?</summary>

![w06_fcn_paper_p5.png](assets/w06_fcn_paper_p5.png "w06_fcn_paper_p5.png")

</details>

<details>
<summary>Hmm - that's interesting - how many times does a prediction occur then?</summary>

- It occurs `n` times, where `n` is the number of max pooling layers.
- As they do max pooling, they also make a prediction (you'll see a diagram in a bit).
- Note that a prediction here is just a convolution to `C` classes in the `z` axis.

</details>

<details>
<summary>Ok, so how is the 1x1 feature map mapped back to the original size?</summary>

1. The parameters of **bilinear interpolation** are used to initialize the parameters of an upsampling inverse (transposed) convolution (referred to as **`deconvolution`**).
2. Nonlinear upsampling is learned through backpropagation.

</details>

<details>
<summary>So what are the three general stages of creating FCNs?</summary>

1. Pick a pre-trained CNN for classification (VGG, AlexNet, etc).
2. Replace the `Linear` layers with `Conv2d`.
3. Upsample the `1 x 1 x C` vector back to the original height and width, **utilizing skip connections in the process**.

</details>

### Upsampling techniques

<details>
<summary>What is their goal?</summary>

- Used between convolutional blocks as an opposite to pooling.
- **Increase height and width while reducing depth.**

</details>

Let's say that we have this feature map:

![w06_bilin_inter.png](assets/w06_bilin_inter.png "w06_bilin_inter.png")

We want to upsample it into a `4x4` feature map.

#### Simplest form - Nearest Neighbor

<details>
<summary>What do you think would be the final result from NN interpolation?</summary>

![w06_nn_inter_result.png](assets/w06_nn_inter_result.png "w06_nn_inter_result.png")

</details>

#### Bilinear interpolation

<details>
<summary>How does bilinear interpolation work visually (from a birds eye view)?</summary>

1. We place the values in the corners of the new map.
2. We create a plane out of those coordinates and look up the values for the missing coordinates.

![w06_bilinear_interpolation.gif](assets/w06_bilinear_interpolation.gif "w06_bilinear_interpolation.gif")

![w06_bilinear_interpolation_result.png](assets/w06_bilinear_interpolation_result.png "w06_bilinear_interpolation_result.png")

Formulas available [in Wikipedia](https://en.wikipedia.org/wiki/Bilinear_interpolation#Repeated_linear_interpolation).

</details>

#### Max Unpooling

<details>
<summary>Do you know how max unpooling works?</summary>

1. We save the coordinates of the maximum values when we do max pooling.
2. We place the input values on these locations.
3. We place `0`s everywhere else.

![w06_max_unpool_res.png](assets/w06_max_unpool_res.png "w06_max_unpool_res.png")

</details>

#### Transposed Convolutions / Inverse Convolutions / Deconvolutions

1. Insert zeros between or around the input feature map.
2. Perform a regular convolution on the zero-padded input.

Regular convolutions:

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="assets/w06_no_padding_no_strides.gif"></td>
    <td><img width="150px" src="assets/w06_arbitrary_padding_no_strides.gif"></td>
    <td><img width="150px" src="assets/w06_no_padding_strides.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides</td>
    <td>Arbitrary padding, no strides</td>
    <td>No padding, strides</td>
  </tr>
</table>

Transposed convolutions:

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td><img width="150px" src="assets/w06_no_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="assets/w06_arbitrary_padding_no_strides_transposed.gif"></td>
    <td><img width="150px" src="assets/w06_no_padding_strides_transposed.gif"></td>
  </tr>
  <tr>
    <td>No padding, no strides, transposed</td>
    <td>Arbitrary padding, no strides, transposed</td>
    <td>No padding, strides, transposed</td>
  </tr>
</table>

This padding logic is a bit confusion, no? Let's explain it in more detail.

<details>
<summary>What did padding mean in the context of regular convolutions?</summary>

Adding zeros around the input image before applying the kernel.

</details>

<details>
<summary>What does padding mean in the context of transposed convolutions given that they are the opposite of regular convolutions?</summary>

Cropping the output edges after the operation has ended.

</details>

<details>
<summary>So how should we interpret the dashed squares / empty cells?</summary>

Note that they are not padding but rather indicate the output locations that haven't received kernel contributions.

Instead of *sliding a kernel across the input* we should think of transposed convolutions as *each input pixel writes a weighted kernel patch into the output grid*.

In pseudocode we'd have this:

```text
output = zeros(big_output_grid)
for each input pixel x[i,j]:
    output[i:i+k, j:j+k] += x[i,j] * kernel
```

So if the input is:

```text
a  b
c  d
```

and we have a kernel of size `3x3`:

```text
k1 k2 k3
k4 k5 k6
k7 k8 k9
```

Then each input pixel produces a `3x3` patch in the output:

- for pixel `a` we'd have:

```text
a*k1 a*k2 a*k3
a*k4 a*k5 a*k6
a*k7 a*k8 a*k9
```

This gets placed into the output grid. Then `b` places another patch shifted right, etc. Where patches overlap, values are summed.

</details>

### Skip connections

Recall our current architecture:

![w06_fcn_paper_p4.png](assets/w06_fcn_paper_p4.png "w06_fcn_paper_p4.png")

<details>
<summary>What would be the result if we apply bilinear interpolation on a vector of size "1 x 1" into a "4 x 4" one?</summary>

It'll actually duplicate the value `16` times.

</details>

<details>
<summary>How do the authors deal with this problem?</summary>

1. They don't actually use pure bilinear interpolation - they use transposed convolutions which learn how to upsample.
   - The link here with the bilinear interpolation is that initally (before training) the parameters of these transposed convolutional layers have the parameters of a bilinear interpolation.
2. They make predictions before every max pooling operation (where the width and height are larger).

</details>

<details>
<summary>How do they do that and what is the goal?</summary>

- They connect them via a skip connection to the output of the upsampling.
- In this way they combine coarse representations (coming from "the deep vector") with finer representations (coming from feature maps that are pooled).
- The goal / added value is **increasing the quality of the segmentation map**.

</details>

<details>
<summary>Open the paper and find the figure with which the authors detail "N" distinct versions of their network - how many are they and what do they comprise of?</summary>

- `FCN-32s`: Single-stream net, described in Section 4.1, upsamples stride `32` predictions back to pixels **in a single step**.
- `FCN-16s`: Combining predictions from both the final layer and the `pool4` layer, at stride `16`.
- `FCN-8s`: Additional predictions from `pool3`, at stride `8`.

![w06_fcn_paper_p7.png](assets/w06_fcn_paper_p7.png "w06_fcn_paper_p7.png")

</details>

<details>
<summary>What operation combines the predictions?</summary>

Addition.

</details>

<details>
<summary>What figure compares the three variants visually?</summary>

![w06_fcn_paper_p8.png](assets/w06_fcn_paper_p8.png "w06_fcn_paper_p8.png")

</details>

So the final model architecture (for `FCN-8` as an example) actually looks something like this:

![w06_fcn_paper_p6.png](assets/w06_fcn_paper_p6.png "w06_fcn_paper_p6.png")

### [Learning Deconvolution Network for Semantic Segmentation](https://www.arxiv.org/pdf/1505.04366)

- Extension of `FCN`.
- An example of the `encoder-decoder` architecture.

<details>
<summary>Open the paper and highlight the two most prominent differences.</summary>

- Does not utilize skip connections.
- Instead it relies on "successive operations of unpooling, deconvolution, and rectification".

![w06_deconv_paper.png](assets/w06_deconv_paper.png "w06_deconv_paper.png")

</details>

<details>
<summary>Which figure presents the architecture?</summary>

![w06_deconv_paper_arch.png](assets/w06_deconv_paper_arch.png "w06_deconv_paper_arch.png")

</details>

### [U-Net](https://arxiv.org/abs/1505.04597)

### U-Net: Architecture

![w06_unet.png](assets/w06_unet.png "w06_unet.png")

- Encoder:
  - Convolutional and pooling layers.
  - Downsampling: reduces spatial dimensions (height, width) while increasing depth.

- Decoder:
  - Symmetric to the encoder.
  - Upsamples feature maps with transposed convolutions.

- Skip connections:
  - Links / **Concatenations** from the encoder to the decoder.
  - Preserve details lost in downsampling.

The full process is therefore the following:

1. Pass input through encoder's convolutional blocks and pooling layers.
2. Decoder and skip connections:
   1. Pass encoded input through transposed convolution.
   2. Concatenate with corresponding encoder output.
   3. Pass through convolutional block.
   4. Repeat.
3. Return output of last decoder step.

![w06_unet_paper.png](assets/w06_unet_paper.png "w06_unet_paper.png")

We can then run inference using our model:

```python
model = UNet()
model.eval()

image = Image.open('car.jpg')
image_tensor = transforms.ToTensor()(image).unsqueeze(0)

with torch.no_grad():
  prediction = model(image_tensor).squeeze(0)

plt.imshow(prediction[1, :, :])
plt.show()
```

![w06_results_unet.png](assets/w06_results_unet.png "w06_results_unet.png")

## Instance segmentation

### [`Mask R-CNN`](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=Mask_RCNN)

- Paper: <https://arxiv.org/abs/1703.06870>.
- Faster R-CNN + FCN.
- Previously we covered Faster R-CNN for object recognition. Given the image, it would predict its class and the bounding box around the object.
  - Two stages:
    1. The `RPN` (Region Proposal Network) proposes candidate object bounding boxes.
    2. The `RoIPool` extracts the features from these boxes by scaling them into a fixed shape.
       - The actual scaling to, e.g., $(7, 7)$, occurs by dividing the region proposal into equally sized sections, finding the largest value in each section, and then copying these max values to the output buffer. In essence, `RoIPool` is max pooling on a discrete grid based on a box.
  - After the features are extracted they can be analyzed using classification and bounding-box regression.

![w06_faster_rcnn.png](assets/w06_faster_rcnn.png "w06_faster_rcnn.png")

![w06_faster_rcnn2.png](assets/w06_faster_rcnn2.png "w06_faster_rcnn2.png")

- Mask R-CNN extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition.
- Done by using a Fully Convolutional Network.
- Easy to generalize to other tasks, e.g., estimating human poses.
- Architecture:
  - Same first stage.
  - The second stage instead of only performing classification and bounding-box regression, also outputs a **binary mask for each RoI**.
    - The final loss is then $L = L_{cls}+L_{box}+L_{mask}$, where:
      - $L_{cls}$: classification loss;
      - $L_{box}$: bounding box loss;
      - $L_{mask}$: average binary cross-entropy loss respectively.

![w06_mask_rcnn.png](assets/w06_mask_rcnn.png "w06_mask_rcnn.png")

#### Pre-trained Mask R-CNN in PyTorch

![w06_cat_and_laptop.png](assets/w06_cat_and_laptop.png "w06_cat_and_laptop.png")

```python
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import detection

model = detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

image = Image.open(os.path.join('assets', 'w06_cat_and_laptop.png')).convert('RGB')
image_tensor = transforms.ToTensor()(image).unsqueeze(0)

with torch.no_grad():
  prediction = model(image_tensor)

print(type(prediction))
print(len(prediction))
print(type(prediction[0]))
print(prediction[0].keys())
```

```console
<class 'list'>
1
<class 'dict'>
dict_keys(['boxes', 'labels', 'scores', 'masks'])
```

- `prediction` is a list of length `1` since we only passed `1` image to the model.

```python
print(prediction[0]['labels'])
print(f'Number of recognized objects: {len(prediction[0]['labels'])}')
```

```console
tensor([17, 73, 76, 73, 67, 84, 42, 65, 63, 73, 73, 32, 31, 73, 62, 17, 76, 84, 17])
Number of recognized objects: 19
```

- `labels` contains the class IDs of recognized objects:
  - The IDs correspond to [the COCO dataset classes](https://cocodataset.org/#home) which we have stored in the variable class_names.
  - These class names are available [here](https://gist.github.com/tersekmatija/9d00c4683d52d94cf348acae29e8db1a).
  - We can see that the top two predicted classes with indices `17` and `73` correspond to a `cat` and a `laptop`, respectively.
    - `(1, 'person'), (2, 'bicycle'), (3, 'car'), (4, 'motorbike'), (5, 'aeroplane'), (6, 'bus'), (7, 'train'), (8, 'truck'), (9, 'boat'), (10, 'trafficlight'), (11, 'firehydrant'), (12, 'streetsign'), (13, 'stopsign'), (14, 'parkingmeter'), (15, 'bench'), (16, 'bird'), (17, 'cat'), (18, 'dog'), (19, 'horse'), (20, 'sheep'), (21, 'cow'), (22, 'elephant'), (23, 'bear'), (24, 'zebra'), (25, 'giraffe'), (26, 'hat'), (27, 'backpack'), (28, 'umbrella'), (29, 'shoe'), (30, 'eyeglasses'), (31, 'handbag'), (32, 'tie'), (33, 'suitcase'), (34, 'frisbee'), (35, 'skis'), (36, 'snowboard'), (37, 'sportsball'), (38, 'kite'), (39, 'baseballbat'), (40, 'baseballglove'), (41, 'skateboard'), (42, 'surfboard'), (43, 'tennisracket'), (44, 'bottle'), (45, 'plate'), (46, 'wineglass'), (47, 'cup'), (48, 'fork'), (49, 'knife'), (50, 'spoon'), (51, 'bowl'), (52, 'banana'), (53, 'apple'), (54, 'sandwich'), (55, 'orange'), (56, 'broccoli'), (57, 'carrot'), (58, 'hotdog'), (59, 'pizza'), (60, 'donut'), (61, 'cake'), (62, 'chair'), (63, 'sofa'), (64, 'pottedplant'), (65, 'bed'), (66, 'mirror'), (67, 'diningtable'), (68, 'window'), (69, 'desk'), (70, 'toilet'), (71, 'door'), (72, 'tvmonitor'), (73, 'laptop'), (74, 'mouse'), (75, 'remote'), (76, 'keyboard'), (77, 'cellphone'), (78, 'microwave'), (79, 'oven'), (80, 'toaster'), (81, 'sink'), (82, 'refrigerator'), (83, 'blender'), (84, 'book'), (85, 'clock'), (86, 'vase'), (87, 'scissors'), (88, 'teddybear'), (89, 'hairdrier'), (90, 'toothbrush'), (91, 'hairbrush')`

```python
prediction[0]['scores']
```

```console
tensor([0.9990, 0.9391, 0.8823, 0.7186, 0.3869, 0.3535, 0.2245, 0.2066, 0.2040,
        0.1558, 0.1145, 0.1007, 0.0870, 0.0849, 0.0788, 0.0737, 0.0660, 0.0631,
        0.0615])
```

- The `scores` key stores the class probabilities:
  - `cat` was detected with a probability of `0.9990`;
  - `laptop` with a probability of `0.9391`
  - The following values correspond to other, less probable classes.

```python
prediction[0]['masks'].shape
```

```console
torch.Size([19, 1, 395, 529])
```

- `masks` stores **instance segmentation *soft* masks** which we will look at next.

```python
prediction[0]['boxes'].shape
```

```console
torch.Size([19, 4])
```

- The Mask R-CNN prediction also contains bounding boxes. We won't be focusing on them at the moment.

#### Soft masks

- Let's see the unique mask values:

```python
print(prediction[0]['masks'].unique())
print(f"Min: {prediction[0]['masks'].min()}")
print(f"Max: {prediction[0]['masks'].max()}")
```

```console
tensor([0.0000e+00, 4.1547e-13, 4.1568e-13,  ..., 9.9996e-01, 9.9996e-01,
        9.9996e-01])
Min: 0.0
Max: 0.9999570846557617
```

- Mask R-CNN masks:
  - hold values between `0` and `1`;
  - represent the model's confidence that each pixel belongs to the object;
  - provide more nuanced information than binary masks;
  - can be binarized by thresholding if needed.

- We can display them as transparent elements of the colormap [`jet`](https://matplotlib.org/stable/users/explain/colors/colormaps.html#miscellaneous).

```python
masks = prediction[0]['masks']
labels = prediction[0]['labels']

fig, axs = plt.subplots(1, 2)

for i in range(2):
  axs[i].axis('off')
  axs[i].imshow(image)
  axs[i].imshow(
    masks[i, 0],
    cmap='jet',
    alpha=0.5,
  )
  axs[i].set_title(f'Object: {class_names[labels[i] - 1]}') # class_names is the list with the COCO labels that you saw earlier (without enumeration)

plt.show()
```

![w06_objects.png](assets/w06_objects.png "w06_objects.png")

## Panoptic segmentation

<details>
<summary>What was panoptic segmentation again?</summary>

Segment an image into semantically meaningful parts or regions, while also detecting and distinguishing individual instances of objects within those regions.

![w06_panoptic_seg_example.png](assets/w06_panoptic_seg_example.png "w06_panoptic_seg_example.png")

- Semantic segmentation doesn't allow us to distinguish between particular cabs.
- Instance segmentation loses the distinction between the street and the buildings which are now combined together as one background.

</details>

<details>
<summary>How can we do panoptic segmentation using what we already know?</summary>

1. Generate a semantic mask with a U-Net. It'll hold the most likely class for each pixel.
2. Generate instance masks with a Mask R-CNN model.
3. Iterate over the instance masks:
   - if an object is detected with high certainty (for ex. above `> 0.5`), overlay it onto the semantic mask.

Obtaining the semantic mask could be done as follows:

```python
model = UNet()
with torch.no_grad():
  semantic_masks = model(image_tensor)
  print(semantic_masks.shape)
```

```console
torch.Size([1, 3, 427, 640])
```

From the shape of the output, we can see that there are `3` classes, likely corresponding to the cars, the buildings, and the street.

Choosing the highest-probability class for each pixel produces the following image:

```python
semantic_mask = torch.argmax(semantic_masks, dim=1)

plt.imshow(semantic_mask)
plt.show(semantic_mask)
```

![w06_sem_seg_results_pan_opt.png](assets/w06_sem_seg_results_pan_opt.png "w06_sem_seg_results_pan_opt.png")

Continuing on with the instance mask:

```python
model = MaskRCNN()

with torch.no_grad():
  instance_masks = model(image_tensor)[0]['masks']
  print(instance_masks.shape)
```

```console
torch.Size([80, 1, 427, 640])
```

The model has identified `80` different instance classes:

![w06_inst_seg_results_pan_opt.png](assets/w06_inst_seg_results_pan_opt.png "w06_inst_seg_results_pan_opt.png")

Finally, we can overlay them:

```python
panoptic_mask = torch.clone(semantic_mask)

# We know that the semantic mask has three classes (cars, street, and buildings) which means it takes the values of 0, 1, and 2.
# Therefore, we will start labeling the instance classes starting from 3 to avoid collisions.
instance_id = 3

for mask in instance_masks:
  panoptic_mask[mask > 0.5] = instance_id
  instance_id += 1
```

</details>
