# How to run the code
```python
# Create conda environment from the provided file
conda env create -f env.yml
```
```python
# Train the model
python train.py # To train the model built from scratch
python train_transferlearning.py # To train the pre-trained model and fine-tune it according to your data
```
```python
# To see and plot the loss curve (in notebook)
from functions import helper_functions

with open('pickle file path', 'rb') as f:
  results = pd.read_pickle(f)

helper_functions.plot_loss_curves(results)
```
# ViT: Vision Transformer for Early Dementia Detection
An attempt to replicate a Vision Transformer Architecture from the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929v2.pdf)

## [Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset)

## Model Architecure

<img width="724" alt="08-vit-paper-figure-1-architecture-overview" src="https://github.com/fbrynpk/Vision-Transformer/assets/85217844/0a1a54c6-ff92-4f16-ba37-86030834bda5">

**What is a Transformer Encoder?**

The Transformer Encoder is a combination of alternating blocks of MSA (Equation 2) and MLP (Equation 3)

And there are residual connections between each block

* Encoder = turn a sequnce into learnable representation
* Decode = go from learned representation back to some sort of sequence
* Residual/Skip connections = add a layer(s) input to its subsequent output, this enables the creation of deeper networks since it prevents the weights from getting too small

In Pseudocode:
```python
# Transformer Encoder
x_input -> MSA Block -> [MSA_block_output + x_input] -> MLP_block -> [MLP_block_output + MSA_block_output + x_input] -> ...
```

## Step Breakdown

* **Inputs** - What goes into the model? (In this case, image tensors)
* **Outputs** - What comes out of the model/layer/block? (In this case, we want the model to output image classification labels)
* **Layers** - Takes an input, manipulates it with a function (e.g. Self-attention)
* **Blocks** - A collection of layers.
* **Model** - A collection of blocks.

## Equations to understand
<img width="708" alt="08-vit-paper-four-equations" src="https://github.com/fbrynpk/Vision-Transformer/assets/85217844/5cbc555f-300e-4ca6-899a-119155a5ad6b">

1. Equation 1 (In Pseudocode)
```python
x_input = [class_token, image_patch_1, image_patch_2, ... image_patch_N] + [class_token_pos, image_patch_1_pos, image_patch_2_pos, ... image_patch_N_pos]
```
2. Equation 2 (In Pseudocode)
```python
x_output_MSA_block = MSA_layer(LN_layer(x_input)) + x_input
```
3. Equation 3 (In Pseudocode)
```python
x_output_MLP_block = MLP_layer(LN_layer(x_output_MSA_block)) + x_output_MSA_block
```
4. Equation 4 (In Pseudocode)
```python
y_output = Linear_layer(LN_layer(x_output_MLP_block))
```

## Vision Transformer Model Variants
<img width="604" alt="08-vit-paper-table-1" src="https://github.com/fbrynpk/Vision-Transformer/assets/85217844/90fb44cd-4b3f-4c4b-9523-7167f47444a7">
<img width="951" alt="Screenshot 2023-09-17 at 2 30 03 AM" src="https://github.com/fbrynpk/Vision-Transformer/assets/85217844/33fd773b-7a69-41e1-b300-3b892605327b">


* This repo will mainly cover how to recreate the ViT-Base model variant but could also be adjusted as needed to replicate ViT-Large or ViT-Huge (Hyperparameter can be adjusted by ourselves)
  * ViT-Base, ViT-Large, ViT-Huge are all different sizes of the same model architecture
  * ViT-B/16 - ViT-Base with patch size of 16 x 16
  * Layers - the number of transformer encoder layers
  * Hidden size $D$ - the embedding size throughout the architecture
  * MLP size - the number of hidden units/neurons in the MLP
  * Head - the number of multi-head-self-attention

# Understanding MSA (Multihead Self-Attention)
<img width="951" alt="Screenshot 2023-09-17 at 2 30 14 AM" src="https://github.com/fbrynpk/Vision-Transformer/assets/85217844/5cb23b87-7b88-40f0-b9d6-afc245cd1259">

* Multihead Self-Attention: Which part of a sequence should pay the most attention to itself?
  * In this case, we have a series of embedded image patches, which patch significantly relates to another patch.
  * We want our neural network (ViT) to learn this relationship/representation
* To replicate MSA in PyTorch we can use the built-in Multihead Self-Attention function from PyTorch `torch.nn.MultiheadAttention()`

* LayerNorm = A technique to normalize the distributions of intermediate layers. It enables smoother gradients, faster training, and better generalization accuracy.
  * Normalization = Make everything have the same mean and the same standard deviation
  * In PyTorch: Normalizes values over $D$ dimension, in this case, the $D$ dimension is the embedding dimension, which is 768
    * When we normalize along the embedding dimension, it's like making all of the stairs in a staircase the same size. (In the sense that it is faster and easier to go down a staircase of the same shape and size, width and height compared to             different shape and sizes of a staircase

# Understanding MLP (Multilayer Perceptron)
* **MLP** = The MLP contains two layers with a GELU non-linearity (section 3.1)
  * MLP = a quite broad term for a block with a series of layer(s), layers can be multiple or even only one hidden layer
  * Layers can mean: fully-connected, dense, linear, feed-forward, all are often similar names for the same thing. In PyTorch, they're often called `torch.nn.Linear()` and in TensorFlow they might be called `tf.keras.layers.Dense()`
  * MLP number of hidden units = MLP Size in Tabel 1, which is 3072

* **Dropout** = Dropout, when used, is applied after
every dense layer except for the the qkv-projections and directly after adding positional- to patch
embeddings. Hybrid models are trained with the exact setup as their ViT counterparts.
  * Value for Dropout available in Table 3

In Pseudocode:
```python
# MLP
x = linear -> non-linear -> dropout -> linear -> dropout


