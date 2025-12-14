# Model stealing attack against cifar10 task
## About
We employ [Stable Diffusion XL](https://huggingface.co/spaces/google/sdxl) to generate various objects (500 images), and then steal the data distribution of (small) victim models.
Specifically, the following instruction is repeated several times.

Also, for extension, we employed Stable Diffusion 1.5 to generate objects.
```
Prompt:
Just a single dog/cat/bird/.../airplane; Realistic style; Clear background
```
## The construction of generated dataset

### Step 0: Prepare the victim model
```
python3 train_victim_model.py --model resnet18 | vgg16
```
After execution, you get the victim model in `./results/{model}_victim`

### Step 1: Prepare the enhancement model
You can download an autoencoder, or train an autoencoder on the emnist, thus getting 'autoencoder.pt'.

### Step 2: Generate data points with augmentation
```
python3 create_stealing_set.py --victim-arch vgg16 --gen-model SD1.5 --save-images
```
After execution, you get the generated dataset 'generated_data.pt' and 'generated_label.pt' in `./stealing_set/{gen_model}/{victim_arch}_{augment}`

## Three privacy attacks based on the generated dataset

### Attack 0: model extraction
```
python3 model_extraction_attack_main.py --model vgg16 gen-model SD1.5
```
After execution, you get the stealing model in `./results/{model}_extraction_{gen_model}_{augment}`

### Attack 1: membership inference
```
python3 membership_inference_attack2.py
```
After execution, you get the meta-attacker model 'meta-attacker.pt'.

### Attack 2: model inversion
```
python3 model_inversion_attack_real.py
```
After execution, you get the inversion model 'inversion.pt'.

### Visualize PCA
```
python3 visualize_pca.py
```
By running it, you can get the distribution of dataset using PCA and projection. You can compare the distribution of original data and synthetic data.

## Note
We consider two baselines, the first using the test set of cifar10 and the second using a noise data set.
Following the code above, you can implement all three attacks using baseline data.