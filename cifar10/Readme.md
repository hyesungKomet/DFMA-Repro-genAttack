# Model stealing attack against cifar10 task
## About
We employ [Stable Diffusion XL](https://huggingface.co/spaces/google/sdxl) to generate various objects (500 images), and then steal the data distribution of (small) victim models.
Specifically, the following instruction is repeated several times.
```
Prompt:
Just a single dog/cat/bird/.../airplane; Realistic style; Clear background
```
## The construction of generated dataset

### Step 0: Prepare the victim model
```
python3 train_victim_model.py
```
After execution, you get the victim model 'victim_model.pt'.

### Step 1: Prepare the enhancement model
You can download an autoencoder, or train an autoencoder on the emnist, thus getting 'autoencoder.pt'.

### Step 2: Generate data points with augmentation
```
python3 generate_data.py
```
After execution, you get the generated dataset 'generated_data.pt' and 'generated_label.pt'.

## Three privacy attacks based on the generated dataset

### Attack 0: model extraction
```
python3 model_extraction_attack_main.py
```
After execution, you get the stealing model 'steal_model.pt'.

### Attack 1: membership inference
```
python3 membership_inference_attack_main.py
```
After execution, you get the meta-attacker model 'meta-attacker.pt'.

### Attack 2: model inversion
```
python3 model_inversion_attack_main.py
```
After execution, you get the inversion model 'inversion.pt'.

## Note
We consider two baselines, the first using the test set of cifar10 and the second using a noise data set.
Following the code above, you can implement all three attacks using baseline data.