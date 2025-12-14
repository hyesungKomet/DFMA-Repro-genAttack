# Model stealing attack against skin cancer task
## About
Similar to the mnist task, open-source LLMs can not generate benign/malignant skin images.
Thereby, we train a generative adversarial network as the large model to steal the data distribution of (small) victim models.

## The construction of generated dataset

### Step 0: Prepare the victim model
```
python3 train_victim_model.py
```
After execution, you get the victim model 'victim_model.pt'.

### Step 1: Prepare the generative model
```
python3 train_large.py
```
After execution, you get two generative models, i.e., 'netG_malignant.pt' and 'netG_benign.pt'.
The first model can craft malignant skin images, and the second model can craft benign skin images.

### Step 2: Generate data points with augmentation & apply filtering
```
python3 create_stealing_set.py
```
After execution, you train an autoencoder as the enhancement model.
Then, you can obtain the generated dataset `./stealing_set_GAN/generated_data.pt` and `./stealing_set_GAN/generated_label.pt`

## Three privacy attacks based on the generated dataset

### Attack 0: model extraction
```
python3 model_extraction_attack_main.py
```
After execution, you get the stealing model on `./results/vgg16_extraction_GAN_filtered`

### Attack 1: membership inference
```
python3 membership_inference_attack_main.py
```
After execution, you get the meta-attacker model 'meta-attacker.pt'.

### Attack 2: model inversion
```
python3 model_inversion_attack_main2.py
```
After execution, you get the inversion model 'inversion.pt'.

## Note
We consider two baselines, the first using the test set of skin cancer and the second using a noise data set.
Following the code above, you can implement all three attacks using baseline data.