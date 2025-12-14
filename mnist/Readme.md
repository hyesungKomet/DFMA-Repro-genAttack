# Model stealing attack against mnist task
## About
Open-source LLMs can not generate hand-writing digits, thereby we train a generative adversarial network on the emnist dataset to steal the data distribution of (small) victim models.

## The construction of generated dataset

### Step 0: Prepare the victim model
```
python3 train_victim_model.py
```
After execution, you get the victim model on `./results/cnn`.

### Step 1: Prepare the (large) generative model
```
python3 train_large_model.py
```
After execution, you get the generative model 'last_model.pt'.

### Step 2: Prepare the enhancement model
```
python3 enhancement_model.py
```
You can download an autoencoder, or train an autoencoder on the emnist, thus getting 'autoencoder.pt'.

### Step 3: Generate data points with augmentation & filtering
```
python3 create_stealing_set.py
```
After execution, you get the generated dataset 'generated_data.pt' and 'generated_label.pt' in `./stealing_set_GAN`.

## Three privacy attacks based on the generated dataset

### Attack 0: model extraction
```
python3 model_extraction_attack_main.py
```
After execution, you get the stealing model in `./results/cnn`

### Attack 1: membership inference
```
python3 membership_inference_attack_main.py
```
After execution, you get the meta-attacker model 'meta-attacker.pt'.

### Attack 2: model inversion
```
python3 model_inversion_attack_main3.py
```
After execution, you get the inversion model 'inversion.pt'.

## Note
We consider two baselines, the first using the test set of mnist and the second using a noise data set.
Following the code above, you can implement all three attacks using baseline data.