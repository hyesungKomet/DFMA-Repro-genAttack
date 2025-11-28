# Model stealing attack against mnist task
## About
Open-source LLMs can not generate hand-writing digits, thereby we train a generative adversarial network on the emnist dataset to steal the data distribution of (small) victim models.

## The construction of generated dataset

### Step 0: Prepare the victim model
```
python3 train_victim_model.py
```
After execution, you get the victim model 'victim_mnist_model.pt'.

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

### Step 3: Generate data points with augmentation
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
We consider two baselines, the first using the test set of mnist and the second using a noise data set.
Following the code above, you can implement all three attacks using baseline data.