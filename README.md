# Dataset

Put the "lego" under ./data/ or anywhere you want and change the paths in the config files.

# Train

python train.py --opt ./configs/config.yml

# Test

python test.py --opt ./configs/config.yml

# Object editing

python move.py --opt ./configs/config.yml

# Main components

Our main components are in the folder "models". The class `VolumetricBank` defined in models/vabank.py is our proposed model. The `VolumetricBank` contains the following key components:

- RGB MLP: defined in models/vabank.py line 52, as `self.norm_mlp`
- Attention MLP: defined in models/vabank.py line 51, as `self.att_mlp`
- Transformer: defined in models/vabank.py line 68, as `self.transformer`

The ray-point embedding is defined in models/vabank.py line 155, where we use ray marching as described in the report to get the embedding.


