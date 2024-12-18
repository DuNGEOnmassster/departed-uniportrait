# departed-uniportrait
Unofficial reproduction of [uniportrait](https://github.com/junjiehe96/UniPortrait.git), take everything apart for research usage.

## Preparation
```
# Clone repository
git clone https://github.com/DuNGEOnmassster/departed-uniportrait.git

# Install requirements
pip install -r requirements.txt

# Download pre-trained model
sh download_weights.sh
#

```

## Usage
```
# Run text-to-image with single ID
sh run_single_id.sh

# Run text-to-image with multiple IDs
sh run_multi_id.sh

# Run stylized image generation with single ID
sh run_stylize.sh
```

This code is created from the source code of [uniportrait](https://github.com/junjiehe96/UniPortrait.git) and modified the structure in clean weights&requirements management for research usage, which also takes gradio apart into separate modules for easier understanding and modification.