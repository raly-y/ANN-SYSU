# ArtificialNeuralNetworks

This is a Chinese-English neural machine translation project based on PyTorch. It uses a Seq2Seq model with attention mechanism. The project supports both training and inference modes and is designed for Chinese-to-English translation tasks.

## Project Structure

```
src/               # Core code directory
├── main.py        # Entry point; supports training and inference
├── train.py       # Training module
├── inference.py   # Inference module
├── model.py       # Model definition
├── data_preprocess.py # Data preprocessing
└── utils.py       # Utility functions

data/              # Data directory
├── train_10k.jsonl        # Training data
├── valid.jsonl            # Validation data
├── test.jsonl             # Test data
├── fasttext.cc.zh.300.vec # Pretrained Chinese word vectors
└── glove.6B.300d.txt      # Pretrained English word vectors

outputs/           # Output directory
├── vocab/         # Vocabulary files
├── models/        # Model checkpoints
├── plots/         # Attention heatmaps
├── logs/          # Training logs
└── translations/  # Translation results

config.yaml        # Configuration file
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Jieba

## Usage

### 1. Modify the configuration file

You can adjust parameters by editing `config.yaml`. Key parameters include:

#### `main.mode`: Running mode

- `train`: training only  
- `inference`: inference only  
- `both`: train then infer  

#### `model`: Model parameters

- `hidden_size`: Hidden layer size (default: `512`)  
- `dropout`: Dropout rate (default: `0.5`)  
- `attention_type`: Type of attention mechanism (`dot_product`, `multiplicative`, or `additive`; default: `dot_product`)

#### `data`: Data paths

- `train_path`: Path to training data  
- `valid_path`: Path to validation data  
- `test_path`: Path to test data  

#### `inference`: Inference parameters

- `decode_type`: Decoding strategy (`greedy` or `beam_search`; default: `beam_search`)  
- `beam_size`: Beam search size (default: `5`)  
- `num_visualize`: Number of samples for attention heatmap visualization (default: `3`)  
- `model_path`: Path to the pretrained model (required for inference mode)

### 2. Run the program

Run `main.py` directly. Depending on the `mode` setting in `config.yaml`, you can choose the following operations:

#### 2.1 Inference using a pretrained model

- Ensure there is a pretrained model in `outputs/models/` (e.g., `model_1_3_1.pt`)  
- Set `config.yaml`:

```yaml
main:
  mode: inference
inference:
  model_path: outputs/models/model_1_3_1.pt
  decode_type: "beam_search"
```

- Run:

```bash
python src/main.py
```

- Output:

  - Translation results are saved to `outputs/translations/`  
  - Attention heatmaps are saved to `outputs/plots/`  
  - Console displays average BLEU and best BLEU scores  

#### 2.2 Train a new model

- Set `config.yaml`:

```yaml
main:
  mode: train
train:
  epochs: 100
  learning_rate: 0.0005
  train_type: "mix_training"
```

- Run:

```bash
python src/main.py
```

- Output:

  - Model weights saved to `outputs/models/`  
  - Training logs saved to `outputs/logs/`
  - BLEU curve saved to `outputs/plots/`
  - Loss curve saved to `outputs/plots/`
  - Vocabulary saved to `outputs/vocab/`  

#### 2.3 Train then infer

- Set `config.yaml`:

```yaml
main:
  mode: both
```

- Run:

```bash
python src/main.py
```

- Output:

  - Inference automatically follows training  
  - Output same as above  

## Notes

- Ensure that data files (`train_10k.jsonl`, `valid.jsonl`, `test.jsonl`) and pretrained word vectors (`fasttext.cc.zh.300.vec`, `glove.6B.300d.txt`) exist. `fasttext.cc.zh.300.vec` and `glove.6B.300d.txt` need to be downloaded separately.
- Ensure that the model path in `config.yaml` is which exactly you want to use for inference after training. Because there will be different models after training with different parameters, you need to specify the correct one for inference
- Ensure you have a GPU, otherwise the training will be slow. If you don't have a GPU, you can still run inference on CPU, but it will be slower


