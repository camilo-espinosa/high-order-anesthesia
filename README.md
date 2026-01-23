
## Setup
Clone this repository:
```bash
git clone https://github.com/camilo-espinosa/high-order-anesthesia.git
cd high-order-anesthesia
```
Install dependencies in requirements.txt

```bash
pip install -r requirements.txt
```
A version of PyTorch, with CUDA compatibility is also necessary to use GPU: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Usage examples (Notebooks):

### Result 1A - Find optimal nplets for C/NR scan pairs: 
[R1_A_nplet_from_pairs.ipynb](https://github.com/camilo-espinosa/high-order-anesthesia/blob/main/notebooks/R1/R1_A_nplet_from_pairs.ipynb)

### Result 1B - Evaluate discovered nplets on the datasets: 
[R1_B_nplet_evaluation.ipynb](https://github.com/camilo-espinosa/high-order-anesthesia/blob/main/notebooks/R1/R1_B_nplet_evaluation.ipynb)
