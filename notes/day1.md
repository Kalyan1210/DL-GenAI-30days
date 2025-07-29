### Day 1 Reflection

* Achieved 97.67 % val accuracy in 5 epochs
* Learned: tensors, autograd, basic training loop
* Roadblocks: initial import path and pre‑commit flow
* Next: manual back‑prop sanity check

## Day 1 — Environment + 2‑Layer MLP

### 🚀 What I built
- Two‑layer MLP on MNIST (`src/train_mnist.py`, `src/models/mlp.py`)
- Achieved **97.67 %** validation accuracy in 5 epochs on ✅ device: <mps/cpu/cuda>

### 🧠 Key concepts learned
- Dynamic computation graph & leaf tensors
- Forward‑backward‑update training loop
- Device management (`mps`, `cuda`, `cpu`)
- Data pipeline with `torchvision.datasets` + `DataLoader`

### 🛠️ Roadblocks / fixes
| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: models` | Added `src/__init__.py` and correct import path |
| Pre‑commit aborts | Staged rewritten files, reran commit |
| Large dataset push | Added `data/` to `.gitignore`, rewrote commit |

### ❓ Open questions
- Why orthogonal + gain = √2 is best for ReLU?
- When to switch from SGD to Adam in practice?

### 📌 TODO tomorrow
- Manual back‑prop sanity check (Day 2 task)
- Try LR finder script to verify `1e‑3` is stable
