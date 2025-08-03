## Day 2 — Manual gradients

### ✅ Gradient check
| Param | Relative diff |
|-------|---------------|
| W1 | 8.49e-08 |
| b1 | 9.75e-08 |
| W2 | 7.95e-08 |
| b2 | 1.12e-07 |

### 💡 One key thing I learned
<e.g. “Softmax-cross-entropy simplifies gradient to probs − one-hot.”>

### ⚠️ One roadblock / surprise
<e.g. “`nn.Sequential` needs an nn.Module, so I had to wrap MyReLU.”>

### 🤔 Question to revisit
<e.g. “Why does He init use √2/ fan_in for ReLU?”>

### 📈 Loss plot saved
Manual training loss curve: `manual_loss.png`
