## Day 3 — Gradient tests & LR exploration

### ✅ Best LR from range test
Chosen LR (for SGD) : **0.05**  
Divergence started around : **0.7**

### 📊 Cosine vs constant LR (Adam, 5 epochs)
| Schedule | Final val acc |
|----------|---------------|
| Constant | 95.46 % |
| Cosine   | 97.24 % |

### 🧠 Insight
Loss curve showed the “golden” region just before 0.1; cosine schedule gave steadier improvements after epoch 3 without manual LR drops.

### ❓ Question
Would a one-cycle policy outperform cosine on this small model?
