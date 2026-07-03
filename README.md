# DL‑GenAI 30 Days

Hands‑on deep learning and generative AI sprint.
Daily build tasks, FastAPI serving path, 3‑hour blocks.

## Run the app (portal + interactives)

From the repo root:

```bash
python3 scripts/serve.py
```

This starts a local server and opens **http://localhost:8080/** in your browser.

| Page | URL |
|------|-----|
| **Home portal** | http://localhost:8080/ |
| **Interactives hub** | http://localhost:8080/interactives/ |
| **Inference Engineering course** | http://localhost:8080/interactives/inference-engineering/ |

No extra dependencies — just Python 3. Use `--no-open` if you don't want the browser to launch automatically:

```bash
python3 scripts/serve.py --no-open -p 8080
```

## Structure

```
├── index.html                          # Home portal (start here)
├── interactives/
│   ├── index.html                      # Interactives hub
│   └── inference-engineering/          # 12-module canvas course
├── src/                                # MNIST training code
├── scripts/                            # lr_finder, serve.py
├── notes/                              # Daily reflections
├── notebooks/
└── tests/
```

## Inference Engineering course

A browser-based interactive course (12 modules) covering LLM serving from first principles through advanced techniques: prefill vs decode, roofline model, KV cache, continuous batching, PagedAttention, quantization, speculative decoding, sampling, multi-GPU parallelism, and a capstone latency lab.

Open it from the home portal or go directly to `interactives/inference-engineering/` after starting the server.
