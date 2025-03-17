<div align="center">
<img src="https://www.cdeng.net/plm/plm_logo.png" alt="k2-logo" width="200"/>
<h2>🖲️ PLM: Efficient Peripheral Language Models Hardware-Co-Designed for Ubiquitous Computing</h2>

<a href='https://www.project-plm.com/'>👉 Project PLM Website</a>

<a href='https://arxiv.org/abs/'><img src='https://img.shields.io/badge/Paper-ArXiv-C71585'></a>
<a href='https://huggingface.co/PLM-Team/PLM-1.8B-Base'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-Base-red'></a>
<a href='https://huggingface.co/PLM-Team/PLM-1.8B-Instruct'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-Instruct-red'></a>
<a href='https://huggingface.co/PLM-Team/PLM-1.8B-Instruct-gguf'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging Face-gguf-red'></a>
<a href='https://huggingface.co/datasets/plm-team/scots'><img src='https://img.shields.io/badge/Data-plm%20mix-4169E1'></img></a>
<a><img src="https://img.shields.io/github/stars/plm-team/PLM"></a>
</div>

---

The PLM (Peripheral Language Model) series introduces a novel model architecture to peripheral computing by delivering powerful language capabilities within the constraints of resource-limited devices. Through modeling and system co-design strategy, PLM optimizes model performance and fits edge system requirements, PLM employs **Multi-head Latent Attention** and **squared ReLU** activation to achieve sparsity, significantly reducing memory footprint and computational demands. Coupled with a meticulously crafted training regimen using curated datasets and a Warmup-Stable-Decay-Constant learning rate scheduler, PLM demonstrates superior performance compared to existing small language models, all while maintaining the lowest activated parameters, making it ideally suited for deployment on diverse peripheral platforms like mobile phones and Raspberry Pis.

---
## News

> The paper **"PLM: Efficient Peripheral Language Models Hardware-Co-Designed for Ubiquitous Computing"** has been released!

## PLM Roadmap

<center>
    <img src="https://www.cdeng.net/plm/pipe.png" width="100%"/>
</center>

## PLM Hightlight

PLM demonstrates highly competitive performance along with a series of advantages stemming from its modeling and system co-design. These benefits include impressive inference speed, extreme sparsity, and reduced KV cache due to MLA, enabling it to outperform models with the same number of layers when handling long-context inference tasks at certain sequence lengths.


- **Sparse** (Less activated parameters but better performance)

<div align="center">
<img src="https://www.cdeng.net/plm/sparse_compare.png" width="50%"/>
</div>

- **High efficiency** (Generate content with low latency while having a good quality)

<center>
    <img src="https://www.cdeng.net/plm/latency/latency_all.png" width="100%"/>
</center>

- **Low kv-cache** on long-context processing leads to a low latency when inference with long sequences.

|||
|:-:|:-:|
|<img src="https://www.cdeng.net/plm/latency/prefill_eff.png"/>|<img src="https://www.cdeng.net/plm/latency/decode_eff.png"/>|

- **More efficiency** when layer-wise loading.

|||
|:-:|:-:|
|<img src="https://www.cdeng.net/plm/latency/prefill_ngl.png"/>|<img src="https://www.cdeng.net/plm/latency/decode_ngl.png"/>|

## Performance

PLM-1.8B is a strong and reliable model, particularly in basic knowledge understanding, coding and simple reasoning tasks.

<center>

| **Benchmarks** | PLM-Instruct | MiniCPM | Yulan-Mini | SmolLM2 | Qwen2.5 | Qwen2 | GLM-Edge |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **ARC-C** | $\underline{\text{51.14}}$ | 43.86 | 50.51 | 50.29 | **53.41** | 43.90 | 24.15 |
| **ARC-E** | $\underline{\text{78.18}}$ | 55.51 | 69.87 | 77.78 | **79.13** | 62.21 | 36.83 |
| **MMLU** | 51.18 | 51.13 | 49.10 | 51.91 | **59.79** | $\underline{\text{56.50}}$ | 54.84 |
| **CMMLU** | 48.18 | 48.97 | 48.35 | 33.46 | $\underline{\text{67.82}}$ | **70.30** | 54.23 |
| **C-Eval** | 44.93 | 48.24 | 51.47 | 35.10 | $\underline{\text{69.05}}$ | **70.60** | 55.05 |
| **GSM8K** | 60.73 | 53.83 | $\underline{\text{66.65}}$ | 47.68 | **68.50** | 46.90 | 54.89 |
| **MathQA** | 33.23 | 30.59 | $\underline{\text{34.84}}$ | 34.30 | **35.14** | 31.66 | 33.94 |
| **HumanEval** | **64.60** | 50.00 | $\underline{\text{61.60}}$ | 23.35 | 37.20 | 34.80 | 1.21 |
| **MBPP** | $\underline{\text{60.40}}$ | 47.31 | **66.70** | 45.00 | 60.20 | 46.90 | 3.44 |
| **BoolQ** | $\underline{\text{77.86}}$ | 73.55 | 70.89 | 72.26 | 72.91 | 72.69 | 60.95 |
| **Hellaswag** | 68.17 | 53.06 | $\underline{\text{71.47}}$ | **71.48** | 67.73 | 65.41 | 29.39 |
| **LogiQA** | 30.12 | **31.64** | 29.65 | 29.65 | $\underline{\text{31.03}}$ | 31.02 | 22.73 |
| **PIQA** | 76.01 | 77.04 | 76.50 | 77.04 | **76.01** | $\underline{\text{75.35}}$ | 74.32 |
| **Average** | **57.29 (3rd)** | 51.13 | **57.51 (2nd)** | 49.95 | **59.84 (1st)** | 54.48 | 38.92 |

</center>

## How to use PLM

Here we introduce some methods to use PLM models.

### Hugging Face

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("PLM-Team/PLM-1.8B-Instruct")
model = AutoModelForCausalLM.from_pretrained("PLM-Team/PLM-1.8B-Instruct", torch_dtype=torch.bfloat16)

# Input text
input_text = "Tell me something about reinforcement learning."
inputs = tokenizer(input_text, return_tensors="pt")

# Completion
output = model.generate(inputs["input_ids"], max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### llama.cpp

The original contribution to the llama.cpp framwork is [Si1w/llama.cpp](https://github.com/Si1w/llama.cpp). Here is the usage:

```bash
git clone https://github.com/Si1w/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

Then, we can build with CPU of GPU (e.g. Orin). The build is based on `cmake`. 

- For CPU

```bash
cmake -B build
cmake --build build --config Release
```

- For GPU

```bash
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

Don't forget to download the GGUF files of the PLM. We use the quantization methods in `llama.cpp` to generate the quantized PLM.

```bash
huggingface-cli download --resume-download PLM-Team/PLM-1.8B-Instruct-gguf --local-dir PLM-Team/PLM-1.8B-Instruct-gguf
```

After build the `llama.cpp`, we can use `llama-cli` script to launch the PLM.

```bash
./build/bin/llama-cli -m ./PLM-Team/PLM-1.8B-Instruct-gguf/PLM-1.8B-Instruct-Q8_0.gguf -cnv -p "hello!" -n 128
```

## Future works

- [ ] Release vLLM, SGLang, and PowerInfer inference scripts for PLM.
- [ ] Release reasoning model trained on PLM.
- [ ] Release vision model based on PLM.

## Acknowledgements

We sincerely thank Deepseek for its contributions to the community through the MLA architecture and the PowerInfer project for inspiring our model architecture design. We are grateful to Yixin Song, Yan Song, and Yang Li for their insightful suggestions throughout the project. We also acknowledge the ADC of the Hong Kong University of Science and Technology (Guangzhou) for providing essential computing resources. Finally, we extend our deepest appreciation to our team members for their dedication and contributions from September 2024 to the present.

## License
The code in this repository is released under the MIT License. 
Limitations: While we strive to address safety concerns and promote the generation of ethical and lawful text, the probabilistic nature of language models may still produce unforeseen outputs. These may include biased, discriminatory, or otherwise harmful content. Users are advised not to disseminate such material. We disclaim any liability for consequences resulting from the distribution of harmful information.


## Citation
If you find **Project PLM** helpful for your research or applications, please cite as follows:

```
@misc{cheng2025plm,
      title={PLM: Efficient Peripheral Language Models Hardware-Co-Designed for Ubiquitous Computing}, 
      author={Cheng Deng, Luoyang Sun, Jiwen Jiang, Yongcheng Zeng, Xinjian Wu, Wenxin Zhao, Qingfa Xiao, Jiachuan Wang, Lei Chen, Lionel M. Ni, Haifeng Zhang, Jun Wang},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}
```
