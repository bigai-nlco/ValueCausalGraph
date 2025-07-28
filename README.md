# Are the Values of LLMs Structurally Aligned with Humans? A Causal Perspective 

Welcome! This codebase accompanies the ACL2025 paper [Are the Values of LLMs Structurally Aligned with Humans? A Causal Perspective](https://aclanthology.org/2025.findings-acl.1188/) and is based on [SAELens](https://github.com/jbloomAus/SAELens).

## 1. Python Environment Setup

```bash
pip install -r requirements_vsa.txt
```
## 2. Directory Structure Setup
Set up the following directory structure outside the main project directory:
- .
  - ├── model_data
  - │   ├── google
  - │   │   └── [gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
  - │   ├── jbloom
  - │   │   ├── [Gemma-2b-IT-Residual-Stream-SAEs](https://huggingface.co/jbloom/Gemma-2b-Residual-Stream-SAEs)
  - │   ├── meta-llama
  - │   │   └── [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - │   ├── Juliushanhanhan
  - │   │   └── [llama-3-8b-it-res](https://huggingface.co/Juliushanhanhan/llama-3-8b-it-res)
  - │   └── value_data
  - │       └── [value_orientation.csv](https://github.com/ValueByte-AI/ValueBench/blob/main/data/value_orientation.csv)
  - └── SAELens

## 3. Execution Instructions
### Generate Data
Run the following notebook to generate data with different role and SAE settings for all values:

tutorials/value_causal_graph.ipynb

### Analyze Data
After generating the result CSV files, use the following notebook for data analysis by loading the CSV files:

tutorials/value_causal_graph_analysis.ipynb
