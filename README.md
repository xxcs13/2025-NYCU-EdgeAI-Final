# 2025-NYCU-Edgeai-Final


## 專案簡介
本專案為 Edge AI 期末專案，主要目的是載入 ExLlamaV2 模型，並進行文本生成、推論效能測試（Throughput）、以及困惑度（Perplexity, PPL）評估。

## 主要功能
- 載入 ExLlamaV2 Q4 量化模型
- 使用自訂 `generate` 函數進行文本生成
- 以 WikiText-2 資料集評估模型困惑度（PPL）
- 計算推論 Throughput（每秒生成 token 數）
- 將結果輸出至 `19.csv`

## 執行方式

1. **環境設定**
   ```
   git clone https://github.com/eric1236002/2025-NYCU-Edgeai-Final.git
   conda create -n exllama python=3.12
   conda activate exllama
   git clone https://github.com/turboderp-org/exllamav2
   cd exllamav2
   pip install -r requirements.txt
   <-----T4 only---->
   conda install nvidia::cuda-toolkit
   <-----T4 only---->
   pip install . --use-pep517 --no-build-isolation
   pip install datasets==3.5.0
   ```

2. **準備模型**
   - 先前往https://huggingface.co/wei123602/exllama-Llama-3.2-3B-Instruct-3.0bpw 下載 ExLlamaV2 Q4 量化模型。
   - 將 ExLlamaV2 Q4 量化模型放置於 `./llama3.0bpw` 目錄下，或依需求修改 `result.py` 內的 `model_path`。


   ```
   cd 2025-NYCU-Edgeai-Final
   pip install -U huggingface_hub
   huggingface-cli download --resume-download --local-dir ./llama3.0bpw wei123602/exllama-Llama-3.2-3B-Instruct-3.0bpw 
   

   ```
   或是自行量化模型

   ```
   cd exllamav2
   python convert.py \
       -i  /root/Sherry/EdgeAI/final/1w/llama-3.2-3b-instruct \
       -o  /root/Sherry/EdgeAI/final/1w/tmp \
       -cf /root/Sherry/EdgeAI/final/1w/llama3.0bpw/ \
       -b 3.0
   ```
3. **執行主程式**
   ```
   python result.py
   ```

4. **輸出結果**
   - 結果會輸出於 `19.csv`，格式如下：
     | Id | value      |
     |----|------------|
     | 0  | PPL        |
     | 1  | Throughput |


##  成果總結

| 指標       | AWQ 4bit +lora | AWQ 4bit | HQQ + StaticCache | ExLlamaV2（3.0bpw） | ExLlamaV2（2.5bpw） |
| ---------- | -------------- | -------- | ----------------- | ------------------- | ------------------- |
| Perplexity | 11.6           | 40435    | 11.37             | 11.4                | 33                  |
| Throughput | 51             | 51       | 62.6              | 110.7               | 122|

## 主要檔案說明

- [`result.py`](result.py)：主程式，包含模型載入、生成、效能與困惑度評估邏輯。
- [`19.csv`]：程式執行後自動產生，紀錄 PPL 與 Throughput。

## 參考
- [ExLlamaV2](https://github.com/turboderp/exllamav2)
- [WikiText-2 dataset](https://huggingface.co/datasets/wikitext)
