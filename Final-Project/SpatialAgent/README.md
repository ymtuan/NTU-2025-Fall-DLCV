# SpatialAgent

**1st Place Solution of the ICCV 2025 AI City Challenge, Track 3.**

📄 [Paper on arXiv](https://arxiv.org/abs/2507.10778v1)

<p align="center">
  <img src="asset/leaderboard.png" alt="Leaderboard Result"/>
</p>


---

## 🔧 Installation

1. Create and activate a conda environment with Python 3.10

       conda create -n spatialagent python=3.10 -y
       conda activate spatialagent

2. Install Python dependencies (Adjust pytorch installation with your CUDA version)

       pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
       pip install -r requirements.txt

---

## 📦 Preparation

1. Model checkpoints and pre-processed QA data can be downloaded from [here](<https://drive.google.com/drive/u/1/folders/1_ovPjqADpvM0fQdNBLAPdWiemC5MFaG7>).

2. Place the downloaded files in corresponding directory following the below Project Structure.

3. 準備資料
       參考https://huggingface.co/datasets/yaguchi27/DLCV_Final1
       結構會像：

       dlcv_final
       ├── DLCV_Final1
       └── SpatialAgent

---

## 📂 Project Structure

    SpatialAgent
    ├── agent
    ├── distance_est/
    │   └──  ckpt/
    │       ├── 3m_epoch6.pth
    │       └── epoch_5_iter_6831.pth
    ├── inside_pred/
    │   └── ckpt/
    │       └── epoch_4.pth
    ├── utils
    ├── data/
    │   ├── train
    │   ├── val
    │   └── test/
    │       └── images/
    │       └── depths/
    └── README.md

---

## 🧠 Usage

### 1. Inference on test set 
```
cd SpatialAgent/agent
python train_eval.py --dataset test --limit 3 --verbose
//測試前三筆、打印出過程


//後面是原本repo的內容
cd agent
python agent_run.py --project_id <your Vertex AI API>

```
Additionally, some QA might failed because Gemini return invalid format or answer, run again with thinking mode enabled can solve this issue. 
Running this command will re-run those failure cases.
```

cd agent
python agent_run.py --project_id <your Vertex AI API> --think_mode

```

## ⚒️ QA Data Pre-processing and Model Training (Optional)

### 0. QA Data Pre-processing

To pre-process the QA, you need to update the below script with your Google API key.
Note that this step is optional because data.zip already provide the processed QA data.

```
python utils/question_rephrase.py
```


We provide the pre-trained model checkpoint, but we also provide the training script of our model as follows.

### 1. Train the distance estimation model

```
cd distance_est
python train.py
```

### 2. Train the inclusion classification model

```
cd inside_pred
python train.py
```
