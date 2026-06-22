conda create -n q2_env python=3.10 -y
conda run -n q2_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 transformers accelerate bitsandbytes scipy matplotlib seaborn tqdm sentencepiece datasets
conda run -n q2_env python C:\Q2_Sai\run_all_final.py
