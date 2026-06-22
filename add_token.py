import os

with open(r"C:\Q2_Sai\run_all_final.py", "r", encoding="utf-8") as f:
    content = f.read()

# Insert the HF token after import os
token_code = 'os.environ["HF_TOKEN"] = "<YOUR_HF_TOKEN>"\n'
content = content.replace('import os\n', f'import os\n{token_code}')

with open(r"C:\Q2_Sai\run_all_final.py", "w", encoding="utf-8") as f:
    f.write(content)
