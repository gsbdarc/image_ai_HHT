You should now be in the main `image_ai_HHT` folder. 

Navigate into `gpt` folder:

```bash
cd gpt 
```

Create a new venv environment:
```bash
/usr/bin/python3 -m venv venv
```

Activate it:
```bash
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

Add kernel to JupyterHub:
```bash
python -m ipykernel install --user --name=gpt
```


Run the notebook with `gpt` as a kernel:

1. [`gpt-zero-shot.ipynb`](`gpt-zero-shot.ipynb`)
2. [`gpt-structured-outputs.ipynb`](`gpt-structured-outputs.ipynb`)
3. [`gpt-few-shot.ipynb`](`gpt-few-shot.ipynb`)
4. [`gpt-vision-fine-tune.ipynb`](`gpt-vision-fine-tune.ipynb`)
   - Before fine-tuning, run [`create_test_jsonl_base.py`](`create_test_jsonl_base.py`)
   - After fine-tuning, edit the python file to have the fine-tuned model name then run [`create_test_jsonl_fine_tuned.py`](`create_test_jsonl_fine_tuned.py`)
6. [`gpt-eval.ipynb`](`gpt-eval.ipynb`)
