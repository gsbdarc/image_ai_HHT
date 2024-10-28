Navigate to a ZFS space where you have write permissions.

```bash
cd /zfs/projects/<my-project>
```

Clone the GitHub repository:
```bash
git clone https://github.com/gsbdarc/image_ai_HHT.git
```

```bash
cd image_ai_HHT 
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
python -m ipykernel install --user --name=image-ai
```


Run:
```bash

python download_cifar10.py
python extract_cifar10_images.py
python prepare_data_splits.py
python create_test_jsonl_base_model.py
```
Open `vision-fine-tune.ipynb` notebook.

After fine-tuning is complete, run:

```
python create_test_jsonl_finetuned.py
```
