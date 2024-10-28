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
```

After fine-tuning is complete, run:

```
```
