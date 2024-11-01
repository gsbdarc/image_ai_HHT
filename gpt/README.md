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
