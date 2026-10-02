"""Clear and execute the canonical notebook against the current saved results."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient
path=Path('notebooks/nyc_taxi_portfolio.ipynb')
notebook=nbformat.read(path, as_version=4)
for cell in notebook.cells:
    if cell.cell_type=='code': cell.outputs=[]; cell.execution_count=None
NotebookClient(notebook, timeout=120, kernel_name='python3', resources={'metadata':{'path':str(Path.cwd())}}).execute()
nbformat.validate(notebook)
nbformat.write(notebook,path)
print('Saved the executed notebook with current results.')
