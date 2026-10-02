"""Execute the results walkthrough and save identical root and notebook copies."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path=Path('notebooks/nyc_taxi_portfolio.ipynb')
notebook=nbformat.read(path, as_version=4)
client=NotebookClient(notebook, timeout=120, kernel_name='python3', resources={'metadata':{'path':str(Path.cwd())}})
client.execute()
nbformat.validate(notebook)
for output in (path, Path('newyork-taxi.ipynb')):
    nbformat.write(notebook, output)
print('Executed real-results walkthrough successfully; saved outputs in both notebook copies.')
