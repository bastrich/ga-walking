install miniconda https://docs.anaconda.com/free/miniconda/

conda create -n uol-final -c kidzik -c conda-forge opensim-org::opensim python=3.8.19
activate uol-final
pip install osim-rl
pip install scipy