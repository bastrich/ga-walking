Details about setting up a project with OpenSim environment are available here: https://opensimconfluence.atlassian.net/wiki/spaces/OpenSim/pages/53085346/Scripting+in+Python

### Short instructions for this specific project (Windows 10):

1) Install miniconda https://docs.anaconda.com/free/miniconda/
2) Cd to the project dir and run in miniconda terminal: '_conda env create -f environment.yaml_'
3) Activate env with '_activate ga-walking_' or import it to your IDE
4) Run ga.py for executing the Genetic algorithm
5) Run visualize.py for visualization

### Running unit tests
1) _coverage run -m unittest discover tests/unit_tests_
2) _coverage html_