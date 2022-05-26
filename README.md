# Posthoc Privacy guarantees for neural network queries
To run the code, first you should set up appropriate dependencies by creating a virtual environment and then running the following:
```
pip install requirements.txt
```
Once the packages are installed, you will have to set up Gurobi Optimizer. Note that if you only want to run the first two stages - embedding.py and adversarial_training.py then you can skip installing the Gurobi optimizer. However, for estimating the Local Lipschitz constant it is a necessary requirement.

## Installing Gurobi and Gurobipy
This excerpt is taken from [LipMip repository](https://github.com/revbucket/lipMIP). First download the most recent version of the Gurobi optimizer (accessible with a free academic license). [Gurobi Website.](https://www.gurobi.com/downloads/gurobi-optimizer-eula/)

Then add the following environment variables to a file that runs on terminal startup (e.g. `~/.bash_profile`):
```
export GUROBI_HOME="/your/path/to/gurobi/linux64"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib"
```

And install the license obtained from the Gurobi website with the command:
```
$ grbgetkey aaaa0000-0000-0000-0000-000000000000
```

Finally install and test the gurobipy package to your python virtual environment by calling 
```
$ cd $GUROBI_HOME
$ python3 setup.py install 
$ python3 -c "import gurobipy"
```