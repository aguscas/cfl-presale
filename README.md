# Savvy Freight ETA 

This project contains the code for our project in collaboration with CFL to predict the ETA of a freight train from Savvy measurements. 

This repository contains the code necesary to run the project, the transformations to integrate it in Pentaho but also the Docker configuration to run it locally (not necessary if working in CFL servers).

## Installation

Navigate to the root of the project and create a new virtual environment
```sh
python -m virtualenv .venv
```

Activate the virtual environment (in Linux/Mac)

```sh
source .venv/bin/activate
```
In Windows Power Shell
```sh
.venv\Scripts\Activate.ps1
```

Inside the `project` folder and after activating the virtual environment run:

```sh
python -m pip install -e . # This will install the project as a development one
```

## Running a jupyter notebook

Wth the activated virutal environment run

```sh
jupyter-notebook
```

After running the above command, if a new tab in the browser does not pop up, navigate in your preferred browser to `http:localhost:8888`


To get up and running:

1. Copy the excel file file with the raw data inside the folder `data` in the root of the project.
2. Rename it to `transition_time.xlsx`
3. Open and execute the notebook titled: `From EXECL to formatted csv`
4. Verify that all experiments are running properly by executing the notebook `analysis/Experiments showcase with crossvalidation`
5. Good to Go :)



