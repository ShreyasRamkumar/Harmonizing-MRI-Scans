FROM continuumio/miniconda3
WORKDIR C:\GPN\Harmonizing-MRI-Scans

# Create the Environment
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment 
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# Demonstrate that the environment is activated
RUN python -c "import torchio"

# the code to run when container is started
COPY run.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "run.py"]

