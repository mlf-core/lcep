FROM mlfcore/base:1.2.0

# Install the conda environment
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -a

# Activate the environment
RUN echo "source activate lcep" >> ~/.bashrc
ENV PATH /home/user/miniconda/envs/lcep/bin:$PATH

# Dump the details of the installed packages to a file for posterity
RUN conda env export --name lcep > lcep_environment.yml

# Currently required, since mlflow writes every file as root!
USER root
