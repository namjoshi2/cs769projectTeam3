# Use a Miniconda image as the base for Python environment management
FROM continuumio/miniconda3

# Set the Conda environment name used by setup_lm.sh
ENV LM_ENV_NAME b2txt25_lm

# 1. Install System Dependencies
# The n-gram model uses a Kaldi-based implementation which requires CMake and build tools (GCC)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory and copy the code
WORKDIR /app/nejm-brain-to-text
COPY . /app/nejm-brain-to-text

# 3. Run the language model setup script
# This script creates the b2txt25_lm environment and installs the required packages, 
# including compiling the necessary Kaldi components.
RUN bash setup_lm.sh

# 4. Set the default command to activate the environment and start a shell
# This makes it easy to run the model commands inside the container
CMD ["conda", "run", "-n", "b2txt25_lm", "/bin/bash"]