# Pull the debian-10 image
FROM debian:11

# Update the image
RUN apt update && \
    apt upgrade -y

# Install python3
RUN apt install python3 -y

# Install python packages
RUN apt install python3-numpy -y
RUN apt install python3-pandas -y
RUN apt install python3-tqdm -y

# Set the WORKDIR
WORKDIR /app
