# Pull the debian-bookworm-slim image
FROM debian:bookworm-slim

# Update the image
RUN apt update && \
    apt upgrade -y

# Install python3
RUN apt install python3 -y

# Install python packages
RUN apt install python3-numpy -y
RUN apt install python3-pandas -y

# Set the WORKDIR
WORKDIR /app
