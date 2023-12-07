{\rtf1\ansi\ansicpg1252\cocoartf2636
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Use an official Python runtime as a parent image\
FROM python:3.9\
\
# Set the working directory in the container\
WORKDIR /usr/src/app\
\
# Copy the current directory contents into the container at /usr/src/app\
COPY . .\
\
# Install any needed packages specified in requirements.txt\
RUN pip install --no-cache-dir -r requirements.txt\
\
# Make port 8050 available to the world outside this container\
EXPOSE 8050\
\
# Define environment variable\
ENV NAME World\
\
# Run app.py when the container launches\
CMD ["python", \'93./dash_app.py"]\
}