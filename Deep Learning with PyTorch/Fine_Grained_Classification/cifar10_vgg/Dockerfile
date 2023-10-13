FROM shiska07/pytorch_lightning:v2

WORKDIR /app

# create folder to store
RUN mkdir /app/outputs

# cope code to container dolder
COPY ./code /app

# Run the application
CMD python ./src/main.py params_local.json
