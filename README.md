This Repo contains the implementation of ML ops for CIFAR-10 Classification task and FastAPI to deploy the model with dockerization

Training





FasTAPI






Dockerization 
Files for docker can be found at /Monitoring_and_Dockerization

To create docker image of the app, run the following commands:

  cd Monitoring_and_Dockerization/
  sudo docker build -t cifar_app .

After successfull build to view the docker image:

  sudo docker images
  
Run the docker container using the command:

  sudo docker run -d -p 8000:8000 -p 15000:15000 cifar_app
