This Repo contains the implementation of ML ops for CIFAR-10 Classification task and FastAPI to deploy the model with dockerization

#### Training:





#### FasTAPI with Monitoring:

• The fastapi_prometheus.py includes the code for CIFAR FastAPI and the monitoring of metrics such as the __api runtime__ and __total api call counter__ using prometheus_client and these metrics are available at the address localhost:15000.

• Download the prometheus-monitoring and node-exporter precompiled binaries from https://prometheus.io/download/ and place it in /Monitoring_and_Dockerization folder.

• Modified prometheus.yml file to include metrics from node-exporter and cifar app. 

• By default, node-exporter logs the metrics in the address localhost:9100 and prometheus has the default address localhost:9090.

• Install Grafana by following the steps mention here https://grafana.com/docs/grafana/latest/setup-grafana/installation/debian/

• Visualization in Grafana is added from the prometheus data source for the metrics API runtime, total API calls, API memory utilization, API CPU utilization and API network I/O bytes.

• To start the node-exporter enter the following command:
        cd node_exporter-1.8.0.linux-amd64/ 
        ./node_exporter

• To start prometheus enter the following command from work dir:

      cd prometheus-2.45.5.linux-amd64/
      ./prometheus –config.file=./../Monitoring_and_Dockerization/prometheus.yml

• Start the cifar app with the command:

      python fastapi_prometheus.py ../training/models/model_2.keras

• Start grafana server with the following command:

      sudo service grafana-server start

Grafana dashboard would run at the address __localhost:3000__(default).

The dashboard can be imported from the API_Grafana_Dashboard.json and metrics can be visualized 


#### Dockerization:

Files for docker can be found at /Monitoring_and_Dockerization

To create docker image of the app, run the following commands:

    cd Monitoring_and_Dockerization/
    sudo docker build -t cifar_app .

After successfull build to view the docker image:

    sudo docker images
  
Run the docker container using the command:

    sudo docker run -d -p 8000:8000 -p 15000:15000 cifar_app
