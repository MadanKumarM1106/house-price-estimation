run the following commands for the building the image in the docker file:
  docker build -t house-price-app .
  docker run -p 5000:5000 house-price-app
run the following command in the cmd:
    curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{\"features\": [8.3252, 41, 6.9841, 1.02381, 322, 2.555, 37.88, -122.23]}"
		
