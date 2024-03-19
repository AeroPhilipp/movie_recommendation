.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y movie_recom || :
	@pip install -e .

data_embedding_with_mini:
	python -c 'from movie_recom.interface.main import embed_data_with_mini; embed_data_with_mini()'

fit_nearest_neighbors:
	python -c 'from movie_recom.interface.main import fit_nearest_neighbors; fit_nearest_neighbors()'

prediction:
	python -c 'from movie_recom.interface.main import predict_movie; predict_movie()'

run_api:
	uvicorn movie_recom.api.fast:app --reload

run_all:
	prediction

docker_build_local:
	docker build --tag=${GAR_IMAGE}:dev .

docker_run:
	docker run -it -e PORT=8000 -p 8000:8000 ${GAR_IMAGE}:dev sh

docker_cloud:
	gcloud auth configure-docker ${GCP_REGION}-docker.pkg.dev
	gcloud artifacts repositories create movierecom --repository-format=docker \
	--location=${GCP_REGION} --description="Repository for storing movie_recom images"

docker_build_cloud:
	docker build -t  ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/movierecom/${GAR_IMAGE}:prod .

docker_push:
	docker push ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/movierecom/${GAR_IMAGE}:prod

docker_deploy:
	gcloud run deploy --image ${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/movierecom/${GAR_IMAGE}:prod --cpu ${GAR_CPU} --memory ${GAR_MEMORY} --region ${GCP_REGION}
