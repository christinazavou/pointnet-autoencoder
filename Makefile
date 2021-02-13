#opencv3.1.0: git checkout af64ecdf25e450785c9abf29cfd2085c01d027fb

build:
	cd .. \
		&& docker build \
		-t pointnet-autoencoder-img \
		-f pointnet-autoencoder/Dockerfile pointnet-autoencoder/

run:
	docker run \
		-it \
		--gpus all \
		-e DISPLAY=${DISPLAY} \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		-v data:/pointnet-autoencoder/data \
		--name pointnet-autoencoder-container \
		pointnet-autoencoder-img

run-bash:
	docker run \
		-it \
		--rm \
		--gpus all \
		-e DISPLAY=${DISPLAY} \
		-v /tmp/.X11-unix:/tmp/.X11-unix \
		pointnet-autoencoder-img \
		bash

clean:
	docker rmi -f pointnet-autoencoder-img

tiny-clean:
	docker container rm pointnet-autoencoder-container
