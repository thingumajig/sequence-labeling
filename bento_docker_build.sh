if [ -z $1]
then
    echo "Please specify docker image name as {docker_username}/{image_name}"
else
    saved_path=$(bentoml get TransformerService:latest --print-location --quiet)
    docker build -t $1 $saved_path
fi