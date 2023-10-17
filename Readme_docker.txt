# Build the Docker image
# az acr build --registry storytelleappregistry --resource-group storytelleapp --image storytelle-app:python-v1 .
az acr build --registry storytelleregistry --resource-group storytelleapp --image storytelle-app:python-v1 .

# Create web app
# az webapp create -g storytelleapp -p storytelleAppServicePlan -n storytelle-web-app -i storytelleappregistry.azurecr.io/storytelle-app:python-v1
az webapp create -g storytelleapp -p storytelleAppServicePlan -n storytelle-app -i storytelleregistry.azurecr.io/storytelle-app:python-v1
# TODO update
