# Deploying to Google Cloud Platform

## Prerequisites

1. Install Google Cloud SDK:
```bash
# Download and install from:
# https://cloud.google.com/sdk/docs/install
```

2. Install kubectl:
```bash
gcloud components install kubectl
```

## Step 1: Initial Setup

1. Set up environment variables:
```bash
# Set your project ID
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export CLUSTER_NAME="prescription-analyzer"

# Configure gcloud
gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION
```

2. Enable required APIs:
```bash
# Enable Container Registry API
gcloud services enable containerregistry.googleapis.com

# Enable Kubernetes Engine API
gcloud services enable container.googleapis.com

# Enable Cloud Storage API
gcloud services enable storage.googleapis.com
```

## Step 2: Build and Push Docker Image

1. Build the Docker image:
```bash
# Build the image
docker build -t gcr.io/$PROJECT_ID/prescription-analyzer:latest .

# Push to Container Registry
docker push gcr.io/$PROJECT_ID/prescription-analyzer:latest
```

## Step 3: Create GKE Cluster

1. Create cluster with GPU support:
```bash
gcloud container clusters create $CLUSTER_NAME \
    --region $REGION \
    --num-nodes 3 \
    --machine-type n1-standard-4 \
    --accelerator type=nvidia-tesla-t4,count=1 \
    --enable-ip-alias \
    --enable-autoscaling \
    --min-nodes 3 \
    --max-nodes 10

# Get credentials
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION
```

## Step 4: Install NVIDIA GPU Drivers

1. Install NVIDIA drivers on the cluster:
```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## Step 5: Create Storage Buckets

1. Create GCS buckets:
```bash
# Create input bucket
gsutil mb gs://$PROJECT_ID-prescription-images

# Create output bucket
gsutil mb gs://$PROJECT_ID-prescription-results

# Make buckets public
gsutil iam ch allUsers:objectViewer gs://$PROJECT_ID-prescription-images
gsutil iam ch allUsers:objectViewer gs://$PROJECT_ID-prescription-results
```

## Step 6: Create Kubernetes Resources

1. Create namespace:
```bash
kubectl apply -f kubernetes/base/namespace.yaml
```

2. Create secrets:
```bash
# Create service account
gcloud iam service-accounts create prescription-analyzer \
    --display-name "Prescription Analyzer Service Account"

# Download key
gcloud iam service-accounts keys create credentials.json \
    --iam-account=prescription-analyzer@$PROJECT_ID.iam.gserviceaccount.com

# Create Kubernetes secret for GCP credentials
kubectl create secret generic gcp-credentials \
    --from-file=credentials.json \
    --namespace=prescription-system

# Create secret for OpenFDA API key
kubectl create secret generic api-keys \
    --from-literal=openfda-key=your-api-key \
    --namespace=prescription-system
```

3. Update and apply ConfigMap:
```bash
# Update bucket names in configmap.yaml
sed -i "s/prescription-images/$PROJECT_ID-prescription-images/g" kubernetes/base/configmap.yaml
sed -i "s/prescription-results/$PROJECT_ID-prescription-results/g" kubernetes/base/configmap.yaml

# Apply ConfigMap
kubectl apply -f kubernetes/base/configmap.yaml
```

4. Deploy application:
```bash
# Update PROJECT_ID in deployment.yaml
sed -i "s/\[PROJECT_ID\]/$PROJECT_ID/g" kubernetes/base/deployment.yaml

# Apply deployment
kubectl apply -f kubernetes/base/deployment.yaml
```

5. Create service:
```bash
kubectl apply -f kubernetes/base/service.yaml
```

## Step 7: Configure Public Access

1. Get the external IP:
```bash
kubectl get service prescription-analyzer-service \
    --namespace=prescription-system
```

2. Create DNS record (optional):
```bash
# Replace with your domain
export DOMAIN="prescriptions.yourdomain.com"

# Get external IP
export EXTERNAL_IP=$(kubectl get service prescription-analyzer-service \
    --namespace=prescription-system \
    -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Create DNS A record (using gcloud DNS)
gcloud dns record-sets transaction start --zone=your-zone
gcloud dns record-sets transaction add $EXTERNAL_IP \
    --name=$DOMAIN \
    --ttl=300 \
    --type=A \
    --zone=your-zone
gcloud dns record-sets transaction execute --zone=your-zone
```

## Step 8: Verify Deployment

1. Check deployment status:
```bash
kubectl get deployments -n prescription-system
kubectl get pods -n prescription-system
kubectl get services -n prescription-system
```

2. Test the service:
```bash
# Using curl
curl http://$EXTERNAL_IP/health

# Or using the domain if configured
curl http://$DOMAIN/health
```

## Step 9: Enable Monitoring

1. Enable Cloud Monitoring:
```bash
# Install metrics server
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# Enable monitoring
gcloud container clusters update $CLUSTER_NAME \
    --enable-stackdriver-kubernetes \
    --region $REGION
```

## Step 10: Setup Autoscaling

1. Apply HPA (Horizontal Pod Autoscaler):
```bash
kubectl apply -f kubernetes/base/hpa.yaml
```

## Cleanup (if needed)

1. Delete resources:
```bash
# Delete cluster
gcloud container clusters delete $CLUSTER_NAME --region $REGION

# Delete buckets
gsutil rm -r gs://$PROJECT_ID-prescription-images
gsutil rm -r gs://$PROJECT_ID-prescription-results

# Delete container images
gcloud container images delete gcr.io/$PROJECT_ID/prescription-analyzer:latest
```