# TrueScope Release Workflow (Step-by-Step)

Use this exact sequence every time you deploy.

## 1. Local: Build and Push New Image

Open PowerShell on your local machine.

Go to project:

```powershell
cd C:\Users\jomka\OneDrive\Desktop\THESIS\api_v1
```

Login to Docker Hub:

```powershell
docker login -u <DOCKERHUB_USERNAME>
```

When prompted:

- `Password:` paste/type your Docker Hub password, then press Enter.

Set image and create release tag:

```powershell
$IMAGE="<DOCKERHUB_USERNAME>/truescope-api"
$TAG=(Get-Date -Format "yyyyMMdd-HHmm")
```

Build for Linux AMD64 and push both timestamp tag and latest:

```powershell
docker buildx build --platform linux/amd64 -t ${IMAGE}:$TAG -t ${IMAGE}:latest --push .
```

## 2. Connect to VM

From local terminal:

```bash
ssh root@<VM_PUBLIC_IPV4>
```

When prompted:

- `root@<VM_PUBLIC_IPV4>'s password:` paste/type the VM root password, then press Enter.

## 3. VM: Go to Deploy Folder

```bash
cd /opt/truescope/run
pwd
ls -la
```

## 4. VM: Pull and Deploy

Pull latest image referenced by compose:

```bash
docker compose -f compose.yml pull
```

Recreate containers:

```bash
docker compose -f compose.yml up -d
```

## 5. VM: Verify Deployment

Check status:

```bash
docker compose -f compose.yml ps
```

Check API logs:

```bash
docker compose -f compose.yml logs --tail=100 api
```

Check running image reference:

```bash
docker inspect $(docker compose -f compose.yml ps -q api) --format '{{.Config.Image}}'
```

## 6. Rollback (If Needed)

If newest release is bad, pin a previous timestamp tag in `/opt/truescope/run/compose.yml`:

```yaml
services:
  api:
    image: <DOCKERHUB_USERNAME>/truescope-api:<PREVIOUS_TAG>
```

Then re-deploy:

```bash
docker compose -f compose.yml pull
docker compose -f compose.yml up -d
docker compose -f compose.yml ps
docker compose -f compose.yml logs --tail=100 api
```

## 7. One-Block Quick Version

### Local (PowerShell)

```powershell
cd C:\Users\jomka\OneDrive\Desktop\THESIS\api_v1
docker login -u <DOCKERHUB_USERNAME>
$IMAGE="<DOCKERHUB_USERNAME>/truescope-api"
$TAG=(Get-Date -Format "yyyyMMdd-HHmm")
docker buildx build --platform linux/amd64 -t ${IMAGE}:$TAG -t ${IMAGE}:latest --push .
echo $TAG
```

Prompt during `docker login`:

- `Password:` paste your Docker Hub password or access token, then press Enter.

### VM

```bash
ssh root@<VM_PUBLIC_IPV4>
cd /opt/truescope/run
docker compose -f compose.yml pull
docker compose -f compose.yml up -d
docker compose -f compose.yml ps
docker compose -f compose.yml logs --tail=100 api
docker inspect $(docker compose -f compose.yml ps -q api) --format '{{.Config.Image}}'
```

Prompt during `ssh`:

- `root@<VM_PUBLIC_IPV4>'s password:` type the VM root password, then press Enter.
