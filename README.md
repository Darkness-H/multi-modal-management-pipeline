# multi-modal-management-pipeline



\# To deploy a MinIO container:



mkdir -p ~/minio/data



\# Open Docker Desktop



docker run -d --name minio -p 9000:9000 -p 9001:9001 -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin -v C:\minio\data:/data minio/minio server /data --console-address ":9001"



\# WEB UI: http://127.0.0.1:9001

