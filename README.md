# multi-modal-management-pipeline



\# To deploy a MinIO container:



mkdir -p ~/minio/data



\# Open Docker Desktop



docker run -p 9000:9000 -p 9001:9001 --name minio -v ~/minio/data:/data -e "MINIO\_ROOT\_USER=minioadmin" -e "MINIO\_ROOT\_PASSWORD=minioadmin" quay.io/minio/minio server data --console-address ":9001"



\# WEB UI: http://127.0.0.1:9001

