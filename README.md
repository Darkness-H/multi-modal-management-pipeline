# Multi-Modal Management Pipeline

The aim of this school project is to develop a pipeline that simulates an end-to-end workflow, from data ingestion and preprocessing to structured management in layered zones, ultimately supporting advanced tasks such as similarity searches and cross-modal generation using pre-trained models.

---

## First Steps

### Working Environment for Notebooks

- Open a terminal and start the Jupyter Notebook framework:
```bash
jupyter notebook
```
- Execute the notebooks in your browser.

---

### Start Docker Containers

Open Docker Desktop and launch the following containers:

#### MinIO
```bash
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin \
  -e MINIO_ROOT_PASSWORD=minioadmin \
  -v C:\minio\data:/data \
  minio/minio server /data --console-address ":9001"
```
- Access the Web UI at [http://127.0.0.1:9001](http://127.0.0.1:9001)

#### ChromaDB
```bash
docker run -d --name chroma \
  -p 8000:8000 \
  -e PERSIST_DIRECTORY=/chroma/chroma \
  -v C:\chromadb\exploitation-zone:/chroma/chroma \
  chromadb/chroma:latest
```
- Although we set a path for ChoromaDB to persistently store embeddings here. It seems like ChromaDB actually stores the embeddings inside the container

---

## Execution Order

### Landing Zone
- `landing_zone.ipynb`  
- `temporal_landing.ipynb`  
- `persistent_landing.ipynb`  

### Formatted Zone
- `formatted_zone.ipynb`  
- `formatted_zone_homogenizer_texts.ipynb`  
- `formatted_zone_homogenizer_images.ipynb`  
- `formatted_zone_homogenizer_videos.ipynb`  

### Trusted Zone
- `trusted_zone.ipynb`  
- `trusted_zone_image_quality_processes.ipynb`  

### Exploitation Zone
- `exploitation_zone_image_embeddings.ipynb`  

> Follow this order so that data flows correctly from ingestion to exploitation while maintaining the layered structure.

---

## Notes

- Each notebook corresponds to a specific stage in the pipeline.

