# Deep Visual Search Engine

A deep-learning-based image retrieval system that uses **CLIP embeddings** to retrieve visually or semantically similar images from a large image collection. Supports both **text-to-image search** and **image-to-image search** through a Streamlit web interface.

## Features

- **Text-to-image search:** Find images using natural-language descriptions.
- **Image-to-image search:** Upload a query image to retrieve visually similar images.
- **CLIP embeddings:** Represent images and text in a shared 512-dimensional embedding space using OpenCLIP's ViT-B-32 model.
- **Cosine similarity retrieval:** Rank indexed images by similarity to the query embedding.
- **Batch indexing:** Generate embeddings for an entire image collection and save them for subsequent searches.
- **Hardware acceleration:** Automatically selects Apple Silicon MPS, NVIDIA CUDA, or CPU when available.
- **Automated testing:** Uses pytest to validate ranking, input handling, index compatibility, and error cases.

## Architecture

```text
                 IMAGE COLLECTION
                        |
                        v
                Image Preprocessing
                        |
                        v
                 CLIP Image Encoder
                        |
                        v
               512-D Image Embeddings
                        |
                        v
                Persistent Index
                 (NumPy NPZ File)


 Text Query                         Query Image
     |                                   |
     v                                   v
 CLIP Text Encoder                CLIP Image Encoder
     |                                   |
     +---------------+-------------------+
                     |
                     v
             Query Embedding
                     |
                     v
          Cosine Similarity Search
                     |
                     v
             Top-K Ranked Images
                     |
                     v
              Streamlit Web UI
```

The indexing process runs separately from retrieval. Once embeddings have been generated and saved, searches reuse the existing index rather than re-encoding the entire collection.

## Tech Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Deep learning | PyTorch |
| Vision-language model | OpenCLIP, ViT-B-32 |
| Pretrained weights | `laion2b_s34b_b79k` |
| Image processing | Pillow |
| Numerical operations | NumPy |
| Web interface | Streamlit |
| Progress reporting | tqdm |
| Automated testing | pytest |

## Project Structure

```text
deep_visual_search/
├── app/
│   └── streamlit_app.py
├── data/
│   └── index/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── build_index.py
│   └── search.py
├── tests/
│   └── test_search.py
├── dataset/
│   └── ukbench/          # Download separately; not tracked by Git
├── .gitignore
├── README.md
└── requirements.txt
```

The generated embedding index is stored locally at `data/index/clip_index.npz` and is not committed to Git.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/deep_visual_search.git
cd deep_visual_search
```

Replace `YOUR_USERNAME` with your GitHub username after creating the repository.

### 2. Create a virtual environment

Python 3.12 is the development environment used for this project.

**macOS / Linux**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Prepare the dataset

Download the UKBench image dataset and place the image files in:

```text
dataset/ukbench/
```

The directory should contain the image collection used for indexing. The dataset is not included in this repository, so obtain it separately and follow the dataset's applicable terms of use.

### 5. Build the image index

From the project root, run:

```bash
python src/build_index.py
```

The script:

1. Finds supported images in `dataset/ukbench/`.
2. Loads the pretrained CLIP model.
3. Preprocesses images in batches.
4. Extracts image embeddings.
5. L2-normalizes the embeddings.
6. Saves the embeddings, relative image paths, and model metadata to `data/index/clip_index.npz`.

The index is generated locally and reused for future searches. Rebuild it when the indexed image collection or embedding model changes.

### 6. Launch the web application

```bash
streamlit run app/streamlit_app.py
```

Open the local URL printed by Streamlit in your browser.

Choose a search mode, provide a text description or upload an image, select the number of results, and run the search.

### 7. Run automated tests

```bash
python -m pytest -v
```

To generate a coverage report:

```bash
python -m pip install pytest-cov
python -m pytest --cov=src --cov-report=term-missing
```

The automated tests use synthetic embeddings and mocked model outputs where appropriate, avoiding the need to load CLIP or rebuild the complete dataset for routine testing.

## How Retrieval Works

### 1. Image indexing

Each image is passed through the pretrained CLIP image encoder to obtain a 512-dimensional feature vector. The vector is L2-normalized and stored alongside the image's relative path.

For normalized vectors, the dot product is equivalent to cosine similarity.

### 2. Text-to-image retrieval

A natural-language query is tokenized and passed through CLIP's text encoder. The resulting text embedding is compared with the stored image embeddings.

### 3. Image-to-image retrieval

The uploaded query image is preprocessed using the same CLIP preprocessing pipeline as the indexed images. Its embedding is compared with the stored image embeddings.

When the query image is itself part of the indexed collection, that exact image is excluded from its own results.

### 4. Ranking

For query embedding \(q\) and normalized image embedding \(v_i\), similarity is computed as:

\[
s_i = q^\top v_i
\]

Images are sorted by similarity, and the highest-ranked \(k\) results are returned. Similarity scores measure embedding alignment; they should not be interpreted as calibrated probabilities.

## Performance Notes

The initial development run indexed **10,200 images** on an Apple Silicon MacBook Air M2 using MPS acceleration.

| Metric | Observed result |
|---|---:|
| Images indexed | 10,200 |
| Embedding dimensions | 512 |
| Indexing time | Approximately 169 seconds |
| Compressed index size | Approximately 20 MB |
| Device used | Apple Silicon MPS |

These are measurements from the development environment, not universal performance guarantees. Results may vary with hardware, batch size, software versions, and dataset characteristics.

## Limitations

- Retrieval quality depends on the pretrained model, query wording, and image collection.
- The current implementation uses dense embedding similarity rather than a learned task-specific reranker.
- The dataset and generated index must be supplied or generated locally.
- GPU acceleration is optional; CPU execution is supported but may be slower.
- The project currently focuses on retrieval rather than object detection, segmentation, or image classification.

## Future Improvements

- Add caching and more robust error handling.
- Add evaluation metrics such as Recall@K, Precision@K, and mean reciprocal rank.
- Improve retrieval latency for larger image collections.
- Add a dedicated FastAPI backend and separate the retrieval service from the UI.
- Deploy a public demo with documented resource requirements.
- Add CI using GitHub Actions to run automated tests on every push.

## License

Add a license appropriate for your intended use before distributing the repository. Review the pretrained model's and dataset's respective terms separately.
