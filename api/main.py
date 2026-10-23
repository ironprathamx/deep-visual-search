
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from src.search import VisualSearchEngine

def validate_image(image_path: Path) -> None:
    try:
        with Image.open(image_path) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Only JPEG, PNG, and WebP images are supported.")
            
            width, height = image.size
            if width > 10_000 or height > 10_000:
                raise ValueError(
                    "Image dimensions must not exceed 10,000 pixels per side."
                )

            if width * height > 40_000_000:
                raise ValueError(
                    "Image must not exceed 40 megapixels."
                )

            image.verify()

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the CLIP model and image index once when the API starts.
    app.state.engine = VisualSearchEngine()
    yield
    # Add resource cleanup here later if needed.


app = FastAPI(
    title="Deep Visual Search API",
    description="Search indexed images using CLIP text and image embeddings.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/search/text")
async def search_by_text(
    query: Annotated[str, Form()],
    top_k: Annotated[int, Form()] = 10,
):
    query = query.strip()
    if len(query) > 500:
        raise HTTPException(
            status_code=400,
            detail="Text query must be 500 characters or fewer.",
        )
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Text query cannot be empty.",
        )

    if not 1 <= top_k <= 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20.",
        )

    try:
        results = await run_in_threadpool(
            app.state.engine.search_by_text,
            query,
            top_k,
        )
        return {"query": query, "count": len(results), "results": results}

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/search/image")
async def search_by_image(
    file: Annotated[UploadFile, File()],
    top_k: Annotated[int, Form()] = 10,
):
    if not 1 <= top_k <= 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20.",
        )

    allowed_types = {"image/jpeg", "image/png", "image/webp"}

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    suffix_by_type = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    suffix = suffix_by_type[file.content_type]

    temp_path = None

    try:
        # Save the upload temporarily because the existing engine
        # accepts an image path rather than an in-memory image.
        with NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            size = 0

            while chunk := await file.read(1024 * 1024):
                size += len(chunk)

                if size > 10 * 1024 * 1024:
                    raise HTTPException(
                        status_code=413,
                        detail="Image must be 10 MB or smaller.",
                    )

                temp_file.write(chunk)

        validate_image(temp_path)
        results = await run_in_threadpool(
            app.state.engine.search_by_image,
            temp_path,
            top_k,
        )

        return {
            "filename": file.filename,
            "count": len(results),
            "results": results,
        }

    except HTTPException:
        raise

    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process image: {exc}",
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

        await file.close()
