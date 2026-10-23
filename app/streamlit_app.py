from pathlib import Path

import requests
import streamlit as st
from PIL import Image

import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Deep Visual Search",
    page_icon="🔎",
    layout="wide",
)

st.title("Deep Visual Search")
st.write(
    "Search a collection of images using natural language "
    "or an example image."
)

search_type = st.radio(
    "Search method",
    ["Text search", "Image search"],
    horizontal=True,
)

top_k = st.slider(
    "Number of results",
    min_value=1,
    max_value=20,
    value=10,
)

query_text = None
uploaded_file = None

if search_type == "Text search":
    query_text = st.text_input(
        "Describe the image you're looking for",
        placeholder="e.g. a building with distinctive architecture",
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a query image",
        type=["jpg", "jpeg", "png", "webp"],
    )

search_clicked = st.button(
    "Search",
    type="primary",
    use_container_width=True,
)

if search_clicked:
    try:
        if search_type == "Text search":
            if not query_text or not query_text.strip():
                st.warning("Enter a text description first.")
                st.stop()

            with st.spinner("Searching images..."):
                response = requests.post(
                    f"{API_URL}/search/text",
                    data={"query": query_text.strip(), "top_k": top_k},
                    timeout=120,
                )
                response.raise_for_status()
                results = response.json()["results"]

        else:
            if uploaded_file is None:
                st.warning("Upload an image first.")
                st.stop()

            with st.spinner("Searching images..."):
                response = requests.post(
                    f"{API_URL}/search/image",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    },
                    data={"top_k": top_k},
                    timeout=120,
                )
                response.raise_for_status()
                results = response.json()["results"]

        st.session_state["search_results"] = results
        st.session_state["search_query"] = (
            query_text if search_type == "Text search"
            else uploaded_file.name
        )

    except Exception as exc:
        st.error(f"Search failed: {exc}")


if "search_results" in st.session_state:
    results = st.session_state["search_results"]

    st.subheader("Search results")
    st.caption(
        f"Showing {len(results)} results for: "
        f"{st.session_state['search_query']}"
    )

    if not results:
        st.info("No results found.")
    else:
        columns_per_row = 4

        for start in range(0, len(results), columns_per_row):
            columns = st.columns(columns_per_row)

            for offset, column in enumerate(columns):
                result_index = start + offset

                if result_index >= len(results):
                    break

                result = results[result_index]
                image_path = Path(result["path"])

                with column:
                    st.markdown(f"**Rank {result_index + 1}**")

                    try:
                        with Image.open(image_path) as image:
                            st.image(
                                image.copy(),
                                use_container_width=True,
                            )
                    except Exception:
                        st.warning("Could not display this image.")

                    st.caption(f"Score: {result['score']:.4f}")
                    st.caption(image_path.name)
