
from pathlib import Path
import sys
import tempfile

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.search import VisualSearchEngine


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


@st.cache_resource
def load_engine():
    return VisualSearchEngine()


try:
    engine = load_engine()
except Exception as exc:
    st.error(f"Could not load the search engine: {exc}")
    st.stop()


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
                results = engine.search_by_text(
                    query_text.strip(),
                    top_k=top_k,
                )

        else:
            if uploaded_file is None:
                st.warning("Upload an image first.")
                st.stop()

            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_file.name).suffix
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file.flush()

                with st.spinner("Searching images..."):
                    results = engine.search_by_image(
                        temp_file.name,
                        top_k=top_k,
                    )

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
