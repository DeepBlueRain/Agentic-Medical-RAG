import json
import os
import re

from bs4 import BeautifulSoup


HTML_DIRECTORY = "./data/"
OUTPUT_JSON_PATH = "./data/processed_data.json"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def extract_text_and_title_from_html(html_filepath):
    """Extract title and main text from an HTML file."""
    try:
        with open(html_filepath, "r", encoding="utf-8") as file:
            html_content = file.read()

        soup = BeautifulSoup(html_content, "lxml")

        title_tag = soup.find("title")
        title = title_tag.string.strip() if title_tag and title_tag.string else os.path.basename(html_filepath)
        title = title.replace(".html", "")

        content_tag = (
            soup.find("content")
            or soup.find("div", class_="rich_media_content")
            or soup.find("article")
            or soup.find("main")
            or soup.find("body")
        )

        if not content_tag:
            print(f"Warning: no main content tag found in {html_filepath}")
            return title, None

        text = content_tag.get_text(separator="\n", strip=True)
        text = re.sub(r"\n\s*\n", "\n", text).strip()
        text = text.replace("阅读全文", "").strip()
        return title, text

    except FileNotFoundError:
        print(f"Error: file not found: {html_filepath}")
        return None, None
    except Exception as exc:
        print(f"Error processing {html_filepath}: {exc}")
        return None, None


def split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""
    if not text:
        return []
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks = []
    step = chunk_size - chunk_overlap

    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(text):
            break

    return chunks


def build_processed_dataset(html_directory=HTML_DIRECTORY, output_json_path=OUTPUT_JSON_PATH):
    """Parse HTML files and save chunked documents for vector indexing."""
    all_data_for_milvus = []
    file_count = 0
    chunk_count = 0

    print(f"Processing HTML files under {html_directory}...")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    html_files = [name for name in os.listdir(html_directory) if name.endswith(".html")]
    print(f"Found {len(html_files)} HTML files.")

    for filename in html_files:
        filepath = os.path.join(html_directory, filename)
        file_count += 1

        title, main_text = extract_text_and_title_from_html(filepath)
        if not main_text:
            print(f"Warning: no valid text extracted from {filename}")
            continue

        chunks = split_text(main_text)
        for index, chunk in enumerate(chunks):
            chunk_count += 1
            all_data_for_milvus.append(
                {
                    "id": f"{filename}_{index}",
                    "title": title or filename,
                    "abstract": chunk,
                    "source_file": filename,
                    "chunk_index": index,
                }
            )

    with open(output_json_path, "w", encoding="utf-8") as file:
        json.dump(all_data_for_milvus, file, ensure_ascii=False, indent=2)

    print(f"Processed {file_count} files and generated {chunk_count} chunks.")
    print(f"Saved result to {output_json_path}")


if __name__ == "__main__":
    build_processed_dataset()
