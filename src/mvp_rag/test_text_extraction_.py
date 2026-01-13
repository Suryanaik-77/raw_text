import fitz
from pypdf import PdfReader
from PIL import Image
import numpy as np
import io
import boto3
from ultralytics import YOLO
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# ENV FLAGS (fix boolean bug)
# -----------------------------
TABLE_EXTRACTION = os.getenv("table_extraction")
IMAGE_SUMMARY = os.getenv("image_summary")

# -----------------------------
# PDF PROCESSOR
# -----------------------------
class PDFProcessor:
    def __init__(
        self,
        yolo_model_path: str,
        aws_region: str = "us-east-1",
        yolo_conf_threshold: float = 0.7,
    ):
        self.yolo = YOLO(yolo_model_path)
        self.yolo_conf_threshold = yolo_conf_threshold
        self.textract = boto3.client("textract", region_name=aws_region)

    @staticmethod
    def pdf_page_to_image(page, dpi=300):
        pix = page.get_pixmap(dpi=dpi)
        return Image.open(io.BytesIO(pix.tobytes("png")))

    @staticmethod
    def extract_ordered_content(blocks):
        block_map = {b["Id"]: b for b in blocks}
        items = []

        for block in blocks:
            if block["BlockType"] in ("LINE", "TABLE"):
                top = block["Geometry"]["BoundingBox"]["Top"]
                items.append((top, block))

        items.sort(key=lambda x: x[0])

        ordered_output = []

        for _, block in items:
            if block["BlockType"] == "LINE":
                ordered_output.append({
                    "type": "text",
                    "content": block["Text"]
                })

            elif block["BlockType"] == "TABLE":
                table = {}
                for rel in block.get("Relationships", []):
                    if rel["Type"] == "CHILD":
                        for cell_id in rel["Ids"]:
                            cell = block_map[cell_id]
                            if cell["BlockType"] == "CELL":
                                r = cell["RowIndex"]
                                c = cell["ColumnIndex"]
                                text = []
                                for rel2 in cell.get("Relationships", []):
                                    if rel2["Type"] == "CHILD":
                                        for wid in rel2["Ids"]:
                                            word = block_map[wid]
                                            if word["BlockType"] == "WORD":
                                                text.append(word["Text"])
                                table.setdefault(r, {})[c] = " ".join(text)

                ordered_output.append({
                    "type": "table",
                    "content": table
                })

        return ordered_output

    # -----------------------------
    # MAIN ENTRY (BYTES, NOT PATH)
    # -----------------------------
    def process_pdf(self, pdf_bytes: bytes) -> str:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        final_text = []

        for i, page in enumerate(reader.pages):
            page_num = i + 1
            final_text.append(f"\n----------- page number {page_num} -----------")

            text = page.extract_text() or ""

            page_img = self.pdf_page_to_image(doc[i])
            page_np = np.array(page_img)

            detections = self.yolo(page_np, conf=0.25, verbose=False)[0]

            has_table = any(
                self.yolo.names[int(b.cls[0])] == "Table"
                for b in detections.boxes
            )

            has_picture = any(
                self.yolo.names[int(b.cls[0])] == "Picture"
                and float(b.conf[0]) >= self.yolo_conf_threshold
                for b in detections.boxes
            )

            # -----------------------------
            # TABLE EXTRACTION
            # -----------------------------
            if has_table and TABLE_EXTRACTION:
                img_bytes = io.BytesIO()
                page_img.save(img_bytes, format="JPEG")

                response = self.textract.analyze_document(
                    Document={"Bytes": img_bytes.getvalue()},
                    FeatureTypes=["TABLES"]
                )

                ordered = self.extract_ordered_content(response["Blocks"])
                skip_text_after_table = False

                for item in ordered:
                    if item["type"] == "table":
                        final_text.append(f"\n--- TABLE (Page {page_num}) ---")
                        table = item["content"]
                        for r in sorted(table):
                            row = [table[r].get(c, "") for c in sorted(table[r])]
                            final_text.append(" | ".join(row))
                        skip_text_after_table = True
                        continue

                    if skip_text_after_table:
                        txt = item["content"]
                        if "|" in txt or len(txt.split()) <= 6:
                            continue
                        skip_text_after_table = False

                    final_text.append(item["content"])

            # -----------------------------
            # IMAGE SUMMARY (future)
            # -----------------------------
            elif has_picture and IMAGE_SUMMARY:
                pass  # hook for image captioning

            else:
                final_text.append(text)

        doc.close()
        return "\n".join(final_text)
