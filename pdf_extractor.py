#!/usr/bin/env python3
"""
PDF Text Extraction Script - Batch Processing Version
Enhanced column detection for all PDFs in a folder
"""

import json
import os
from pathlib import Path
from datetime import datetime
import argparse
import glob
import argparse

try:
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Installing...")
    os.system("pip install PyMuPDF")
    import fitz


def extract_page_with_columns(page, page_num):
    """Extract text from a single page with proper column detection"""
    page_rect = page.rect
    page_width = page_rect.width
    page_height = page_rect.height
    
    print(f"  - Page dimensions: {page_width:.0f} x {page_height:.0f}")
    
    # Get all text elements with their coordinates
    text_dict = page.get_text("dict")
    words_with_coords = []
    
    for block in text_dict["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        words_with_coords.append({
                            "text": text,
                            "x0": span["bbox"][0],
                            "y0": span["bbox"][1],
                            "x1": span["bbox"][2],
                            "y1": span["bbox"][3],
                            "center_x": (span["bbox"][0] + span["bbox"][2]) / 2
                        })
    
    if not words_with_coords:
        return page.get_text(), "fallback"
    
    print(f"  - Found {len(words_with_coords)} text elements")
    
    # Analyze x-coordinates to detect columns
    center_x_positions = [w["center_x"] for w in words_with_coords]
    center_x_positions.sort()
    
    # Simple two-column detection: find the midpoint
    mid_x = page_width / 2
    
    # Split into left and right based on center position
    left_words = [w for w in words_with_coords if w["center_x"] < mid_x]
    right_words = [w for w in words_with_coords if w["center_x"] >= mid_x]
    
    print(f"  - Left column: {len(left_words)} elements")
    print(f"  - Right column: {len(right_words)} elements")
    
    # If we have substantial content in both columns, treat as two-column
    if len(left_words) > 5 and len(right_words) > 5:
        # Process left column
        left_words.sort(key=lambda w: (w["y0"], w["x0"]))
        left_text_parts = []
        current_line = []
        current_y = None
        
        for word in left_words:
            # If this word is on a new line (significant y difference)
            if current_y is None or abs(word["y0"] - current_y) > 3:
                if current_line:
                    left_text_parts.append(" ".join(current_line))
                current_line = [word["text"]]
                current_y = word["y0"]
            else:
                current_line.append(word["text"])
        
        if current_line:
            left_text_parts.append(" ".join(current_line))
        
        # Process right column
        right_words.sort(key=lambda w: (w["y0"], w["x0"]))
        right_text_parts = []
        current_line = []
        current_y = None
        
        for word in right_words:
            if current_y is None or abs(word["y0"] - current_y) > 3:
                if current_line:
                    right_text_parts.append(" ".join(current_line))
                current_line = [word["text"]]
                current_y = word["y0"]
            else:
                current_line.append(word["text"])
        
        if current_line:
            right_text_parts.append(" ".join(current_line))
        
        # Combine: all left column content first, then all right column content
        left_text = "\n".join(left_text_parts)
        right_text = "\n".join(right_text_parts)
        
        combined_text = left_text + "\n\n" + right_text
        method = "two_column"
        
        print(f"  - Left column text length: {len(left_text)}")
        print(f"  - Right column text length: {len(right_text)}")
        
    else:
        # Single column - process all words together
        words_with_coords.sort(key=lambda w: (w["y0"], w["x0"]))
        
        text_parts = []
        current_line = []
        current_y = None
        
        for word in words_with_coords:
            if current_y is None or abs(word["y0"] - current_y) > 3:
                if current_line:
                    text_parts.append(" ".join(current_line))
                current_line = [word["text"]]
                current_y = word["y0"]
            else:
                current_line.append(word["text"])
        
        if current_line:
            text_parts.append(" ".join(current_line))
        
        combined_text = "\n".join(text_parts)
        method = "single_column"
    
    return combined_text, method


def extract_text_from_pdf(pdf_path):
    """Extract text using enhanced column detection"""
    text = ""
    pages_data = []
    
    try:
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"Processing page {page_num + 1}...")
            
            try:
                page_text, method = extract_page_with_columns(page, page_num + 1)
                
                print(f"  - Used {method} method ({len(page_text)} chars)")
                
                if page_text and page_text.strip():
                    clean_text = page_text.strip()
                    text += f"\n--- Page {page_num + 1} ---\n{clean_text}\n"
                    
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": clean_text,
                        "char_count": len(clean_text),
                        "word_count": len(clean_text.split()),
                        "extraction_method": method
                    })
                else:
                    print(f"  - Warning: No text extracted from page {page_num + 1}")
                    
            except Exception as e:
                print(f"  - Error processing page {page_num + 1}: {e}")
                # Fallback to basic extraction
                fallback_text = page.get_text()
                if fallback_text:
                    text += f"\n--- Page {page_num + 1} ---\n{fallback_text}\n"
                    pages_data.append({
                        "page_number": page_num + 1,
                        "text": fallback_text,
                        "char_count": len(fallback_text),
                        "word_count": len(fallback_text.split()),
                        "extraction_method": "fallback"
                    })
        
        doc.close()
        
    except Exception as e:
        print(f"Error extracting text: {e}")
        return "", []
    
    return text, pages_data


def extract_pdf_to_json(pdf_path, output_path=None):
    """Extract PDF text and save to structured JSON in knowledgebase folder"""
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    if output_path is None:
        knowledgebase_dir = Path("knowledgebase")
        knowledgebase_dir.mkdir(exist_ok=True)
        output_path = knowledgebase_dir / f"{pdf_path.stem}_extracted.json"
    
    print(f"\n{'='*60}")
    print(f"📄 Extracting text from: {pdf_path.name}")
    print(f"{'='*60}")
    print("Using enhanced column detection")
    
    full_text, pages_data = extract_text_from_pdf(pdf_path)
    
    extracted_data = {
        "metadata": {
            "source_file": str(pdf_path),
            "extraction_method": "enhanced_column_detection",
            "extraction_date": datetime.now().isoformat(),
            "total_pages": len(pages_data),
            "total_characters": len(full_text),
            "total_words": len(full_text.split())
        },
        "content": {
            "full_text": full_text.strip(),
            "pages": pages_data
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Text extracted successfully!")
    print(f"📄 Total pages: {extracted_data['metadata']['total_pages']}")
    print(f"📝 Total words: {extracted_data['metadata']['total_words']:,}")
    print(f"💾 Saved to: {output_path}")
    
    return extracted_data


def process_all_pdfs_in_folder(pdfs_folder="pdfs"):
    """Process all PDF files in the specified folder"""
    pdfs_path = Path(pdfs_folder)
    
    if not pdfs_path.exists():
        print(f"❌ PDFs folder not found: {pdfs_path}")
        return []
    
    # Find all PDF files (case-insensitive, avoid duplicates)
    pdf_files = list(pdfs_path.glob("*.pdf")) + list(pdfs_path.glob("*.PDF"))
    # Remove duplicates by converting to set and back (comparing by full path)
    pdf_files = list(set(str(f) for f in pdf_files))
    pdf_files = [Path(f) for f in pdf_files]
    
    if not pdf_files:
        print(f"❌ No PDF files found in: {pdfs_path}")
        return []
    
    print(f"\n🔍 Found {len(pdf_files)} PDF file(s) in '{pdfs_folder}' folder:")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf_file.name}")
    
    processed_files = []
    failed_files = []
    
    # Create knowledgebase directory
    knowledgebase_dir = Path("knowledgebase")
    knowledgebase_dir.mkdir(exist_ok=True)
    
    for pdf_file in pdf_files:
        try:
            result = extract_pdf_to_json(pdf_file)
            processed_files.append(pdf_file)
            print(f"✅ Successfully processed: {pdf_file.name}")
        except Exception as e:
            print(f"❌ Failed to process {pdf_file.name}: {e}")
            failed_files.append((pdf_file, str(e)))
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully processed: {len(processed_files)} files")
    print(f"❌ Failed: {len(failed_files)} files")
    
    if processed_files:
        print(f"\n📁 Extracted JSON files saved to: knowledgebase/")
        for pdf_file in processed_files:
            json_file = f"{pdf_file.stem}_extracted.json"
            print(f"  - {json_file}")
    
    if failed_files:
        print(f"\n❌ Failed files:")
        for pdf_file, error in failed_files:
            print(f"  - {pdf_file.name}: {error}")
    
    return processed_files


def main():
    parser = argparse.ArgumentParser(description="Extract text from PDF(s) with enhanced column detection")
    parser.add_argument("pdf_path", nargs='?', default=None, help="Path to specific PDF file (optional)")
    parser.add_argument("-o", "--output", help="Output JSON file path (for single file)")
    parser.add_argument("-f", "--folder", default="pdfs", help="Folder containing PDFs (default: pdfs)")
    parser.add_argument("--single", action="store_true", help="Process only the specified single file")
    
    args = parser.parse_args()
    
    # If specific PDF file is provided and --single flag is used
    if args.pdf_path and args.single:
        if not os.path.exists(args.pdf_path):
            print(f"❌ PDF file not found: {args.pdf_path}")
            return 1
        
        try:
            extract_pdf_to_json(args.pdf_path, args.output)
            print("\n🎉 Single file extraction complete! Ready for LightRAG pipeline.")
            return 0
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    # Process all PDFs in folder (default behavior)
    try:
        processed_files = process_all_pdfs_in_folder(args.folder)
        
        if processed_files:
            print(f"\n🎉 Batch processing complete! {len(processed_files)} files ready for LightRAG pipeline.")
            print("📝 To process with LightRAG, restart lightrag-server to pick up new files.")
        else:
            print("\n❌ No files were processed successfully.")
            return 1
            
        return 0
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        return 1


if __name__ == "__main__":
    main()