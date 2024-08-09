import fitz  # PyMuPDF
import os
import re
import json

def filter_text(text):
    """
    Filter out unwanted sections such as tables of contents and symbols, and format the text to be more readable.
    
    Args:
    text (str): The extracted text from the PDF.
    
    Returns:
    str: The filtered text.
    """
    # Define patterns to identify unwanted sections
    unwanted_patterns = [
        re.compile(r'\bLEMBAR PERSEMBAHAN\b', re.IGNORECASE),
        re.compile(r'\bLEMBAR PERNYATAAN DIRI\b', re.IGNORECASE),
        re.compile(r'\bLEMBAR PERSETUJUAN PUBLIKASI KARYA ILMIAH\b', re.IGNORECASE),
        re.compile(r'\bLEMBAR PERSETUJUAN TUGAS AKHIR\b', re.IGNORECASE),
        re.compile(r'\bLEMBAR PENGESAHAN TUGAS AKHIR\b', re.IGNORECASE),
        re.compile(r'\bLEMBAR PENGUJIAN TUGAS AKHIR\b', re.IGNORECASE),
        re.compile(r'\bLEMBAR KONSULTASI BIMBINGAN\b', re.IGNORECASE),
        re.compile(r'\bPEDOMAN PENGGUNAAN HAK CIPTA\b', re.IGNORECASE),
        re.compile(r'\bKATA PENGANTAR\b', re.IGNORECASE),
        re.compile(r'\bABSTRAK\b', re.IGNORECASE),
        re.compile(r'\bABSTRACT\b', re.IGNORECASE),
        re.compile(r'\bDAFTAR\b', re.IGNORECASE),
        re.compile(r'\bBAB\b', re.IGNORECASE),
        re.compile(r'\bDAFTAR ISI\b', re.IGNORECASE),
        re.compile(r'\bDAFTAR GAMBAR\b', re.IGNORECASE),
        re.compile(r'\bDAFTAR TABEL\b', re.IGNORECASE),
        re.compile(r'\bDAFTAR SIMBOL\b', re.IGNORECASE),
        re.compile(r'\bLAMPIRAN\b', re.IGNORECASE),
    ]
    
    # Remove unwanted sections based on the defined patterns
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text)
    
    # Further cleaning to make the text more readable
    text = re.sub(r'\n+', '\n\n', text)  # Replace excessive newlines with two newlines (paragraph spacing)
    text = re.sub(r'\.{2,}', '', text)  # Remove sequences of dots
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with a single space
    
    return text

def extract_title(text):
    """
    Extract the title from the text. The title is assumed to be the text before the specified phrase or after a specific phrase.
    
    Args:
    text (str): The text from which to extract the title.
    
    Returns:
    str: The extracted title.
    """
    # Define possible phrases that indicate the end of the title or the start of the title
    keyword_phrases = [
        "SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan Program Strata Satu (S1)",
        "TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Program Diploma Tiga(D3)",
        "TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Diploma Tiga (D3)",
        "SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan Program Sarjana (S1)",
        "SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan Program Sarjana",
        "TUGAS AKHIR Diajukan untuk memenuhi mata kuliah Kerja Praktik pada Program Diploma Tiga",
        ". SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan strata satu (S1)",
        " SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan Program Strata satu (S1)",
        " SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan pada Program Sarjana",
        " SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan Sarjana 1 (S1)",
        "SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan Sarjana 1 (S1)",
        " SKRIPSI Diajukan sebagai salah satu syarat untuk memperoleh gelar Sarjana Program Studi Teknik Informatika",
        "SKRIPSI Diajukan sebagai salah satu syarat untuk memperoleh gelar Sarjana Program Studi Teknik Informatika",
        " SKRIPSI Diajukan untuk memenuhi salah satu syarat melakukan kegiatan skripsi",
        "SKRIPSI Diajukan untuk memenuhi salah satu syarat melakukan kegiatan skripsi",
        " SKRIPSI",
        " SKRIPSI Diajukan untuk memenuhi salah satu syarat kelulusan Program Strata Satu (SI)",
        " LEMBAR JUDUL TUGAS AKHIR TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan program Diploma Tiga",
        " TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Program Diploma Tiga (D3)",
        "TUGAS AKHIR Diajukan untuk memenuhi syarat kelulusan Program Diploma Tiga (D3)",
        "TUGAS AKHIR Diajukan untuk memenuhi syarat kelulusan pada Program Diploma Tiga",
        " TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Program Diploma Tiga (DIII)",
        "TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Program Diploma Tiga (DIII)",
        " LEMBAR JUDUL TUGAS AKHIR TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Diploma Tiga",
        "LEMBAR JUDUL TUGAS AKHIR TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Diploma Tiga",
        " LEMBAR JUDUL TUGAS AKHIR TUGAS AKHIR",
        " TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Diploma Tiga",
        "TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan Diploma Tiga",
        " TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan diploma",
        "TUGAS AKHIR Diajukan untuk memenuhi salah satu syarat kelulusan diploma",
        "PAGE * MERGEFORMAT I LEMBAR JUDUL SKRIPSI"
    ]
    
    # Find the earliest occurrence of any of the phrases
    earliest_index = len(text)
    for phrase in keyword_phrases:
        index = text.find(phrase)
        if index != -1 and index < earliest_index:
            earliest_index = index
    
    # Extract title based on the earliest phrase
    if earliest_index < len(text):
        # If the keyword "PAGE * MERGEFORMAT I LEMBAR JUDUL SKRIPSI" is found
        if "PAGE * MERGEFORMAT I LEMBAR JUDUL SKRIPSI" in text:
            # Extract text after this phrase
            start_index = text.find("PAGE * MERGEFORMAT I LEMBAR JUDUL SKRIPSI") + len("PAGE * MERGEFORMAT I LEMBAR JUDUL SKRIPSI")
            title_text = text[start_index:].strip()
        else:
            title_text = text[:earliest_index].strip()
        
        lines = title_text.split('\n')
        for line in reversed(lines):
            if line.strip():  # Non-empty line
                return line.strip()
    
    # Fallback: return the first non-empty line if none of the keyword phrases are found
    lines = text.split('\n')
    for line in lines:
        if line.strip():  # Non-empty line
            return line.strip()
    
    return "Untitled"

def pdf_to_text(pdf_path, txt_path):
    """
    Convert a PDF file to a text file, filtering out unwanted sections.
    
    Args:
    pdf_path (str): The path to the input PDF file.
    txt_path (str): The path where the output text file will be saved.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text()
        
        filtered_text = filter_text(text)
        
        # Save the filtered text to a .txt file
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(filtered_text)
        
        return filtered_text
    except Exception as e:
        print(f"An error occurred while processing '{pdf_path}': {e}")
        return None

def save_json(dataset, json_path):
    """
    Save dataset to a JSON file.
    
    Args:
    dataset (list): The list of datasets to be saved.
    json_path (str): The path where the JSON file will be saved.
    """
    try:
        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(dataset, json_file, ensure_ascii=False, indent=4)
        print(f"Saved JSON dataset at '{json_path}'.")
    except Exception as e:
        print(f"An error occurred while saving JSON to '{json_path}': {e}")

def convert_pdfs_in_directory(pdf_dir, output_dir):
    """
    Convert all PDF files in a directory to text files and save all data into one JSON file.
    
    Args:
    pdf_dir (str): The directory containing PDF files.
    output_dir (str): The directory where text files and JSON file will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset = []
    
    for file_name in os.listdir(pdf_dir):
        if file_name.lower().endswith('.pdf'):
            # Replace spaces with dashes for the output file name
            output_file_name = file_name.replace(' ', '-')
            
            pdf_path = os.path.join(pdf_dir, file_name)
            txt_path = os.path.join(output_dir, output_file_name.replace('.pdf', '.txt'))
            json_path = os.path.join(output_dir, 'dataset.json')
            
            # Convert PDF to text and save to .txt file
            filtered_text = pdf_to_text(pdf_path, txt_path)
            
            if filtered_text is not None:
                # Extract title from the filtered text
                title = extract_title(filtered_text)
                
                # Extract file name without extension
                file_name_no_ext = os.path.splitext(output_file_name)[0]
                
                # Add the filtered text data to the dataset
                dataset.append({
                    "file_name": file_name_no_ext,
                    "title": title,
                    "text": filtered_text
                })
    
    # Save the dataset to a JSON file
    save_json(dataset, json_path)


def main():
    # Define the directory containing PDF files and the output directory for text files and JSON file
    pdf_directory = 'skripsi'
    output_directory = 'hasil-skripsi'
    
    convert_pdfs_in_directory(pdf_directory, output_directory)

if __name__ == "__main__":
    main()
