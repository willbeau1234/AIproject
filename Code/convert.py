"""
@file pdf_to_text.py
@brief Script to extract text from PDF files in a folder and save it as text files.
"""

import os
from PyPDF2 import PdfReader

"""
@brief Converts all PDF files in a specified input folder to text files in the output folder.

@details This function processes all `.pdf` files in the input folder, extracts their text using the PyPDF2 library, 
and writes the text to corresponding `.txt` files in the output folder. Each page's content is prefixed with its page number.

@param input_folder Path to the folder containing PDF files.
@param output_folder Path to the folder where text files will be saved.
"""
def pdf_to_text_folder(input_folder, output_folder):
    try:
        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Process all PDF files in the input folder
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(input_folder, file_name)
                pdf_name = os.path.splitext(file_name)[0]
                text_file_path = os.path.join(output_folder, f"{pdf_name}.txt")

                # Read the PDF and extract text
                reader = PdfReader(pdf_path)
                with open(text_file_path, 'w', encoding='utf-8') as text_file:
                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text:
                            text_file.write(f"Page {page_num + 1}:\n{text}\n\n")
                        else:
                            text_file.write(f"Page {page_num + 1}: [No Text Extracted]\n\n")

                print(f"Text extracted and saved to: {text_file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
if __name__ == "__main__":
    """
    @brief Example usage of the `pdf_to_text_folder` function.

    @details Converts PDF files from the specified input folder to text files in the specified output folder.
    Modify the `input_folder` and `output_folder` variables to your desired paths before running the script.
    """
    input_folder = "/Users/willbeaumaster/Desktop/AI/AIproject/data"  # Folder containing PDFs
    output_folder = "/Users/willbeaumaster/Desktop/AI/AIproject/output"  # Output folder for text files
    pdf_to_text_folder(input_folder, output_folder)
