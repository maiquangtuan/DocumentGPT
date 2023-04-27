import requests
import io

from PyPDF2 import PdfReader

def extract_text_from_pdffile(pdf_file: str) -> str:
    """
    Function to get the text from a pdf file
    Args:
        pdf_file (str): the file path of the pdf file
    Returns:
        extracted_text (str): the text in the pdf file
    """
    
    assert pdf_file.split(".")[-1] == "pdf" #check the file path is pdf
    reader = PdfReader(pdf_file)
    pages = reader.pages
    extracted_text = ""
    
    for page in pages:
        extracted_text += page.extract_text() + " "

    return extracted_text


def extract_text_from_pdflink(pdf_link: str):
    """
    Function to get the text from a pdf link
    Args:
        pdf_link (str): the link of the pdf file
    Returns:
        extracted_text (str): the text in the pdf file
    """
    response = requests.get(pdf_link)
    pdf_content = io.BytesIO(response.content)
    pdf_reader = PdfReader(pdf_content)
    extracted_text = ""
    for i in range(len(pdf_reader.pages)):
        extracted_text += pdf_reader.pages[i].extract_text() + " "

    return extracted_text