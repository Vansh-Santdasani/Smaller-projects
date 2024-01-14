import PyPDF2

def pdf_to_text(pdf_path, txt_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        with open(txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

# Specify the paths for your input PDF and output text files
input_pdf_path = '/home/vansh/Downloads/Java_MCQ.pdf'
output_txt_path = '/home/vansh/Downloads/Java_MCQ.txt'

# Convert PDF to text
pdf_to_text(input_pdf_path, output_txt_path)
