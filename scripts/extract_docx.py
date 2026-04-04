import sys
from docx import Document

def extract_text(docx_path, txt_path):
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(full_text))

if __name__ == '__main__':
    extract_text(sys.argv[1], sys.argv[2])
