import fitz  

def save_first_three_pages(input_pdf, output_pdf):
    # Open the input PDF
    doc = fitz.open(input_pdf)
    
    # Create a new PDF document
    new_doc = fitz.open()
    
    # Copy the first three pages (if available)
    for page_num in range(min(3, len(doc))):
        new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
    
    # Save the new document
    new_doc.save(output_pdf)
    new_doc.close()
    doc.close()
    print(f"Saved first 3 pages to {output_pdf}")

# Example usage
save_first_three_pages(r"pdf_files\BU_CUP_ZNA_US_Captives 2024.pdf", "pdf_files\output.pdf")
