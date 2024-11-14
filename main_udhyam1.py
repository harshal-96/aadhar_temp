import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import zipfile
import io
import random
from datetime import datetime, timedelta
from fpdf import FPDF


def images_to_pdf(image_paths):
    pdf = FPDF()
    
    # A4 dimensions in mm
    A4_width = 210
    A4_height = 297
    
    # Calculate dimensions to fit image on A4 while maintaining aspect ratio
    for img_path in image_paths:
        pdf.add_page()
        
        # Get image dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size
        
        # Calculate scaling ratio to fit on A4
        width_ratio = A4_width / img_width
        height_ratio = A4_height / img_height
        ratio = min(width_ratio, height_ratio)
        
        # Calculate final dimensions
        pdf_width = img_width * ratio
        pdf_height = img_height * ratio
        
        # Center image on page
        x = (A4_width - pdf_width) / 2
        
        # Add image to PDF
        pdf.image(img_path, x=x, y=0, w=pdf_width, h=pdf_height)
    
    # Save to bytes
    return pdf.output(dest='S').encode('latin-1')

def generate_class_date():
  """Generates a random date before October 10, 2024."""
  start_date = datetime(2024, 1, 1)
  end_date = datetime(2024, 10, 10)
  delta = end_date - start_date
  int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
  random_second = random.randrange(int_delta)
  random_date = start_date + timedelta(seconds=random_second)
  return random_date.strftime("%d/%m/%Y")

def generate_incorp_date():
  """Generates a random date before October 10, 2024."""
  start_date = datetime(2008, 1, 1)
  end_date = datetime(2012, 10, 10)
  delta = end_date - start_date
  int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
  random_second = random.randrange(int_delta)
  random_date = start_date + timedelta(seconds=random_second)
  return random_date.strftime("%d/%m/%Y")

def generate_commence_date():
  """Generates a random date before October 10, 2024."""
  start_date = datetime(2010, 1, 1)
  end_date = datetime(2014, 10, 10)
  delta = end_date - start_date
  int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
  random_second = random.randrange(int_delta)
  random_date = start_date + timedelta(seconds=random_second)
  return random_date.strftime("%d/%m/%Y")


def generate_print_date():
    # Define date range
    start_date = datetime(2024, 1, 5, 12, 1)  # Start from 01/05/2024 12:01 PM
    end_date = datetime(2024, 1, 10, 18, 0)   # End at 01/10/2024 6:00 PM

    # Generate random date and time within range
    random_date = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

    # Format the date and time
    return random_date.strftime("%m/%d/%y, %I:%M %p")

def generate_print_date_md():
    # Define date range
    start_date = datetime(2024, 1, 5, 12, 1)  # Start from 01/05/2024 12:01 PM
    end_date = datetime(2024, 1, 10, 18, 0)   # End at 01/10/2024 6:00 PM

    # Generate random date and time within range
    random_date = start_date + timedelta(seconds=random.randint(0, int((end_date - start_date).total_seconds())))

    # Format the date and time
    return random_date.strftime("%d/%m/%y")


# def generate_random_4digit_number():
#     """Generates a random 4-digit number."""
#     return random.randint(1000, 9999)

# def generate_random_code():
#   """Generates a 16-digit random code with three spaces in between."""
#   code = ''.join(random.choices('0123456789', k=16))
#   return f"{code[:4]} {code[4:8]} {code[8:12]} {code[12:]}"

def get_predefined_coordinates():
    """Returns predefined coordinates for text placement"""
    coordinates = [
        # Format: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        [[575, 769], [1120, 769], [1120, 789], [575, 789]],  # Name 
        [[676, 848], [818, 848], [818, 872], [676, 872]],  # flat
        [[957, 861], [1093, 861], [1093, 881], [957, 881]],  # building
        [[679, 919], [751, 919], [751, 937], [679, 937]],  # village
        [[679, 953], [811, 953], [811, 973], [679, 973]],  # road
        [[677, 973], [784, 973], [784, 998], [677, 998]],
        [[676, 1001], [762, 1001], [762, 1019], [676, 1019]],
        [[955, 975], [1029, 975], [1029, 995], [955, 995]],  # city
        [[679, 1035], [797, 1035], [797, 1053], [679, 1053]],  # state
        [[955, 1033], [1031, 1033], [1031, 1053], [955, 1053]],  # district
        [[677, 1068], [775, 1068], [775, 1087], [677, 1087]],  # mobile
        [[956, 1068], [1144, 1068], [1144, 1094], [956, 1094]]  # email
    ]
    return coordinates

def process_single_entry(row, output_dir):
    """Process a single entry from the Excel sheet and generate an image"""
    # Convert all input values to strings and handle None/NaN values
    def safe_str(value):
        if pd.isna(value) or value is None:
            return ""
        return str(value)

    # Extract and sanitize data from row
    name = safe_str(row['Name'])
    udhyam_no = safe_str(row['udhyam_no'])
    flat = safe_str(row['flat_details'])
    building = safe_str(row['building_name'])
    village = safe_str(row['village_town'])
    road = safe_str(row['road'])
    street=safe_str(row['street'])
    lane=safe_str(row['lane'])
    city = safe_str(row['city'])
    state = 'RAJASTHAN'
    district = safe_str(row['district'])
    mobile = safe_str(row['mobile'])
    email = safe_str(row['email'])
    dist_ind=safe_str(row['dist_ind'])
    msme_dfo=safe_str(row['msme_dfo'])

    random.seed(hash(str(row.to_dict().values())))  # Use row data as seed
    class_date = generate_class_date()
    incorp_date = generate_incorp_date()
    commence_date = generate_commence_date()
    print_date = generate_print_date()
    print_date_md=str(pd.to_datetime(print_date, format="%m/%d/%y, %I:%M %p").strftime('%d/%m/%y'))
    
    # Reset random seed to ensure independence between iterations
    random.seed()
    # Get predefined coordinates
    coordinates = get_predefined_coordinates()

    # Prepare replacement phrases
    replace_phrases = [
        name,flat,building,village,road,street,lane,city,state,district,mobile,email
    ]

    # Read template image
    image = cv2.imread('udhyam1.jpg')
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Use default font
    font = ImageFont.truetype("Times CG Bold.otf", 20)

    # Process text replacements
    for coords, replace_phrase in zip(coordinates, replace_phrases):
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        x2, y2 = coords[2]
        x3, y3 = coords[3]

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Calculate text position
        text_bbox = draw.textbbox((0, 0), replace_phrase, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x0
        text_y = int((y0 + y2) / 2) - (text_height // 2)

        # Add text
        draw.text((x0, y0), replace_phrase, font=font, fill=(0, 0, 0))

    font_udhyam = ImageFont.truetype("Times CG Bold.otf", 30)
    aadhar_positions = [
        [[699, 319], [963, 319], [963, 345], [699, 345]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = udhyam_no
        text_bbox = draw.textbbox((0, 0), text, font=font_udhyam)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw.text((x0, y0), text, font=font_udhyam, fill=(0, 0, 0))


    font_name_ = ImageFont.truetype("Times CG Bold.otf", 30)
    aadhar_positions = [
        [[564, 378], [1098, 378], [1098, 402], [564, 402]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = name
        text_bbox = draw.textbbox((0, 0), text, font=font_name_)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw.text((x0, y0), text, font=font_name_, fill=(0, 0, 0))

    font_class_date = ImageFont.truetype("Times CG Bold.otf", 23)
    aadhar_positions = [
        [[988, 508], [1094, 508], [1094, 532], [988, 532]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = class_date
        text_bbox = draw.textbbox((0, 0), text, font=font_class_date)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw.text((x0, y0+3), text, font=font_class_date, fill=(0, 0, 0))

    font_incorp_date = ImageFont.truetype("Times CG Bold.otf", 23)
    aadhar_positions = [
        [[780, 1140], [884, 1140], [884, 1164], [780, 1164]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = incorp_date
        text_bbox = draw.textbbox((0, 0), text, font=font_incorp_date)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw.text((x0, y0), text, font=font_incorp_date, fill=(0, 0, 0))


    font_commence_date = ImageFont.truetype("Times CG Bold.otf", 23)
    aadhar_positions = [
        [[780, 1222], [886, 1222], [886, 1246], [780, 1246]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = commence_date
        text_bbox = draw.textbbox((0, 0), text, font=font_commence_date)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw.text((x0, y0), text, font=font_commence_date, fill=(0, 0, 0))

    font_print_date = ImageFont.truetype("AnekDevanagari-VariableFont_wdth,wght.ttf", 18)
    aadhar_positions = [
        [[49, 31], [189, 31], [189, 51], [49, 51]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = print_date
        text_bbox = draw.textbbox((0, 0), text, font=font_print_date)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw.text((x0, y0), text, font=font_print_date, fill=(0, 0, 0))


    # Save intermediate image
    temp_image_path = os.path.join(output_dir, f'temp_{name}.jpg')
    image_final = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_image_path, image_final)

    image2 = cv2.imread('udhyam2.jpg')
    image_pil2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    draw2 = ImageDraw.Draw(image_pil2)

    font_print_date = ImageFont.truetype("AnekDevanagari-VariableFont_wdth,wght.ttf", 18)
    aadhar_positions = [
        [[49, 31], [189, 31], [189, 51], [49, 51]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw2.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = print_date
        text_bbox = draw2.textbbox((0, 0), text, font=font_print_date)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw2.text((x0, y0), text, font=font_print_date, fill=(0, 0, 0))


    font_class_date_2 = ImageFont.truetype("Times CG Bold.otf", 22)
    aadhar_positions = [
        [[780, 74], [884, 74], [884, 98], [780, 98]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw2.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = class_date
        text_bbox = draw2.textbbox((0, 0), text, font=font_class_date_2)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw2.text((x0, text_y), text, font=font_class_date_2, fill=(0, 0, 0))



    font_print_date_2 = ImageFont.truetype("Times CG.otf", 18)
    aadhar_positions = [
        [[577, 197], [667, 197], [667, 217], [577, 217]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw2.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = print_date_md
        text_bbox = draw2.textbbox((0, 0), text, font=font_print_date_2)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw2.text((x0, y0), text, font=font_print_date_2, fill=(0, 0, 0))
    
    font_dist_ind = ImageFont.truetype("Times CG Bold.otf", 22)
    aadhar_positions = [
        [[437, 307], [561, 307], [561, 325], [437, 325]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw2.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = dist_ind
        text_bbox = draw2.textbbox((0, 0), text, font=font_dist_ind)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw2.text((x0, y0), text, font=font_dist_ind, fill=(0, 0, 0))

    font_msme_dfo = ImageFont.truetype("Times CG Bold.otf", 22)
    aadhar_positions = [
        [[437, 365], [509, 365], [509, 385], [437, 385]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw2.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = msme_dfo
        text_bbox = draw2.textbbox((0, 0), text, font=font_msme_dfo)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw2.text((x0, y0), text, font=font_msme_dfo, fill=(0, 0, 0))
    temp_image_path2 = os.path.join(output_dir, f'temp_{name}_2.jpg')
    image_final2 = cv2.cvtColor(np.array(image_pil2), cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_image_path2, image_final2)

    return temp_image_path,temp_image_path2


def main():
    st.title("Udhyam Certificate Generator")
    st.write("Upload an Excel file with the required details to generate Udhyam certificates.")
    
    # File uploader for Excel file
    excel_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
    
    if excel_file:
        # Create temporary directory for processing
        if not os.path.exists('temp'):
            os.makedirs('temp')
        
        # Read Excel file
        df = pd.read_excel(excel_file)
        
        if st.button("Generate Certificates"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create a BytesIO object to store the zip file
            zip_buffer = io.BytesIO()
            
            try:
                # Create ZIP file
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Process each row
                    for index, row in df.iterrows():
                        status_text.text(f"Processing {row['Name']}...")
                        
                        # Get both image paths
                        image_path1, image_path2 = process_single_entry(row, 'temp')
                        
                        if image_path1 and image_path2:
                            # Create PDF from images
                            pdf_bytes = images_to_pdf([image_path1, image_path2])
                            
                            # Add PDF to ZIP file
                            pdf_name = f"{row['Name']}_certificate.pdf"
                            zip_file.writestr(pdf_name, pdf_bytes)
                            
                            # Clean up temporary image files
                            os.remove(image_path1)
                            os.remove(image_path2)
                        
                        progress_bar.progress((index + 1) / len(df))
                
                # Reset pointer to start of BytesIO
                zip_buffer.seek(0)
                
                # Offer download
                st.download_button(
                    label="Download All Certificates (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="udhyam_certificates.zip",
                    mime="application/zip"
                )
            
            finally:
                # Try to remove temp directory
                try:
                    os.rmdir('temp')
                except Exception as e:
                    st.error(f"Error removing temp directory: {str(e)}")
            
            status_text.text("Processing complete!")


if __name__ == "__main__":
    main()