import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import zipfile
import io
import random

def generate_random_4digit_number():
    """Generates a random 4-digit number."""
    return random.randint(1000, 9999)

def generate_random_code():
  """Generates a 16-digit random code with three spaces in between."""
  code = ''.join(random.choices('0123456789', k=16))
  return f"{code[:4]} {code[4:8]} {code[8:12]} {code[12:]}"

def get_predefined_coordinates():
    """Returns predefined coordinates for text placement"""
    coordinates = [
        # Format: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        [[350, 1344], [771, 1344], [771, 1384], [350, 1384]],  # Name in Devnagri
        [[350, 1384], [771, 1384], [771, 1428], [350, 1428]],  # Name in English
        [[350, 1428], [771, 1428], [771, 1472], [350, 1472]],  # Flat no
        [[350, 1472], [771, 1472], [771, 1512], [350, 1512]],  # Society name
        [[350, 1512], [771, 1512], [771, 1556], [350, 1556]],  # Area
        [[350, 1556], [771, 1556], [771, 1600], [350, 1600]],  # VTC
        [[350, 1600], [771, 1600], [771, 1644], [350, 1644]],  # PO
        [[350, 1644], [771, 1644], [771, 1688], [350, 1688]],  # District
        [[350, 1688], [771, 1688], [771, 1732], [350, 1732]],  # State
        [[350, 1732], [771, 1732], [771, 1776], [350, 1776]],  # PIN
        [[350, 1776], [771, 1776], [771, 1820], [350, 1820]],  # Mobile
        [[559, 2875], [696, 2875], [696, 2908], [559, 2908]],  # Name in Devnagri (bottom)
        [[559, 2909], [724, 2909], [724, 2952], [559, 2952]],  # Name in English (bottom)
        [[559, 2952], [949, 2952], [949, 2995], [559, 2995]],  # DOB
        [[559, 2993], [731, 2993], [731, 3037], [559, 3037]],
        [[1474, 3034], [2149, 3034], [2149, 3071], [1474, 3071]],  # Address line 1
        [[1474, 3072], [2149, 3072], [2149, 3114], [1474, 3114]],  # Address line 2
        [[1474, 3114], [2149, 3114], [2149, 3154], [1474, 3154]],  # District
        [[1474, 3154], [2149, 3154], [2149, 3189], [1474, 3189]],  # State and PIN
        [[1474, 2876], [2144, 2876], [2144, 2917], [1474, 2917]],  # Address in Devnagri 1
        [[1474, 2918], [2144, 2918], [2144, 2950], [1474, 2950]],  # VTC in Devnagri
        [[1474, 2951], [2144, 2951], [2144, 2989], [1474, 2989]]   # District in Devnagri
    ]
    return coordinates

def process_single_entry(row, output_dir, photo_files):
    """Process a single entry from the Excel sheet and generate an image"""
    # Extract data from row
    name = row['Name in English ']
    flat = row['Flat no. ']
    soc_name = row['Society name ']
    area = row['Area Name ']
    VTC_name = row['Village/Town/City Name']
    PO_name = row['Post office Name']
    district = row['District']
    state = row['State']
    pincode = row['Pin code ']
    mobile = row['Mobile No.']
    dob = row['Date of Birth ']
    gender = 'महिला/ Female' if row['Gender'] == 'Female' else 'पुरुष/ Male'
    name_in_dev = row['Name  in Devnagri']
    soc_name_dev = row['Society Name in Devnagri ']
    vtc_dev = row['VTC Name in Devnagri']
    district_dev = row['District Name in Devnagri ']
    state_dev = row['State in Devnagri']
    photo_id = row['Photo Id']
    random_code = generate_random_code()
    # Get predefined coordinates
    coordinates = get_predefined_coordinates()

    # Prepare replacement phrases
    replace_phrases = [
        name_in_dev, name, flat, soc_name, area,
        f'VTC: {VTC_name}', f'PO: {PO_name}', f'District: {district}',
        f'State: {state}', f'PIN Code: {pincode}', f'Mobile: {mobile}',
        name_in_dev, name, f'जन्म तिथि/DOB: {dob}', gender,
        f'{flat}, {soc_name}', f'{VTC_name}, {PO_name} ', district, f'{state} - {pincode}',
        f'{flat}, {soc_name_dev}', vtc_dev, f'{district_dev} , {state_dev} - {pincode}'
    ]

    # Read template image
    image = cv2.imread('output.jpg')
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Use default font
    font = ImageFont.truetype("AnekDevanagari-VariableFont_wdth,wght.ttf", 30)

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
        draw.text((text_x, text_y), replace_phrase, font=font, fill=(0, 0, 0))

    font_medium = ImageFont.truetype("AnekDevanagari-VariableFont_wdth,wght.ttf", 30)
    aadhar_positions = [
        [[634, 2522], [993, 2522], [993, 2559], [634, 2559]],
        [[1912, 3398], [2271, 3398], [2271, 3426], [1912, 3426]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = random_code
        text_bbox = draw.textbbox((0, 0), text, font=font_medium)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)-(text_height // 2)
        draw.text((x0, text_y), text, font=font_medium, fill=(0, 0, 0))

    # Process Aadhar number with larger font
    font_large = ImageFont.truetype("AnekDevanagari-VariableFont_wdth,wght.ttf", 60)
    aadhar_positions = [
        [[545, 3370], [1017, 3370], [1017, 3428], [545, 3428]],
        [[1823, 3329], [2295, 3329], [2295, 3390], [1823, 3390]],
        [[495, 2451], [1048, 2451], [1048, 2517], [495, 2517]]
    ]

    random_number = generate_random_4digit_number()
    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = f'XXXX XXXX {random_number}'
        text_bbox = draw.textbbox((0, 0), text, font=font_large)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - text_height
        draw.text((x0, text_y), text, font=font_large, fill=(0, 0, 0))

    # Save intermediate image
    temp_image_path = os.path.join(output_dir, f'temp_{name}.jpg')
    image_final = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_image_path, image_final)

    # Add profile photo if available
    try:
        # Look for the photo in the uploaded files dictionary
        if str(photo_id) in photo_files:
            photo_data = photo_files[str(photo_id)]
            profile_photo = Image.open(io.BytesIO(photo_data))
            if profile_photo.mode == 'RGBA':
                profile_photo = profile_photo.convert('RGB')
            profile_photo = profile_photo.resize((250, 330), Image.Resampling.LANCZOS)
            
            result = Image.open(temp_image_path)
            result.paste(profile_photo, (270, 2858))
            
            # Save final image
            final_path = os.path.join(output_dir, f'{name}_{photo_id}_card.jpg')
            result.save(final_path, quality=95)
            
            # Clean up temporary file
            os.remove(temp_image_path)
            return final_path
        else:
            st.error(f"Photo ID {photo_id} not found in uploaded files")
            return temp_image_path
    except Exception as e:
        st.error(f"Error processing photo for {name}: {str(e)}")
        return temp_image_path

def main():
    st.title("Aadhar Card Generator")
    st.write("Upload an Excel file with the required details and profile photos to generate Aadhar cards.")

    # File uploader for Excel file
    excel_file = st.file_uploader("Upload Excel file", type=['xlsx', 'xls'])
    
    # Multiple file uploader for profile photos
    profile_photos = st.file_uploader("Upload profile photos", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if excel_file and profile_photos:
        # Create temporary directory for processing
        if not os.path.exists('temp'):
            os.makedirs('temp')

        # Create a dictionary of photo files using their names as keys
        photo_files = {}
        for photo in profile_photos:
            # Remove file extension to get the ID
            photo_id = os.path.splitext(photo.name)[0]
            photo_files[photo_id] = photo.read()

        # Read Excel file
        df = pd.read_excel(excel_file)

        if st.button("Generate Cards"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process each row
            generated_files = []
            for index, row in df.iterrows():
                status_text.text(f"Processing {row['Name in English ']}...")
                output_path = process_single_entry(row, 'temp', photo_files)
                if output_path:
                    generated_files.append(output_path)
                progress_bar.progress((index + 1) / len(df))

            # Create ZIP file
            if generated_files:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for file in generated_files:
                        zip_file.write(file, os.path.basename(file))

                # Offer download
                st.download_button(
                    label="Download All Cards (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="aadhar_cards.zip",
                    mime="application/zip"
                )

                # Clean up
                for file in generated_files:
                    try:
                        os.remove(file)
                    except:
                        pass

                try:
                    os.rmdir('temp')
                except:
                    pass

            status_text.text("Processing complete!")

if __name__ == "__main__":
    main()