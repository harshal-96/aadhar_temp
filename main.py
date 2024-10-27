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

# Pre-defined coordinates instead of using EasyOCR
TEMPLATE_COORDINATES = {
    'name_dev': [[350, 1344], [771, 1344], [771, 1384], [350, 1384]],
    'name_eng': [[350, 1384], [771, 1384], [771, 1428], [350, 1428]],
    'flat': [[350, 1428], [771, 1428], [771, 1472], [350, 1472]],
    'society': [[350, 1472], [771, 1472], [771, 1512], [350, 1512]],
    'area': [[350, 1512], [771, 1512], [771, 1556], [350, 1556]],
    'vtc': [[350, 1556], [771, 1556], [771, 1600], [350, 1600]],
    'po': [[350, 1600], [771, 1600], [771, 1644], [350, 1644]],
    'district': [[350, 1644], [771, 1644], [771, 1688], [350, 1688]],
    'state': [[350, 1688], [771, 1688], [771, 1732], [350, 1732]],
    'pin': [[350, 1732], [771, 1732], [771, 1776], [350, 1776]],
    'mobile': [[350, 1776], [771, 1776], [771, 1820], [350, 1820]],
    'card_name_dev': [[559, 2875], [696, 2875], [696, 2908], [559, 2908]],
    'card_name_eng': [[559, 2909], [724, 2909], [724, 2952], [559, 2952]],
    'dob_gender': [[559, 2952], [949, 2952], [949, 2995], [559, 2995]],
    'card_address1': [[1474, 3034], [2149, 3034], [2149, 3071], [1474, 3071]],
    'card_address2': [[1474, 3072], [2149, 3072], [2149, 3114], [1474, 3114]],
    'card_district': [[1474, 3114], [2149, 3114], [2149, 3145], [1474, 3145]],
    'card_state_pin': [[1474, 3145], [2149, 3145], [2149, 3180], [1474, 3180]],
    'card_address1_dev': [[1474, 2876], [2144, 2876], [2144, 2917], [1474, 2917]],
    'card_address2_dev': [[1474, 2918], [2144, 2918], [2144, 2950], [1474, 2950]],
    'card_district_state_dev': [[1474, 2951], [2144, 2951], [2144, 2989], [1474, 2989]],
    'aadhar1': [[545, 3370], [1017, 3370], [1017, 3428], [545, 3428]],
    'aadhar2': [[1823, 3329], [2295, 3329], [2295, 3390], [1823, 3390]],
    'aadhar3': [[495, 2451], [1048, 2451], [1048, 2517], [495, 2517]]
}

def process_single_entry(row, output_dir):
    """Process a single entry from the Excel sheet and generate an image"""
    # Extract data from row
    name = row['Name in English ']
    flat = row['Flat no. ']
    soc_name = row['Society name ']
    area = row['Area Name ']
    VTC = row['Village/Town/City Name']
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

    # Create mapping of locations to text
    text_mapping = {
        'name_dev': name_in_dev,
        'name_eng': name,
        'flat': flat,
        'society': soc_name,
        'area': area,
        'vtc': f'VTC: {VTC}',
        'po': f'PO: {PO_name}',
        'district': f'District: {district}',
        'state': f'State: {state}',
        'pin': f'PIN Code: {pincode}',
        'mobile': f'Mobile: {mobile}',
        'card_name_dev': name_in_dev,
        'card_name_eng': name,
        'dob_gender': f'जन्म तिथि/DOB: {dob}, {gender}',
        'card_address1': f'{flat}, {soc_name}',
        'card_address2': f'{VTC} , {PO_name}',
        'card_district': district,
        'card_state_pin': f'{state} - {pincode}',
        'card_address1_dev': f'{flat}, {soc_name_dev}',
        'card_address2_dev': vtc_dev,
        'card_district_state_dev': f'{district_dev}, {state_dev} - {pincode}'
    }

    # Read template image
    image = cv2.imread('output.jpg')
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Use default font
    font = ImageFont.truetype("AnekDevanagari-VariableFont_wdth,wght.ttf", 30)
    font_large = ImageFont.truetype("AnekDevanagari-VariableFont_wdth,wght.ttf", 60)

    # Process text replacements
    for key, text in text_mapping.items():
        coords = TEMPLATE_COORDINATES[key]
        x0, y0 = map(int, coords[0])
        x1, y1 = map(int, coords[1])
        x2, y2 = map(int, coords[2])
        x3, y3 = map(int, coords[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Calculate text position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2)

        # Add text
        draw.text((x0, text_y), text, font=font, fill=(0, 0, 0))

    # Process Aadhar number
    random_number = generate_random_4digit_number()
    aadhar_text = f'XXXX XXXX {random_number}'
    
    for key in ['aadhar1', 'aadhar2', 'aadhar3']:
        coords = TEMPLATE_COORDINATES[key]
        x0, y0 = map(int, coords[0])
        x1, y1 = map(int, coords[1])
        x2, y2 = map(int, coords[2])
        x3, y3 = map(int, coords[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text_bbox = draw.textbbox((0, 0), aadhar_text, font=font_large)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - text_height
        draw.text((x0, text_y), aadhar_text, font=font_large, fill=(0, 0, 0))

    # Save intermediate image
    temp_image_path = os.path.join(output_dir, f'temp_{name}.jpg')
    image_final = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_image_path, image_final)

    # Add profile photo if available
    try:
        profile_photo = Image.open(f'{photo_id}.jpg')
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

        # Save profile photos
        for photo in profile_photos:
            photo_path = os.path.join('temp', photo.name)
            with open(photo_path, 'wb') as f:
                f.write(photo.getvalue())

        # Read Excel file
        df = pd.read_excel(excel_file)

        if st.button("Generate Cards"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process each row
            generated_files = []
            for index, row in df.iterrows():
                status_text.text(f"Processing {row['Name in English ']}...")
                output_path = process_single_entry(row, 'temp')
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

                # Clean up profile photos
                for photo in profile_photos:
                    try:
                        os.remove(os.path.join('temp', photo.name))
                    except:
                        pass

                try:
                    os.rmdir('temp')
                except:
                    pass

            status_text.text("Processing complete!")

if __name__ == "__main__":
    main()