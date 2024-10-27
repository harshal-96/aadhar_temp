import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import zipfile
import io
import random
import easyocr
import matplotlib.pyplot as plt

def generate_random_4digit_number():
    """Generates a random 4-digit number."""
    return random.randint(1000, 9999)

def detect_text(image_path):
    reader = easyocr.Reader(['en'])
    image = cv2.imread(image_path)

    # Perform OCR
    results = reader.readtext(image)

    # Create a copy of the image for drawing
    output = image.copy()

    # Draw bounding boxes and text
    for (bbox, text, prob) in results:
        # Define the bounding box
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Draw the bounding box
        cv2.rectangle(output, top_left, bottom_right, (0, 255, 0), 2)

        # Put the text above the bounding box
        cv2.putText(output, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


    # Display the image using matplotlib

    return results
def process_single_entry(row, detected_text, output_dir):
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

    # Predefined coordinates for text placement
    match_phrases = [31,33,35,37,39,40,42,44,47,50,52,
                    91,94,97,100,
                    105,107,115,113,
                    92,95,99]

    replace_phrases = [
        name_in_dev, name, flat, soc_name, area,
        f'VTC: {VTC}', f'PO: {PO_name}', f'District: {district}',
        f'State: {state}', f'PIN Code: {pincode}', f'Mobile: {mobile}',
        name_in_dev, name, f'जन्म तिथि/DOB: {dob}', gender,
        f'{flat}, {soc_name}', f'{VTC} , {PO_name}', district, f'{state} - {pincode}',
        f'{flat}, {soc_name_dev}', vtc_dev, district_dev, f'{state_dev} - {pincode}'
    ]

    # Read template image
    image = cv2.imread('output.jpg')
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Use default font
    font = ImageFont.truetype("AnekDevanagari-VariableFont_wdth,wght.ttf", 30)

    # Process text replacements
    for match_phrase, replace_phrase in zip(match_phrases, replace_phrases):
        if match_phrase < len(detected_text):
            box = detected_text[0][match_phrase]
            
            # Extract coordinates
            x0, y0 = map(int, box[0])
            x1, y1 = map(int, box[1])
            x2, y2 = map(int, box[2])
            x3, y3 = map(int, box[3])

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
        profile_photo = Image.open(f'{photo_id}.jpg')
        if profile_photo.mode == 'RGBA':
            profile_photo = profile_photo.convert('RGB')
        profile_photo = profile_photo.resize((250, 330), Image.Resampling.LANCZOS)
        
        result = Image.open(temp_image_path)
        result.paste(profile_photo, (270, 2858))
        
        # Save final image
        final_path = os.path.join(output_dir, f'{name}_card.jpg')
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

        # Detect text in template once
        detected_text = detect_text('output.jpg')
        detected_text=pd.DataFrame(detected_text)
        detected_text[0].iloc[31]=[[350, 1344], [771, 1344], [771, 1384], [350, 1384]]
        detected_text[0].iloc[33]=[[350, 1384], [771, 1384], [771, 1428], [350, 1428]]
        detected_text[0].iloc[35]=[[350, 1428], [771, 1428], [771, 1472], [350, 1472]]
        detected_text[0].iloc[37]=[[350, 1472], [771, 1472], [771, 1512], [350, 1512]]
        detected_text[0].iloc[39]=[[350, 1512], [771, 1512], [771, 1556], [350, 1556]]
        detected_text[0].iloc[40]=[[350, 1556], [771, 1556], [771, 1600], [350, 1600]]
        detected_text[0].iloc[42]=[[350, 1600], [771, 1600], [771, 1644], [350, 1644]]
        detected_text[0].iloc[44]=[[350, 1644], [771, 1644], [771, 1688], [350, 1688]]
        detected_text[0].iloc[47]=[[350, 1688], [771, 1688], [771, 1732], [350, 1732]]
        detected_text[0].iloc[50]=[[350, 1732], [771, 1732], [771, 1776], [350, 1776]]
        detected_text[0].iloc[52]=[[350, 1776], [771, 1776], [771, 1820], [350, 1820]]
        detected_text[0].iloc[91]=[[559, 2875], [696, 2875], [696, 2908], [559, 2908]]
        detected_text[0].iloc[94]=[[559, 2909], [724, 2909], [724, 2952], [559, 2952]]
        detected_text[0].iloc[97]=[[559, 2952], [949, 2952], [949, 2995], [559, 2995]]
        detected_text[0].iloc[105]=[[1474, 3034], [2149, 3034], [2149, 3071], [1474, 3071]]
        detected_text[0].iloc[107]=[[1474, 3072], [2149, 3072], [2149, 3114], [1474, 3114]]
        detected_text[0].iloc[113]=[[1474, 3114], [2149, 3114], [2149, 3145], [1474, 3145]]
        detected_text[0].iloc[115]=[[1474, 3145], [2149, 3145], [2149, 3180], [1474, 3180]]
        detected_text[0].iloc[92]=[[1474, 2876], [2144, 2876], [2144, 2917], [1474, 2917]]
        detected_text[0].iloc[95]=[[1474, 2918], [2144, 2918], [2144, 2950], [1474, 2950]]
        detected_text[0].iloc[99]=[[1474, 2951], [2144, 2951], [2144, 2989], [1474, 2989]]
        detected_text[0].iloc[124]=[[545, 3370], [1017, 3370], [1017, 3428], [545, 3428]]
        detected_text[0].iloc[123]=[[1823, 3329], [2295, 3329], [2295, 3390], [1823, 3390]]
        detected_text[0].iloc[70]=[[495, 2451], [1048, 2451], [1048, 2517], [495, 2517]]
        if st.button("Generate Cards"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Process each row
            generated_files = []
            for index, row in df.iterrows():
                status_text.text(f"Processing {row['Name in English ']}...")
                output_path = process_single_entry(row, detected_text, 'temp')
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