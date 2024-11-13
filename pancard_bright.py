import streamlit as st
import pandas as pd
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import zipfile
import io
import random
import string
from datetime import datetime, timedelta

def adjust_photo_brightness(photo, factor=0.8):
    """
    Adjusts the brightness of a profile photo.
    
    Args:
        photo (PIL.Image): The input profile photo
        factor (float): Brightness adjustment factor (0.0 to 1.0)
    
    Returns:
        PIL.Image: The photo with adjusted brightness
    """
    # Convert PIL Image to numpy array
    photo_np = np.array(photo)
    
    # Convert to HSV color space
    if len(photo_np.shape) == 3:  # Color image
        photo_hsv = cv2.cvtColor(photo_np, cv2.COLOR_RGB2HSV)
        # Adjust the value channel (brightness)
        photo_hsv[..., 2] = photo_hsv[..., 2] * factor
        # Convert back to RGB
        adjusted_photo = cv2.cvtColor(photo_hsv, cv2.COLOR_HSV2RGB)
        return Image.fromarray(adjusted_photo)
    else:  # Grayscale image
        return Image.fromarray((photo_np * factor).astype(np.uint8))

def generate_random_alphanumeric():
    """Generates a random string with 1 letter and 1 number."""
    letter = random.choice(string.ascii_uppercase)
    number = random.randint(0, 9)
    return letter + str(number)

def generate_text(name, fathers_name, surname, pattern):
    # Define a mapping for easy access in the f-string pattern
    initials = {
        'name': name,
        'name_initial': name[0] if name else '',
        'fathers_name': fathers_name,
        'fathers_name_initial': fathers_name[0] if fathers_name else '',
        'surname': surname,
        'surname_initial': surname[0] if surname else '',
        'surname_title': surname.title(),
        'name_title': name.title()
    }

    try:
        # Format the pattern using the initials dictionary
        text = pattern.format(**initials)
    except KeyError as e:
        raise ValueError(f"Invalid pattern or missing data: {e}")

    return text

def get_predefined_coordinates():
    """Returns predefined coordinates for text placement"""
    coordinates = [
        # Format: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        [[31, 449], [567, 449], [567, 485], [31, 485]],  # Name
        [[31, 571], [405, 571], [405, 607], [31, 607]]  # Fathers name # Date of birth
    ]
    return coordinates

def process_single_entry(row, output_dir, photo_files):
    """Process a single entry from the Excel sheet and generate an image"""
    # Extract data from row
    name = row['name']
    fathers_name = row['father_name']
    surname = row['surname']
    birthdate = row['birthdate'].strftime('%d/%m/%Y')
    
    photo_id = row['Photo Id']
    random.seed(hash(str(row.to_dict().values())))  # Use row data as seed
    random_number = generate_random_alphanumeric()
    
    # Reset random seed to ensure independence between iterations
    random.seed()
    
    # Get predefined coordinates
    coordinates = get_predefined_coordinates()

    # Prepare replacement phrases
    replace_phrases = [
        f'{name} {fathers_name} {surname}', f'{fathers_name} {surname}'
    ]

    # Read template image
    image = cv2.imread('pan.jpg')
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)

    # Use default font
    font = ImageFont.truetype("Generic-G50-CC-Typic-DEMO.otf", 30)

    # Process text replacements
    for coords, replace_phrase in zip(coordinates, replace_phrases):
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        x2, y2 = coords[2]
        x3, y3 = coords[3]

        # Calculate text position
        text_bbox = draw.textbbox((0, 0), replace_phrase, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x0
        text_y = int((y0 + y2) / 2) - (text_height // 2)

        # Add text
        draw.text((text_x, text_y), replace_phrase, font=font, fill=(0, 0, 0))

    # Process birthdate
    font_date = ImageFont.truetype("Poppins-Bold.ttf", 30)
    aadhar_positions = [
        [[31, 689], [201, 689], [201, 725], [31, 725]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Add birthdate
        text_bbox = draw.textbbox((0, 0), birthdate, font=font_date)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - text_height
        draw.text((x0, y0), birthdate, font=font_date, fill=(0, 0, 0))

    # Process signature
    fonts = ['Bellarosta.ttf', 'Bestone.ttf']
    font_medium = ImageFont.truetype(random.choice(fonts), 50)
    signature_positions = [
        [[569, 606], [871, 606], [871, 705], [569, 705]]
    ]

    for pos in signature_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])
        
        patterns = [
            '{name_initial}{surname_title}',
            '{name_title}',
            '{name_initial}{fathers_name_initial}{surname_title}',
        ]

        random_pattern = random.choice(patterns)
        text = generate_text(name, fathers_name, surname, random_pattern)
        text_bbox = draw.textbbox((0, 0), text, font=font_medium)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - (text_height // 2) - (text_height // 2)
        draw.text((x0, text_y), text, font=font_medium, fill=(0, 0, 0))

    # Process Aadhar number
    font_large = ImageFont.truetype("Generic-G50-CC-Typic-DEMO.otf", 32)
    aadhar_positions = [
        [[453, 331], [635, 331], [635, 363], [453, 363]]
    ]

    for pos in aadhar_positions:
        x0, y0 = map(int, pos[0])
        x1, y1 = map(int, pos[1])
        x2, y2 = map(int, pos[2])
        x3, y3 = map(int, pos[3])

        # Create white polygon
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)], fill=(255, 255, 255))

        # Add masked Aadhar number
        text = f'XXXXXXXX {random_number}'
        text_bbox = draw.textbbox((0, 0), text, font=font_large)
        text_height = text_bbox[3] - text_bbox[1]
        text_y = int((y0 + y2) / 2) - text_height
        draw.text((x0, y0), text, font=font_large, fill=(0, 0, 0))

    # Save the base image first
    temp_image_path = os.path.join(output_dir, f'temp_{name}.jpg')
    image_final = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_image_path, image_final)

    # Add profile photo if available
    try:
        if str(photo_id) in photo_files:
            # Open and process the profile photo
            photo_data = photo_files[str(photo_id)]
            profile_photo = Image.open(io.BytesIO(photo_data))
            
            # Convert RGBA to RGB if necessary
            if profile_photo.mode == 'RGBA':
                profile_photo = profile_photo.convert('RGB')
            
            # Resize the photo
            profile_photo = profile_photo.resize((200, 220), Image.Resampling.LANCZOS)
            
            # Adjust the brightness of the profile photo
            adjusted_photo = adjust_photo_brightness(profile_photo, factor=0.6)
            
            # Create the final image
            result = Image.open(temp_image_path)
            result.paste(adjusted_photo, (820, 500))
            
            # Save the final image
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
                status_text.text(f"Processing {row['name']}...")
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