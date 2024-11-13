# main_app.py
import streamlit as st
import main_aadhar # type: ignore
import pancard_bright  # type: ignore
import main_udhyam

# Import more sub-apps as needed

def main_app():
    st.title("Main App")
    st.write("Welcome to the Multi-App Streamlit Integrator. Select a sub-app from the sidebar to get started.")

def main():
    st.set_page_config(page_title="Multi-App Streamlit Integrator", layout="wide")

  
    apps = {
        "Main App": main_app,
        "Aadhar": main_aadhar.main,
        "Pancard": pancard_bright.main,
        "Udhyam" : main_udhyam.main

    }
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.selectbox("Go to", list(apps.keys()))
    
    app = apps[selection]
    app()

if __name__ == "__main__":
    main()
