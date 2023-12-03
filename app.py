import streamlit as st
import pyedflib
import numpy as np
import tempfile
import os
import requests

# Replace 'http://127.0.0.1:8000/uploadfile/' with your actual endpoint URL
def ret_url(file_name):
    print(file_name)
    url = f'http://127.0.0.1:8000/predict_steamlit?name={file_name}'
    return url
#ENDPOINT_URL = 'http://127.0.0.1:8000/predict_steamlit?name=h01.edf'
UPLOAD_FOLDER = 'edf_buffer'

def load_edf_file(uploaded_file):
    # Save the .edf file content to a temporary file
    temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(uploaded_file.read())

    # Load the .edf file
    f = pyedflib.EdfReader(temp_file_path)

    # Get the signals
    signals = [f.readSignal(i) for i in range(f.signals_in_file)]

    # Get the signal labels
    signal_labels = f.getSignalLabels()

    # Close the file
    f.close()

    # Remove the temporary file
    os.remove(temp_file_path)

    return signals, signal_labels

def load_images_from_directory(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith('.png')]
    return image_files

def call_endpoint(file_name):
    # Set the headers with the file content
    # headers = {"name": file_name}
    print(file_name)

    # Make a POST request to the endpoint
    url_call = ret_url(file_name)
    response = requests.get(url_call)

    # Check the response status
    if response.status_code == 200:
        st.success("Endpoint called successfully!")
    else:
        st.error(f"Failed to call the endpoint. Status code: {response.status_code}")

    return response

def main():
    st.title("EDF File Viewer")

    # Upload .edf file through Streamlit
    uploaded_file = st.file_uploader("Choose an .edf file", type=["edf"])
    
   # print(uploaded_file.getvalue())

    if uploaded_file is not None:
        print(type(uploaded_file))
        # Display file details
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write("### Uploaded File Details")
        st.write(file_details)

        # Load and display content
        signals, signal_labels = load_edf_file(uploaded_file)

        save_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        print(save_path)
        with open(save_path, "wb+") as file:
            file.write(uploaded_file.getvalue())
        st.success(f"File saved at: {save_path}")
        print(uploaded_file.name)

        # Display a small view of the signals in a grid
        st.write("### Small View of Signals")

        # Specify the number of columns in the grid
        num_columns = 2
        # Calculate the number of rows needed
        num_rows = (len(signal_labels) + num_columns - 1) // num_columns

        # Create a grid layout
        for row in range(num_rows):
            columns = st.columns(num_columns)
            for col in range(num_columns):
                idx = row * num_columns + col
                if idx < len(signal_labels):
                    # Display the line chart without using pyplot
                    columns[col].line_chart(np.array(signals[idx][:1000]), use_container_width=True)
                    columns[col].write(f"Signal {idx + 1}: {signal_labels[idx]}")

        # Call the endpoint when the user finishes uploading the file
        if st.button("Call FastAPI Endpoint"):
            #files = {"file": (uploaded_file.name, uploaded_file.read(), uploaded_file.type)}
            response = call_endpoint(uploaded_file.name)

            # Check the response status
            if response.status_code == 200:
                st.success("FastAPI Endpoint called successfully!")
            else:
                st.error(f"Failed to call the FastAPI Endpoint. Status code: {response.status_code}")

            st.subheader("Response from FastAPI Endpoint")
            url_predict = "http://127.0.0.1:8000/predict"
            response = requests.get(url_predict)
            st.markdown(response.json())
            ans = response.json()['prediction']

            image_directory = "grad_cams"
            image_files = load_images_from_directory(image_directory)

            skizo_positive_text = "## You have a high possibility of having schizophrenia"
            skizo_negative_text = "## You have a low possibility of having schizophrenia"

            if not image_files:
                st.warning("No images found in the directory!")
            else:
                for image_file in image_files:
                    image_path = os.path.join(image_directory, image_file)
                    st.image(image_path, caption=image_file, use_column_width=True)
            if ans == 'Schizo positive':
                #st.markdown("you have :" + ans)
                st.write(skizo_positive_text)
            else:
                #st.markdown("you have :" + ans)
                st.write(skizo_negative_text)
            #st.markdown("you have :" + ans)
            requests.get("http://127.0.0.1:8000/clear_dir")
            prediction_image_path = "predictions.png"
            st.image(prediction_image_path, caption='bar graph', use_column_width=True)
            os.remove(prediction_image_path)




if __name__ == "__main__":
    main()
