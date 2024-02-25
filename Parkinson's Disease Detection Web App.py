import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Load the saved model
loaded_model = pickle.load(open('C:/MCA/MCA/Project/Saved Model/parkinson_model.sav', 'rb'))

# Creating a function for prediction
def parkinson_detection(input_data):
    # Changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the data - assuming you have a scaler used during training
    scaler = StandardScaler()  # Create the scaler
    std_data = scaler.fit_transform(input_data_reshaped)

    # Make predictions
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return 'The person does not have Parkinson disease'
    else:
        return 'The person has Parkinson disease'

def main():
    # Giving a title
    st.title('Parkinson Disease Detection App')

    # Getting the input data from the user
    name = st.text_input('Number of the patient', key='name')
    mdvp_fo = st.text_input('Average vocal fundamental frequency', key='mdvp_fo')
    mdvp_fhi = st.text_input('Maximum vocal fundamental frequency', key='mdvp_fhi')
    mdvp_flo = st.text_input('Minimum vocal fundamental frequency', key='mdvp_flo')
    mdvp_jitter_percent = st.text_input('Measure of the variation in the time between consecutive pitch periods', key='mdvp_jitter_percent')
    mdvp_jitter_abs = st.text_input('Variation in the time between consecutive pitch periods', key='mdvp_jitter_abs')
    mdvp_rap = st.text_input('Five measures of variation in fundamental frequency', key='mdvp_rap')
    mdv_ppq = st.text_input('Variations in the timing of vocal fold vibrations', key='mdv_ppq')
    jitter_ddp = st.text_input('Cycle-to-cycle variations in pitch periods', key='jitter_ddp')
    mdvp_shimmer = st.text_input('Measure of the amplitude variation in consecutive cycles of a speech signal', key='mdvp_shimmer')
    mdvp_shimmer_db = st.text_input('Measures shimmer in decibels', key='mdvp_shimmer_db')
    shimmer_apq3 = st.text_input('Amplitude variations in the speech signal over a short time span', key='shimmer_apq3')
    shimmer_apq5 = st.text_input('Six measures of variation in amplitude', key='shimmer_apq5')
    mdvp_apq = st.text_input('Quantify the variations in the amplitude of the speech signal', key='mdvp_apq')
    shimmer_dda = st.text_input('Six measures of variation in amplitude', key='shimmer_dda')
    nhr = st.text_input('Two measures of ratio of noise to tonal components in the voice', key='nhr')
    hnr = st.text_input('Two measures of ratio of noise to tonal components in the voice', key='hnr')
    rpde = st.text_input('Two measures of ratio of noise to tonal components in the voice', key='rpde')
    dfa = st.text_input('Signal fractal scaling exponent', key='dfa')
    spread1 = st.text_input('Three nonlinear measures of fundamental frequency variation', key='spread1')
    spread2 = st.text_input('Three nonlinear measures of fundamental frequency variation', key='spread2')
    d2 = st.text_input('Two nonlinear dynamical complexity measures', key='d2')
    ppe = st.text_input('Three nonlinear measures of fundamental frequency variation', key='ppe')

    # Code for prediction
    diagnosis = ''

    # Creating a button for prediction
    if st.button('Parkinsons Test Result'):
        diagnosis = parkinson_detection([name, mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_percent, mdvp_jitter_abs,
                                         mdvp_rap, mdv_ppq, jitter_ddp, mdvp_shimmer, shimmer_apq3, shimmer_apq5,
                                         mdvp_apq, shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe])

    st.success(diagnosis)

if __name__ == '__main__':
    main()