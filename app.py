import streamlit as st
import pickle
import os
import sklearn

# Load the models
knn_model = pickle.load(open('Model/knn_model.pkl', 'rb'))
log_model = pickle.load(open('Model/log_model.pkl', 'rb'))
dtree_model = pickle.load(open('Model/dtree_model.pkl', 'rb'))

def classify(label):
    img_dir = 'IMG'  # Directory where your images are stored
    
    if label == 'setosa':
        img_path = os.path.join(img_dir, 'Iris-setosa.png')
        return 'Setosa', img_path
    elif label == 'versicolor':
        img_path = os.path.join(img_dir, 'Iris-versicolor.png')
        return 'Versicolor', img_path
    elif label == 'virginica':
        img_path = os.path.join(img_dir, 'Iris-virginica.png')
        return 'Virginica', img_path
    else:
        return None, None

def main():
    st.title("Iris Classification")
    html_temp = """
    <div style="background-color:teal; padding:10px">
    <h2 style="color:white; text-align:center;">Iris Classification</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    activities = ['Logistic Regression', 'KNN', 'Decision Tree']
    option = st.sidebar.selectbox('Which model would you like to use?', activities)
    st.subheader(option)
    
    sl = st.slider('Select Sepal Length', 0.0, 10.0)
    sw = st.slider('Select Sepal Width', 0.0, 10.0)
    pl = st.slider('Select Petal Length', 0.0, 10.0)
    pw = st.slider('Select Petal Width', 0.0, 10.0)
    inputs = [[sl, sw, pl, pw]]

    if st.button('Classify'):
        if option == 'Logistic Regression':
            prediction = log_model.predict(inputs)[0]
        elif option == 'KNN':
            prediction = knn_model.predict(inputs)[0]
        else:
            prediction = dtree_model.predict(inputs)[0]

        st.write(f"Prediction: {prediction}, Type: {type(prediction)}")

        # Display the classification result and the image
        label, img_path = classify(prediction)
        if label and img_path:
            st.markdown(f"**Predicted Class: {label}**", unsafe_allow_html=True)
            st.image(img_path, caption=label)
        else:
            st.error("Error: Could not find the corresponding class.")

if __name__ == '__main__':
    main()
