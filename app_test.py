import streamlit as st
import pandas as pd
from pycaret.classification import setup as setup_classification, compare_models as compare_classification_models, plot_model as plot_classification_model, create_model as create_classification_model
from pycaret.regression import setup as setup_regression, compare_models as compare_models_regression, plot_model as plot_model_regression, create_model as create_model_regression
from dotenv import dotenv_values
from openai import OpenAI
import base64

env = dotenv_values(".env")

# OpenAI API key protection
if not st.session_state.get("openai_api_key"):#get albo przypuści skrypt albo wyrzuci none (nie wyrzuci błędu, tylko none)
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

openai_client = OpenAI(api_key=st.session_state['openai_api_key'])

def prepare_image_for_open_ai(image_path):
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    return f"data:image/png;base64,{image_data}"

def describe_image(image_path):
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Jesteś DataScientist. Stwórz opis wykresu, użyj języka przystępnego dla początkującego odbiorcy."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": prepare_image_for_open_ai(image_path),
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content

st.title("Wykrywacz najważniejszych zmiennych wpływających na wynik")

uploaded_file = st.sidebar.file_uploader("Wczytaj plik CSV", type="csv")

separators = [(".", "kropka"), (",", "przecinek"), (" ", "spacja"), (";", "średnik"), ("\t", "tabulator")]
descriptions = [desc for sep, desc in separators]
separator_row = st.sidebar.selectbox("Wybierz separator:", descriptions, key='unique_selectbox_row')
separator = [sep for sep, desc in separators if desc == separator_row][0]

@st.cache_data

def load_data(file, sep):
    return pd.read_csv(file, sep=sep)

if uploaded_file is not None:
    data = load_data(uploaded_file, separator)
    st.markdown("### **20 losowych wierszy:**")

    @st.cache_data
    def select_rows(dataframe):
        return dataframe.sample(20)

    rows = select_rows(data)
    st.write(rows)

    def is_column_selectable(column):
        number_of_rows = len(column)
        number_of_missing_rows = column.isnull().sum()
        return (number_of_missing_rows / number_of_rows) <= 0.2

    selectable_columns = [col for col in data.columns if is_column_selectable(data[col])]

    selected_column = st.sidebar.selectbox("Wybierz kolumnę do analizy, której braki stanowią mniej niż 20%:", selectable_columns)

    cleaned_data = data.dropna(subset=[selected_column])

    ignore_f = st.sidebar.multiselect("Wybierz, które kolumny mają być zignorowane przy analizie (opcjonalnie):", cleaned_data.columns.tolist(), [])

    modeling_choice = st.sidebar.radio('Wybierz typ modelowania:', ['klasyfikacja', 'regresja'])
    if st.button("Uruchom analizę"):
        with st.spinner("Proszę czekaj..."):

            if modeling_choice == 'klasyfikacja':
                clf = setup_classification(cleaned_data, target = selected_column, session_id=123, ignore_features=ignore_f)
                chosen_model = create_classification_model("lr")
                img = plot_classification_model(chosen_model, plot='feature', display_format='streamlit', save=True)
                #best_model = compare_classification_models()
                #img = plot_classification_model(best_model, plot='feature', display_format='streamlit', save=True)
                st.image(img)

            elif modeling_choice == 'regresja':
                reg = setup_regression(cleaned_data, target = selected_column, session_id=123, ignore_features=ignore_f)
                chosen_model = create_model_regression("lr")
                img = plot_model_regression(chosen_model, plot='feature', display_format='streamlit', save=True)
                #best_model = compare_models_regression()
                #img = plot_model_regression(best_model, plot='feature', display_format='streamlit', save=True)
                st.image(img)

        with st.spinner("Generuję opis wykresu..."):        
            if img:
                describe = describe_image("Feature Importance.png")
                st.write(describe)    
