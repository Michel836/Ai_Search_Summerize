import streamlit as st

from generate import ContextAwareGenerator
from train import main as train_model


st.title("Sentiment + Contextual Response App")

if "generator" not in st.session_state:
    st.session_state.generator = ContextAwareGenerator()

text_input = st.text_input("Enter your query")

if st.button("Train Model"):
    st.write("Training... (this may take a while)")
    train_model()
    st.success("Training complete")

if st.button("Build Retrieval"):
    st.session_state.generator.build_retrieval(["Example document 1", "Example document 2"])
    st.success("Retrieval built")

if st.button("Generate Response"):
    if text_input:
        response = st.session_state.generator.generate(text_input)
        st.write(response)
    else:
        st.warning("Please enter text")
