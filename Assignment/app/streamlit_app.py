"""Streamlit entry point for the Cityscapes segmentation demo."""

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Cityscapes Segmentation", layout="wide")
    st.title("Cityscapes Semantic Segmentation")
    st.info("The interactive prediction workflow will be implemented in a later phase.")


if __name__ == "__main__":
    main()
