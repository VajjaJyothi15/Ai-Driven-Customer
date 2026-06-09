import streamlit as st


def load_css():

    try:

        with open(
            "assets/styles.css",
            "r",
            encoding="utf-8"
        ) as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

    except:
        pass


def page_header(
    title,
    subtitle
):

    st.markdown(
        f"""
        <div class="main-title">
            {title}
        </div>

        <div class="sub-title">
            {subtitle}
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title):

    st.markdown(
        f"""
        <div class="section-title">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_row(metrics):

    cols = st.columns(len(metrics))

    for col, item in zip(cols, metrics):

        label = item[0]
        value = item[1]

        col.metric(
            label,
            value
        )