import os
import sys

import logging
import pandas as pd
from json import JSONDecodeError
from pathlib import Path
import streamlit as st
from annotated_text import annotation
from markdown import markdown

import SessionState
from utils import gunicorn_is_ready, query, send_feedback, upload_doc,  get_backlink
from streamlit.components.v1 import html
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb


DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "How to start Sidekiq worker on Ubuntu VPS?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", " ")

DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", 2))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", 6))

EVAL_LABELS = os.getenv("EVAL_FILE", Path(__file__).parent / "random_examples.csv")

DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def _max_width_():
    max_width_str = f"max-width: 890px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))

def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)

def layout(*args):
    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;}
     .stApp { bottom: 80px; }
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1
    )

    # 分割线
    style_hr = styles(
        display="block",
        margin=px(0, 0, 0, 0),
        border_style="inset",
        border_width=px(2)
    )

    # 修改p标签内文字的style
    body = p(
        id='myFooter',
        style=styles(
            margin=px(0, 0, 0, 0),
            # 通过调整padding自行调整上下边距以达到满意效果
            padding=px(5),
            # 调整字体大小
            font_size="0.8rem",
            color="rgb(51,51,51)"
        )
    )
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

    # js获取背景色 由于st.markdown的html实际上存在于iframe, 所以js检索的时候需要window.parent跳出到父页面
    # 使用getComputedStyle获取所有stApp的所有样式，从中选择bgcolor
    js_code = '''
    <script>
    function rgbReverse(rgb){
        var r = rgb[0]*0.299;
        var g = rgb[1]*0.587;
        var b = rgb[2]*0.114;

        if ((r + g + b)/255 > 0.5){
            return "rgb(49, 51, 63)"
        }else{
            return "rgb(250, 250, 250)"
        }

    };
    var stApp_css = window.parent.document.querySelector("#root > div:nth-child(1) > div > div > div");
    window.onload = function () {
        var mutationObserver = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    /************************当DOM元素发送改变时执行的函数体***********************/
                    var bgColor = window.getComputedStyle(stApp_css).backgroundColor.replace("rgb(", "").replace(")", "").split(", ");
                    var fontColor = rgbReverse(bgColor);
                    var pTag = window.parent.document.getElementById("myFooter");
                    pTag.style.color = fontColor;
                    /*********************函数体结束*****************************/
                });
            });

            /**Element**/
            mutationObserver.observe(stApp_css, {
                attributes: true,
                characterData: true,
                childList: true,
                subtree: true,
                attributeOldValue: true,
                characterDataOldValue: true
            });
    }


    </script>
    '''
    html(js_code)

def footer():
    # use relative path to show my png instead of url

    myargs = []
    layout(*myargs)

def main():
    #Set page
    st.set_page_config(page_title='DeployQA- Answering Software Deployment Questions', page_icon="🎈")
    _max_width_()
    local_css("style.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
    hide_streamlit_style = """
                    <style>
                    #MainMenu {visibility: hidden;}
                    footer {visibility: hidden;}
                    </style>
                    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Persistent state
    state = SessionState.get(
        question=DEFAULT_QUESTION_AT_STARTUP,
        answer=DEFAULT_ANSWER_AT_STARTUP,
        results=None,
        raw_json=None,
        random_question_requested=False
    )

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        state.answer = None
        state.results = None
        state.raw_json = None

    # Title
    c30, c31, c32 = st.columns([2, 4, 2])

    with c31:
        st.image("logo.png", width=450)

        # st.write("# 🔑 DeployQA Bot")
        # st.title('—— Answering Software Deployment Questions')


    # Sidebar
    st.sidebar.header("Options")

    top_k_reader = st.sidebar.slider(
        "The number of candidate documents",
        min_value=1,
        max_value=10,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
        help='This number indicates top-k documents selected by retriever')

    top_k_retriever = st.sidebar.slider(
        "The max number of answers",
        min_value=1,
        max_value=15,
        value=DEFAULT_NUMBER_OF_ANSWERS,
        step=1,
        on_change=reset_results,
        help='This number indicates the maximum of answers returned by the reader.'
    )
    debug = 0
    eval_mode = 1

    # File upload block
    if not DISABLE_FILE_UPLOAD:
        st.sidebar.write("## File Upload:")
        data_files = st.sidebar.file_uploader("", type=["pdf", "txt"], accept_multiple_files=True)
        for data_file in data_files:
            # Upload file
            if data_file:
                raw_json = upload_doc(data_file)
                st.sidebar.write(str(data_file.name) + " &nbsp;&nbsp; ✅ ")
                if debug:
                    st.subheader("REST API JSON response")
                    st.sidebar.write(raw_json)

        with st.sidebar.expander("ℹ️ - About this bot", expanded=False):

            st.write(
                """
    -   DeployQA is a QA Bot that automatically answers software deployment questions over user manuals and Stack Overflow posts.
    -   DeployQA leverages a retrieval-and-reader framework where a retriever searches for candidate documents, and a reader predicts the answer span from the selected documents using a domain-adapted RoBERTa model.

        	    """
            )

    st.sidebar.markdown(
        f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .footer {{
            text-align: center;
        }}
        .footer h4 {{
            margin: 0.1rem;
            padding:0;
        }}
        footer {{
            opacity: 0;
        }}
    </style>
    <div class="footer">
        <hr />
        <small>Copyright © 2022 <br/>Intelligent Software Engineering Lab, <br/>Shanghai Jiao Tong  University.<br/> All rights reserved.</small>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Load csv into pandas dataframe
    try:
        df = pd.read_csv(EVAL_LABELS, sep=";")
    except Exception:
        st.error(f"The eval file was not found.")
        sys.exit(f"The eval file was not found under `{EVAL_LABELS}`.")

    # Search bar

    # selected = st.text_input("", "Search...")
    # button_clicked = st.button("OK")
    # icon("search")
    question = st.text_input(
                             label='',
                             value=state.question,
                             max_chars=200,
                             on_change=reset_results,
                             help='Enter your question'
                             )
    col1, col2 = st.columns([1, 1])
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)
    col2.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

    # Run button
    run_pressed = col1.button("Run")

    # Get next random question from the CSV
    if col2.button("Random question"):
        reset_results()
        new_row = df.sample(1)
        while new_row["Question Text"].values[
            0] == state.question:  # Avoid picking the same question twice (the change is not visible on the UI)
            new_row = df.sample(1)
        state.question = new_row["Question Text"].values[0]
        state.answer = new_row["Answer"].values[0]
        state.random_question_requested = True
        # Re-runs the script setting the random question as the textbox value
        # Unfortunately necessary as the Random Question button is _below_ the textbox
        raise st.script_runner.RerunException(st.script_request_queue.RerunData(None))
    else:
        state.random_question_requested = False

    run_query = (run_pressed or question != state.question) and not state.random_question_requested

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; DeployQA Bot is starting..."):
        if not gunicorn_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is DeployQA Bot running?")
            run_query = False
            reset_results()

    # Get results for query
    if run_query and question:
        reset_results()
        state.question = question
        with st.spinner(
                "🚀 &nbsp;&nbsp; Performing neural machine reading at scale... \n "
        ):
            try:
                state.results, state.raw_json = query(question, top_k_reader=top_k_reader,
                                                      top_k_retriever=top_k_retriever)
            except JSONDecodeError as je:
                st.error("👓 &nbsp;&nbsp; An error occurred reading the results. Is the document store working?")
                return
            except Exception as e:
                logging.exception(e)
                if "The server is busy processing requests" in str(e) or "503" in str(e):
                    st.error("🧑‍🌾 &nbsp;&nbsp; All our workers are busy! Try again later.")
                else:
                    st.error("🐞 &nbsp;&nbsp; An error occurred during the request.")
                return

    if state.results:

        st.write("___")
        for count, result in enumerate(state.results):
            if result["answer"]:
                source = ""
                url, title = get_backlink(result)
                if url and title:
                    source = f"[{result['document']['meta']['title']}]({result['document']['meta']['url']})"
                else:
                    source = f"{result['source']}"

                if float(result['relevance']) < 1:
                    continue

                answer, context = result["answer"], result["context"]
                start_idx = context.find(answer)
                end_idx = start_idx + len(answer)
                st.write(markdown(context[:start_idx] + str(
                    annotation(answer, "ANSWER", background="#ecfaf4", border="1px solid rgb(73, 204, 144)")) + context[
                                                                                                                end_idx:]),
                         unsafe_allow_html=True)

                st.write("**Confidence:** ", result["relevance"])

            else:
                st.info(
                    "🤔 &nbsp;&nbsp;  DeployQA Bot is unsure whether any of the documents contain an answer to your question. Try to reformulate it!")
                st.write("**Confidence:** ", result["relevance"])

            if result["answer"] is None:
                button_col11, button_col2, button_col3, _ = st.columns([1, 1, 1, 6])


            if result["answer"]:
                # Define columns for buttons
                is_correct_answer = None
                is_correct_document = None

                button_col1, button_col2, button_col3, _ = st.columns([2, 2, 4, 4])
                if button_col1.button("👍Correct", key=f"{result['context']}{count}1", help="Correct answer"):
                    is_correct_answer = True
                    is_correct_document = True

                if button_col2.button("👎Wrong", key=f"{result['context']}{count}2",
                                      help="Wrong answer and wrong passage"):
                    is_correct_answer = False
                    is_correct_document = False

                if button_col3.button("👎👍With some problems", key=f"{result['context']}{count}3",
                                      help="With some problems"):
                    is_correct_answer = False
                    is_correct_document = True

                if is_correct_answer is not None and is_correct_document is not None:
                    try:
                        send_feedback(
                            query=question,
                            answer_obj=result["_raw"],
                            is_correct_answer=is_correct_answer,
                            is_correct_document=is_correct_document,
                            document=result["document"]
                        )
                        st.success("✨ &nbsp;&nbsp; Thanks for your feedback! &nbsp;&nbsp; ✨")
                    except Exception as e:
                        logging.exception(e)
                        st.error("🐞 &nbsp;&nbsp; An error occurred while submitting feedback!")

            st.write("___")

        if debug:
            st.subheader("REST API JSON response")
            st.write(state.raw_json)
            # st.write(state.results)


main()
