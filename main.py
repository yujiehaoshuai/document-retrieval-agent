import os
from pathlib import Path
import streamlit as st
from retrieval import ingest, build_graph

st.set_page_config(page_title='本地文档问答', page_icon='📚')
st.title('本地文档检索问答 Agent')
st.caption('上传 UTF-8 TXT / Markdown，查看引用、检索分数和执行路径')
mode = st.sidebar.radio('模式', ['离线检索', '模型问答'])
uploads = st.file_uploader('资料文件（单文件最多1MB）', type=['txt','md'], accept_multiple_files=True)
files = []
for upload in uploads:
    if upload.size > 1024*1024:
        st.error(f'{upload.name} 超过1MB')
        st.stop()
    try:
        files.append((Path(upload.name).name, upload.getvalue().decode('utf-8-sig')))
    except UnicodeDecodeError:
        st.error('请使用 UTF-8 编码')
        st.stop()
if not uploads:
    files = [(p.name, p.read_text(encoding='utf-8')) for p in sorted(Path(__file__).with_name('docs').glob('*.md'))]
if sum(len(t) for _,t in files) > 2_000_000:
    st.error('资料总量超过演示限制')
    st.stop()
chunks = ingest(files)
st.caption(f'{len(files)} 个文件，去重后 {len(chunks)} 个片段。资料仅在本次运行内索引。')
consent = st.sidebar.checkbox('模型模式会向配置的模型服务发送问题及相关片段，我同意')
question = st.text_input('问题', value='发布失败怎么回滚？', max_chars=2000)
if st.button('查询'):
    model = None
    if mode == '模型问答':
        if not consent or not os.getenv('OPENAI_API_KEY'):
            st.error('请确认发送资料并配置 OPENAI_API_KEY。')
            st.stop()
        from langchain_openai import ChatOpenAI
        model = ChatOpenAI(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'), base_url=os.getenv('OPENAI_BASE_URL') or None, timeout=30, max_retries=1)
    try:
        result = build_graph(chunks, model).invoke({'question': question})
        st.write(result['answer'])
        st.caption('执行路径：'+' → '.join(result['trace']))
        with st.expander('检索证据'):
            st.json(result['hits'])
    except Exception as exc:
        st.error(f'请求失败：{type(exc).__name__}，请检查网络和模型配置。')
