from streamlit.testing.v1 import AppTest
app = AppTest.from_file('main.py', default_timeout=30).run()
assert not app.exception
app.button[0].click().run()
assert not app.exception
assert any('deployment.md' in item.value for item in app.markdown)
app.text_input[0].set_value('火星天气')
app.button[0].click().run()
assert any('没有找到' in item.value for item in app.markdown)
print('PASS: document UI retrieval and rejection')
