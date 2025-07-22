progress_html = '''
<div class="progress">
  *text*
</div>
'''


def make_progress_html(number, text):
    return progress_html.replace('*text*', f'{text} {number}%')