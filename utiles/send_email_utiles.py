import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

import datetime
from secret import get_secret
# 发送邮件的函数
def send_email(text):
    date = str(datetime.date.today())
    # 发件人邮箱账号
    sender = get_secret()['sender']
    # 发件人邮箱密码或授权码
    password = get_secret()['password']
    # 收件人邮箱账号
    receiver = get_secret()['receiver']
    # 邮件主题
    subject = f'Today\'s A-Sahre Trading Recommendation: {date}'
    # 邮件正文
    content = text
    # 邮件对象，MIMEText()参数依次为邮件正文、邮件类型、编码方式
    message = MIMEText(content, 'plain', 'utf-8')
    # 设置发件人和收件人
    message['From'] = formataddr(['PASUW Predictor', sender])
    message['To'] = formataddr(['Receiver', receiver])
    # 设置邮件主题
    message['Subject'] = subject
    # 发送邮件的服务器和端口号
    server = smtplib.SMTP(get_secret()['server_id'], get_secret()['port'])

    server.ehlo()
    server.starttls()
    server.ehlo()
    
    # 登录发件人邮箱
    server.login(sender, password)
    # 发送邮件
    server.sendmail(sender, [receiver], message.as_string())
    # 退出邮件服务器
    server.quit()
    print('Email sent successfully!')




