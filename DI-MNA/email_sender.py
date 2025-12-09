import os
from dotenv import load_dotenv
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email import message
import smtplib

load_dotenv()

senha = os.environ.get("senha")
reme = os.environ.get("remetente")
dest = os.environ.get("destinatarios")

def send_result(file_path, cut_sol, cut_comb_nodes, min_comb_threads):

    msg = MIMEMultipart()

    msg['Subject'] = "Resultados máquina I5_1"
    destinatarios = dest.split(",")
    msg["To"] = ", ".join(destinatarios)
    msg["From"] = reme
    msg.add_header("Content-Type", "text/html")
    file_name = os.path.basename(file_path)

    corpo_email = f"""
        <p>
            <strong> {file_name} </strong><br><br>

            cut_sol : {cut_sol} <br>
            cut_comb_nodes : {cut_comb_nodes} <br>
            min_comb_threads : {min_comb_threads} <br>
        </p>
    """
    msg.attach(MIMEText(corpo_email, "html"))

    with open(file_path, "rb") as attachment:
        
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())

    encoders.encode_base64(part)

    
    part.add_header(
        "Content-Disposition",
        f"attachment; filename= {file_name}",
    )

    msg.attach(part)

    s = smtplib.SMTP('smtp.gmail.com: 587')
    s.starttls()
    s.login(msg['From'],senha)
    s.sendmail(msg["From"], destinatarios, msg.as_string().encode('utf-8'))