import os
import glob
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def get_most_recent_file(folder_path, extensions=[".png", ".jpg", "*.jpeg"]):
    """Get the most recent file from a folder with specific extensions."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
    # Filter out invalid paths (e.g., directories or system files)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return None  # No valid files found
    return max(files, key=os.path.getmtime)  # Find the most recent file


def send_email_alert_with_latest_image(subject, body, to_email, folder_path):
    sender_email = "capstone.pk05@gmail.com"
    sender_password = "isou uvnt cvod vait"  # Replace with your App Password

    # Get the most recent image file
    image_path = get_most_recent_file(folder_path, "*.jpg")  # Change to your image type
    if not image_path:
        print(f"No image files found in folder: {folder_path}")
        return

    # Create the email message
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = to_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    # Attach the image file
    try:
        with open(image_path, 'rb') as image_file:
            mime_base = MIMEBase('application', 'octet-stream')
            mime_base.set_payload(image_file.read())
            encoders.encode_base64(mime_base)
            mime_base.add_header(
                'Content-Disposition',
                f'attachment; filename={os.path.basename(image_path)}'
            )
            message.attach(mime_base)
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return
    except Exception as e:
        print(f"Error attaching the file: {e}")
        return

    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        print(f"Email with image sent successfully! Attached: {image_path}")
    except Exception as e:
        print(f"Failed to send email: {e}")

# # Example call
# send_email_alert_with_latest_image(
#     subject="Anomaly Detected!",
#     body="An anomaly was detected. Please see the attached image for details.",
#     to_email="rohanrj389@gmail.com",
#     folder_path=r"pictures"  # Replace with your folder path
# )