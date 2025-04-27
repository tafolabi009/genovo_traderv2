# utils/notifier.py

import smtplib
import ssl
from email.message import EmailMessage
import logging

# Get a logger instance (assuming logger is set up elsewhere)
# If run standalone, basic logging will be used.
logger = logging.getLogger("genovo_traderv2")
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def send_email_notification(subject, body, config):
    """
    Sends an email notification using SMTP based on the provided configuration.

    Args:
        subject (str): The subject line of the email.
        body (str): The main content/body of the email.
        config (dict): The main application configuration dictionary, expected
                       to contain a 'notifications' sub-dictionary.

    Returns:
        bool: True if the email was sent successfully, False otherwise.
    """
    notify_config = config.get('notifications', {})
    is_enabled = notify_config.get('email_enabled', False)

    if not is_enabled:
        logger.debug("Email notifications are disabled in the configuration.")
        return False # Not an error, just disabled

    # --- Get Email Credentials and Settings ---
    sender_email = notify_config.get('email_address')
    # IMPORTANT: Use an App Password for Gmail if 2FA is enabled!
    password = notify_config.get('email_password')
    smtp_server = notify_config.get('email_smtp_server', 'smtp.gmail.com')
    smtp_port = notify_config.get('email_smtp_port', 587) # 587 for TLS, 465 for SSL
    recipient_emails_str = notify_config.get('email_recipient')

    if not all([sender_email, password, smtp_server, smtp_port, recipient_emails_str]):
        logger.error("Email notification failed: Missing configuration details (address, password, server, port, or recipient).")
        return False

    # Split recipient string into a list, removing duplicates and whitespace
    recipient_list = list(set([email.strip() for email in recipient_emails_str.split(',') if email.strip()]))
    if not recipient_list:
        logger.error("Email notification failed: No valid recipient emails found.")
        return False

    # --- Create Email Message ---
    msg = EmailMessage()
    msg['Subject'] = f"[GenovoTraderV2] {subject}"
    msg['From'] = sender_email
    # Join list for the 'To' header, actual sending happens individually or via BCC later if needed
    msg['To'] = ', '.join(recipient_list)
    msg.set_content(body)

    # --- Send Email ---
    logger.info(f"Attempting to send email notification to {recipient_list}...")
    context = ssl.create_default_context() # Create a secure SSL context
    server = None # Initialize server variable

    try:
        # Try connecting with STARTTLS (common for port 587)
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=30) # 30s timeout
        server.ehlo() # Identify client to server
        server.starttls(context=context) # Secure the connection
        server.ehlo() # Re-identify client over secure connection
        server.login(sender_email, password)
        # Send to all recipients specified in the 'To' header
        server.send_message(msg)
        logger.info("Email notification sent successfully.")
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("Email notification failed: SMTP Authentication Error. Check email address and password (use App Password for Gmail if 2FA is enabled).", exc_info=True)
        return False
    except smtplib.SMTPServerDisconnected:
         logger.error("Email notification failed: Server disconnected unexpectedly. Check server/port or network.", exc_info=True)
         return False
    except smtplib.SMTPException as e:
        logger.error(f"Email notification failed: An SMTP error occurred: {e}", exc_info=True)
        # Optional: Try SSL connection on port 465 as a fallback?
        # try:
        #     logger.info("Retrying with SMTP_SSL on port 465...")
        #     server = smtplib.SMTP_SSL(smtp_server, 465, context=context, timeout=30)
        #     server.login(sender_email, password)
        #     server.send_message(msg)
        #     logger.info("Email notification sent successfully via SMTP_SSL.")
        #     return True
        # except Exception as e_ssl:
        #      logger.error(f"Email notification failed via SMTP_SSL as well: {e_ssl}", exc_info=True)
        #      return False
        return False
    except TimeoutError:
         logger.error("Email notification failed: Connection timed out. Check server/port and network connectivity.", exc_info=True)
         return False
    except Exception as e:
        logger.error(f"Email notification failed: An unexpected error occurred: {e}", exc_info=True)
        return False
    finally:
        if server:
            try:
                server.quit()
            except smtplib.SMTPServerDisconnected:
                 pass # Ignore if already disconnected
            except Exception as e_quit:
                 logger.warning(f"Error during SMTP server quit: {e_quit}")


# Example Usage (can be removed)
if __name__ == '__main__':
    # Create a dummy config for testing
    test_config = {
        'notifications': {
            'email_enabled': True,
            'email_address': "your_sender_email@gmail.com", # Replace
            'email_password': "your_app_password", # Replace with App Password
            'email_smtp_server': "smtp.gmail.com",
            'email_smtp_port': 587,
            'email_recipient': "recipient1@example.com, recipient2@example.com" # Replace
        }
    }
    print("Testing email notification...")
    success = send_email_notification(
        subject="Test Notification",
        body="This is a test email from the GenovoTraderV2 notifier.",
        config=test_config
    )
    if success:
        print("Test email sent (check recipient inboxes).")
    else:
        print("Test email failed. Check logs and configuration.")

