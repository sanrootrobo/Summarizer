import os
import argparse
import pypandoc
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_google_api():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def convert_md_to_odt(md_file):
    odt_file = md_file.replace('.md', '.odt')
    output = pypandoc.convert_file(md_file, 'odt', outputfile=odt_file)
    print(f"✅ Converted {md_file} to {odt_file}")
    return odt_file

def upload_odt_to_gdoc(drive_service, odt_file, title):
    file_metadata = {
        'name': title,
        'mimeType': 'application/vnd.google-apps.document'
    }
    media = MediaFileUpload(odt_file, mimetype='application/vnd.oasis.opendocument.text')
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"✅ Uploaded and converted to Google Doc: {title} (ID: {file.get('id')})")

def main():
    parser = argparse.ArgumentParser(description="Upload markdown files as Google Docs (via ODT)")
    parser.add_argument('files', metavar='file', type=str, nargs='+', help='Markdown files to upload')
    args = parser.parse_args()

    creds = authenticate_google_api()
    drive_service = build('drive', 'v3', credentials=creds)

    for md_file in args.files:
        if not os.path.isfile(md_file):
            print(f"❌ File not found: {md_file}")
            continue
        odt_file = convert_md_to_odt(md_file)
        title = os.path.splitext(os.path.basename(md_file))[0]
        upload_odt_to_gdoc(drive_service, odt_file, title)

if __name__ == '__main__':
    from googleapiclient.http import MediaFileUpload
    main()

