import certifi
import os

def patch_ssl(cert_path):
    os.environ["REQUESTS_CA_BUNDLE"] = cert_path
    os.environ["SSL_CERT_FILE"] = cert_path

    cafile = certifi.where()
    with open(cert_path, 'rb') as infile:
        custom_ca = infile.read()
    with open(cafile, 'ab') as outfile:
        outfile.write(custom_ca)