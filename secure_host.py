#!/usr/bin/env python3
import http.server
import os
import ssl

# Port 6942 is for the website (frontend)
PORT = 6942
# Default to localhost for security; use SERVER_HOST env var for Docker/container deployments
SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_ADDRESS = (SERVER_HOST, PORT)

# Ensure certs exist
if not os.path.exists("key.pem") or not os.path.exists("cert.pem"):
    print("❌ Error: key.pem or cert.pem not found.")
    print("Run ./cert_setup.sh first!")
    exit(1)

httpd = http.server.HTTPServer(SERVER_ADDRESS, http.server.SimpleHTTPRequestHandler)

# Create an SSL context
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(certfile="cert.pem", keyfile="key.pem")

# Wrap the socket
httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

print(f"🔒 Secure Frontend running at https://localhost:{PORT}")
print(f"🔒 External Access: https://pascacktechnology.ddns.net:{PORT}")
print(
    "⚠️  NOTE: You will see a security warning in the browser. Click 'Advanced -> Proceed'."
)
httpd.serve_forever()
