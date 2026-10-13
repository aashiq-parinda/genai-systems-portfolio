#!/usr/bin/env python3
"""
Simple HTTP server to launch and view the ClaudeOps Studio frontend.
"""

import http.server
import socketserver
import os
import webbrowser

PORT = 8899
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def run():
    global PORT
    for port_attempt in range(PORT, PORT + 20):
        try:
            with socketserver.TCPServer(("", port_attempt), Handler) as httpd:
                url = f"http://localhost:{port_attempt}"
                print(f"🚀 ClaudeOps Studio Frontend live at: {url}")
                print("Press Ctrl+C to stop the server.")
                httpd.serve_forever()
                break
        except OSError:
            continue

if __name__ == "__main__":
    run()
