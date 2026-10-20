from http.server import HTTPServer
from api.index import handler

def run(port=8000):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, handler)
    print(f"Starting server on http://127.0.0.1:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

if __name__ == "__main__":
    run()
