# Run a test server.
from app import app

app.run(host='0.0.0.0', port=8081, debug=True)
app.run(host='0.0.0.0', port=8080, debug=True)
