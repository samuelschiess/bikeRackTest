from flask import Flask, jsonify, send_file, request
import pandas as pd
import os

app = Flask(__name__, static_folder="static")

@app.route("/")
def index():
    return send_file("static/index.html")

@app.route("/api/racks")
def get_racks():
    if not os.path.exists("bike_racks_output.csv"):
        return jsonify([])
        
    df = pd.read_csv("bike_racks_output.csv")
    df = df.fillna("")
    return jsonify(df.to_dict(orient="records"))

@app.route("/api/image")
def serve_image():
    # Utilizing query parameters gracefully sidesteps severe filesystem routing logic bugs intrinsically tied to Windows \ parsing.
    filepath = request.args.get("path")
    if not filepath:
        return "No explicit physical path mathematically provided by the client DOM.", 400
        
    if os.path.exists(filepath):
        return send_file(filepath)
    return "Local structural image not physically found.", 404

if __name__ == "__main__":
    print("\n[+] Map Engine Topology Initialized!")
    print("[+] Open http://localhost:5000 in your browser.")
    app.run(debug=True, port=5000)
