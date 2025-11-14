Low-light Image Enhancement
===========================

Setup:
$ python -m venv venv
$ source venv/bin/activate       # Linux / macOS
# or
$ .\venv\Scripts\activate        # Windows PowerShell

$ pip install -r requirements.txt

Usage:
# process single image and save result
$ python main.py --input examples/dark1.jpg --output out.jpg

# process all images in a folder
$ python main.py --input_dir examples --out_dir results

# tune parameters:
$ python main.py --input examples/dark1.jpg --output out.jpg --h 10 --gamma 1.2 --cliplimit 2.0 --tilegrid 8

Open result with an image viewer or show inline if using Jupyter/VS Code interactive window.

Notes:
- The pipeline: (optional color denoise) -> CLAHE (per channel or LAB) -> gamma correction -> sharpening.
- If you install scikit-image, the script can use denoise_nl_means for grayscale images.
(venv) PS C:\Users\nisha\OneDrive\Desktop\low_light_ehnancement> python -m venv venv
>> .\venv\Scripts\activate
>> pip install streamlit opencv-python-headless numpy pillow piexif
>> streamlit run streamlit_app.py
# Low-Light Enhancement (Streamlit)

**Performance in Dark and Low-Light Environment** — a Streamlit app to enhance photos taken in poor lighting.

## Files
- `streamlit_app.py` — main Streamlit application (upload → denoise → enhance → download)
- `LowLight_Enhancement_Synopsis.docx` — project synopsis document
- `.gitignore`, `requirements.txt`, `LICENSE`, `README.md`

## Quick start (Windows)
```powershell
# create venv
python -m venv venv
.\venv\Scripts\activate

# install deps
pip install -r requirements.txt

# run the app
streamlit run streamlit_app.py
