import cv2
import base64
import os

def get_b64(img_path, rect=None):
    if not os.path.exists(img_path): return ""
    img = cv2.imread(img_path)
    if img is None: return ""
    if rect:
        x, y, w, h = rect
        img = img[y:y+h, x:x+w]
    img = cv2.resize(img, (200, 200))
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

pre_img = "/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929095743_0311_D.JPG"
post_img = "/Volumes/T9/ICML/Post_def_rgb_part1/DJI_20250929124149_0029_D.JPG"

# Try to find part2 pre if part1 is missing or just use part1
if not os.path.exists(pre_img):
    pre_img = "/Volumes/T9/ICML/part 2_pre_def_rgb/DJI_20250929093936_0722_D.JPG"

b64_pre = get_b64(pre_img)
b64_post = get_b64(post_img)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
    @import url('https://fonts.googleapis.com/css2?family=Computer+Modern+Serif:ital,wght@0,400;0,700;1,400&family=Fira+Code:wght@400;600&display=swap');
    
    body {{
        margin: 0; padding: 40px;
        background-color: #f8f9fa; /* Light mode for paper */
        font-family: 'Times New Roman', Times, serif; /* Journal style */
        display: flex; justify-content: center; align-items: center;
        min-height: 100vh;
    }}
    .diagram-container {{
        position: relative;
        width: 1400px;
        height: 750px;
        background: #ffffff;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        padding: 40px;
        box-sizing: border-box;
    }}
    .title {{
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 30px;
        color: #111827;
        font-family: 'Computer Modern Serif', serif;
    }}
    .group-box {{
        position: absolute;
        border: 2px dashed;
        border-radius: 12px;
        padding: 15px;
        box-sizing: border-box;
        background-color: rgba(255,255,255,0.8);
    }}
    .group-title {{
        position: absolute;
        top: -14px;
        left: 20px;
        background: white;
        padding: 0 10px;
        font-weight: bold;
        font-size: 16px;
        font-family: 'Computer Modern Serif', serif;
    }}
    .node {{
        position: absolute;
        background: white;
        border: 2px solid;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        font-size: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        z-index: 10;
        font-family: 'Computer Modern Serif', serif;
    }}
    .node img {{
        width: 120px;
        height: 120px;
        border-radius: 6px;
        margin-bottom: 10px;
        object-fit: cover;
    }}
    
    /* Colors */
    .c-input {{ border-color: #3b82f6; color: #1e3a8a; }}
    .c-cv {{ border-color: #8b5cf6; color: #4c1d95; }}
    .c-quant {{ border-color: #10b981; color: #064e3b; }}
    .c-out {{ border-color: #f59e0b; color: #78350f; }}
    
    .g-cv {{ border-color: #c4b5fd; background-color: #f5f3ff; }}
    .g-quant {{ border-color: #6ee7b7; background-color: #ecfdf5; }}
    .g-out {{ border-color: #fcd34d; background-color: #fffbeb; }}

    /* Layout */
    /* Input */
    #n-img1 {{ top: 180px; left: 50px; width: 160px; height: 180px; border-color: #3b82f6; }}
    #n-img2 {{ top: 400px; left: 50px; width: 160px; height: 180px; border-color: #3b82f6; }}
    
    /* CV Group */
    #g-cv {{ top: 120px; left: 280px; width: 600px; height: 260px; }}
    #n-morph {{ top: 160px; left: 310px; width: 150px; min-height: 70px; }}
    #n-mask {{ top: 160px; left: 500px; width: 150px; min-height: 70px; }}
    #n-detect {{ top: 160px; left: 690px; width: 150px; min-height: 70px; }}
    
    #n-feat1 {{ top: 270px; left: 310px; width: 200px; min-height: 80px; }}
    #n-feat2 {{ top: 270px; left: 540px; width: 200px; min-height: 80px; }}

    /* Quant Group */
    #g-quant {{ top: 120px; left: 930px; width: 380px; height: 260px; }}
    #n-mi {{ top: 150px; left: 960px; width: 320px; min-height: 60px; }}
    #n-vqc {{ top: 240px; left: 960px; width: 320px; min-height: 80px; }}
    
    /* Out Group */
    #g-out {{ top: 430px; left: 280px; width: 1030px; height: 220px; }}
    #n-svm {{ top: 470px; left: 960px; width: 320px; min-height: 70px; }}
    #n-class {{ top: 560px; left: 960px; width: 320px; min-height: 70px; }}
    #n-density {{ top: 560px; left: 690px; width: 220px; min-height: 70px; }}

    svg {{
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 1;
    }}
    path {{
        fill: none;
        stroke: #64748b;
        stroke-width: 2.5;
        marker-end: url(#arrow);
    }}
    .dashed {{ stroke-dasharray: 6 6; }}
    
    .latex {{ font-family: 'Computer Modern Serif', serif; font-style: italic; }}
    .code {{ font-family: 'Fira Code', monospace; font-size: 13px; background: #f1f5f9; padding: 2px 4px; border-radius: 4px; border: 1px solid #cbd5e1; }}
</style>
</head>
<body>

<div class="diagram-container">
    <div class="title">QuantumHarvest: Hybrid VAC-QFS Architecture</div>

    <!-- Groups -->
    <div id="g-cv" class="group-box g-cv">
        <div class="group-title" style="color:#6d28d9;">Spatial Context & Classical Features</div>
    </div>
    
    <div id="g-quant" class="group-box g-quant">
        <div class="group-title" style="color:#047857;">Quantum Feature Selection (VAC-QFS)</div>
    </div>
    
    <div id="g-out" class="group-box g-out">
        <div class="group-title" style="color:#b45309;">Classification & Synthesis</div>
    </div>

    <!-- Nodes -->
    <div id="n-img1" class="node c-input">
        <img src="data:image/jpeg;base64,{b64_pre}" alt="Pre">
        <b>Pre-Defoliation</b><br>
        <span style="font-size:12px; color:#666;">Camouflaged</span>
    </div>
    <div id="n-img2" class="node c-input">
        <img src="data:image/jpeg;base64,{b64_post}" alt="Post">
        <b>Post-Defoliation</b><br>
        <span style="font-size:12px; color:#666;">Exposed</span>
    </div>

    <div id="n-morph" class="node c-cv"><b>Morphological</b><br>Top-Hat & CLAHE</div>
    <div id="n-mask" class="node c-cv"><b>HSV Masking</b><br>Otsu Threshold</div>
    <div id="n-detect" class="node c-cv"><b>Boll Detection</b><br>Distance Transform</div>

    <div id="n-feat1" class="node c-cv"><b>Textural Features</b><br>GLCM <span class="latex">(Contrast, Energy)</span></div>
    <div id="n-feat2" class="node c-cv"><b>Spectral Indices</b><br><span class="code">ExG, RBR, NDI</span></div>

    <div id="n-mi" class="node c-quant"><b>Mutual Information Filter</b><br><span class="latex">Top 6 Candidate Selection</span></div>
    
    <div id="n-vqc" class="node c-quant" style="display:flex; flex-direction:row; justify-content:space-around;">
        <div style="text-align:center;">
            <b>Encoding</b><br>
            <span class="code">ZZFeatureMap</span>
        </div>
        <div style="text-align:center; border-left:1px solid #a7f3d0; padding-left:10px;">
            <b>Ansatz</b><br>
            <span class="code">RealAmplitudes</span>
        </div>
        <div style="text-align:center; border-left:1px solid #a7f3d0; padding-left:10px;">
            <b>Optimizer</b><br>
            <span class="code">COBYLA</span>
        </div>
    </div>

    <div id="n-svm" class="node c-out"><b>Quantum Subset RBF SVM</b><br>Optimal Manifold <span class="latex">C(6,4)</span> Space</div>
    <div id="n-class" class="node c-out" style="background:#fef3c7; font-weight:bold; font-size:18px;">Defoliation Readiness Vertex</div>
    
    <div id="n-density" class="node c-out"><b>Spatial Yield Map</b><br>Cotton Density Estimate</div>

    <!-- Edges -->
    <svg>
        <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b" />
            </marker>
        </defs>

        <!-- Image to CV -->
        <path d="M 210 270 C 250 270, 250 195, 310 195" />
        <path d="M 210 490 C 250 490, 250 310, 310 310" />
        <path d="M 230 380 L 310 310" /> <!-- Fake connection from mid -->
        
        <path d="M 210 270 C 250 270, 250 310, 310 310" />
        <path d="M 210 490 C 250 490, 250 195, 310 195" />

        <!-- CV flow -->
        <path d="M 460 195 L 500 195" />
        <path d="M 650 195 L 690 195" />
        <path d="M 510 310 L 540 310" />
        
        <!-- CV to Quant -->
        <path d="M 740 310 C 850 310, 850 180, 960 180" />
        
        <!-- Quant internal -->
        <path d="M 1120 210 L 1120 240" />

        <!-- Quant to Out -->
        <path d="M 1120 320 C 1120 400, 1120 400, 1120 470" />
        
        <!-- Out internal -->
        <path d="M 1120 540 L 1120 560" />
        <path d="M 765 230 L 765 560" />

    </svg>
    
    <div style="position:absolute; bottom:20px; right:40px; font-size:12px; color:#9ca3af; font-family:'Fira Code', monospace;">
        Fig. 1: Hybrid QML pipeline processing UAV imagery into quantum-optimized harvest intelligence.
    </div>
</div>

</body>
</html>
"""

with open("/Volumes/T9/CottonDefoliationApp/architecture_diagram.html", "w") as f:
    f.write(html)
print("Diagram HTML generated successfully.")
