import base64
import cv2
import os

def get_b64(img_path):
    if not os.path.exists(img_path): return ""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (260, 260))
    _, b = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(b).decode()

b64_pre = get_b64("/Volumes/T9/ICML/Part_one_pre_def_rgb/DJI_20250929095743_0311_D.JPG")
if not b64_pre: b64_pre = get_b64("/Volumes/T9/ICML/part 2_pre_def_rgb/DJI_20250929093936_0722_D.JPG")
b64_post = get_b64("/Volumes/T9/ICML/Post_def_rgb_part1/DJI_20250929124149_0029_D.JPG")

html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&family=Inter:wght@400;600;800&family=Playfair+Display:ital,wght@1,600&display=swap" rel="stylesheet">
    <style>
        body {{
            background: #0B0E14; /* Deep academic poster dark */
            margin: 0; padding: 60px;
            display: flex; justify-content: center; align-items: center;
            font-family: 'Inter', sans-serif;
        }}
        .canvas {{ relative; width: 2400px; height: 1150px; background: #0B0E14; }}
        
        .header {{
            position: absolute; top: 0; left: 0; width: 100%; text-align: center;
            font-size: 52px; font-weight: 800; color: #F8FAFC; letter-spacing: 4px;
            text-transform: uppercase; z-index: 100;
        }}
        .header span {{ color: #EAB308; }} /* Gold accent */
        .subtitle {{
            font-family: 'Fira Code', monospace; font-size: 20px; color: #94A3B8;
            margin-top: 15px; letter-spacing: 2px;
        }}

        /* SVG Base */
        svg {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1; }}
        
        /* Wires (Like DiffLogic circuits) */
        .wire-g {{ fill: none; stroke: #EAB308; stroke-width: 5; marker-end: url(#arrow-g); stroke-linejoin: round; filter: drop-shadow(0 0 6px rgba(234,179,8,0.6)); }}
        .wire-p {{ fill: none; stroke: #A855F7; stroke-width: 5; marker-end: url(#arrow-p); stroke-linejoin: round; filter: drop-shadow(0 0 6px rgba(168,85,247,0.6)); }}
        .wire-b {{ fill: none; stroke: #38BDF8; stroke-width: 5; marker-end: url(#arrow-b); stroke-linejoin: round; filter: drop-shadow(0 0 6px rgba(56,189,248,0.6)); }}
        .wire-straight {{ fill: none; stroke: #EAB308; stroke-width: 5; marker-end: url(#arrow-g); filter: drop-shadow(0 0 6px rgba(234,179,8,0.6)); }}

        /* Groups */
        .g-rect {{ fill: #0F131E; stroke: #334155; stroke-width: 4; stroke-dasharray: 12 8; rx: 16; }}
        .g-bg {{ fill: #0B0E14; }}
        .g-title {{ font-size: 24px; font-weight: 800; text-transform: uppercase; letter-spacing: 3px; }}

        /* Nodes */
        .node {{
            position: absolute; z-index: 10;
            display: flex; flex-direction: column; justify-content: center; align-items: center;
            background: #151A28; border: 3px solid #475569; border-radius: 12px;
            color: #F8FAFC; text-align: center; padding: 25px; box-sizing: border-box;
            box-shadow: 0 15px 35px rgba(0,0,0,0.4), inset 0 2px 0 rgba(255,255,255,0.05);
        }}
        
        /* Node Colors matching DiffLogic/academic aesthetic */
        .n-cv {{ border-color: #3B82F6; box-shadow: 0 0 20px rgba(59,130,246,0.15); }}
        .n-feat {{ border-color: #10B981; box-shadow: 0 0 20px rgba(16,185,129,0.15); }}
        .n-quant {{ border-color: #A855F7; box-shadow: 0 0 20px rgba(168,85,247,0.15); }}
        .n-out {{ border-color: #F59E0B; background: #1A1614; box-shadow: 0 0 20px rgba(245,158,11,0.15); }}
        
        .title {{ font-weight: 800; font-size: 26px; margin-bottom: 20px; }}
        
        /* Math/Code Text */
        .math {{ font-family: 'Playfair Display', serif; font-size: 26px; color: #FBBF24; letter-spacing: 1px; }}
        .math sub {{ font-size: 14px; }}
        .math sup {{ font-size: 14px; }}
        .code {{ font-family: 'Fira Code', monospace; font-size: 18px; color: #38BDF8; background: #0B0E14; padding: 6px 12px; border-radius: 6px; border: 1px solid #1E293B; margin-top: 15px; border-left: 4px solid #A855F7; }}

        /* Embedded Images */
        .img-box {{ border: 3px solid #334155; border-radius: 12px; padding: 15px; background: #0B0E14; align-items: center; display: flex; flex-direction: column; }}
        .img-box img {{ width: 260px; height: 260px; border-radius: 6px; margin-bottom: 15px; border: 1px solid #1E293B; }}
        .img-box .img-title {{ font-size: 24px; font-weight: 800; color: #F8FAFC; margin-bottom: 8px; }}
        .img-box .img-sub {{ font-family: 'Fira Code', monospace; font-size: 18px; color: #64748B; }}
    </style>
</head>
<body>
<div class="canvas">
    
    <div class="header">
        QUANTUM<span>HARVEST</span> : HYBRID VAC-QFS INTELLIGENCE
        <div class="subtitle">[ I M A G E &nbsp;&nbsp; R E T R I E V A L &nbsp;&nbsp; ➔ &nbsp;&nbsp; Q U A N T U M &nbsp;&nbsp; S P A C E &nbsp;&nbsp; ➔ &nbsp;&nbsp; Y I E L D &nbsp;&nbsp; V E R T E X ]</div>
    </div>

    <!-- Groups -->
    <svg>
        <defs>
            <marker id="arrow-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#EAB308"/></marker>
            <marker id="arrow-p" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#A855F7"/></marker>
            <marker id="arrow-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#38BDF8"/></marker>
        </defs>

        <!-- Phase I: CV (x=450 to 950) -->
        <rect class="g-rect" x="430" y="160" width="540" height="960" />
        <rect class="g-bg" x="455" y="145" width="490" height="30" />
        <text class="g-title" x="475" y="169" fill="#3B82F6">PHASE I: SPATIAL & SPECTRAL</text>

        <!-- Phase II: Quantum (x=1020 to 1820) -->
        <rect class="g-rect" x="1030" y="160" width="760" height="740" />
        <rect class="g-bg" x="1060" y="145" width="700" height="30" />
        <text class="g-title" x="1080" y="169" fill="#A855F7">PHASE II: VARIATIONAL QUANTUM SUBSET</text>

        <!-- Phase III: Synthesis (x=1880 to 2350) -->
        <rect class="g-rect" x="1850" y="160" width="500" height="960" />
        <rect class="g-bg" x="1875" y="145" width="450" height="30" />
        <text class="g-title" x="1895" y="169" fill="#F59E0B">PHASE III: PREDICTIVE SYNTHESIS</text>

        <!-- Wiring (Orthogonal) -->
        
        <!-- Images to Phase 1 -->
        <path class="wire-b" d="M 330 350 L 380 350 L 380 250 L 460 250" /> <!-- Pre -> Morph -->
        <path class="wire-b" d="M 330 350 L 380 350 L 380 810 L 460 810" /> <!-- Pre -> GLCM -->
        
        <path class="wire-b" d="M 330 850 L 380 850 L 380 450 L 460 450" /> <!-- Post -> HSV -->
        <path class="wire-b" d="M 330 850 L 380 850 L 380 990 L 460 990" /> <!-- Post -> Spectral -->

        <!-- Intra Phase 1 -->
        <path class="wire-b" d="M 700 310 L 700 350" /> <!-- Morph -> HSV -->
        <path class="wire-b" d="M 700 510 L 700 550" /> <!-- HSV -> Distance -->
        
        <!-- CV -> Yield (Phase 3) -->
        <path class="wire-g" d="M 700 680 L 700 720 L 980 720 L 980 1060 L 1800 1060 L 1800 850 L 1890 850" />
        
        <!-- Features -> Phase 2 (MI Filter) -->
        <path class="wire-p" d="M 940 810 L 990 810 L 990 280 L 1050 280" />
        <path class="wire-p" d="M 940 990 L 1005 990 L 1005 280 L 1050 280" />
        
        <!-- MI Filter -> Encoding -->
        <path class="wire-p" d="M 1410 340 L 1410 400 L 1220 400 L 1220 440" />
        <!-- Encoding -> Ansatz -->
        <path class="wire-p" d="M 1390 500 L 1430 500" />
        
        <!-- Optimization Loop -->
        <path class="wire-g" d="M 1600 560 L 1600 610 L 1410 610 L 1410 630" /> <!-- Ansatz down to Opt -->
        <path class="wire-g" d="M 1240 690 L 1200 690 L 1200 480 L 1220 480" /> <!-- Opt back to Encoding/Ansatz (approx) -->
        <path class="wire-p" d="M 1600 560 L 1600 750 L 1570 750" /> <!-- Ansatz out to Subset Eval -->

        <!-- Quantum -> SVM -->
        <path class="wire-g" d="M 1410 810 L 1410 860 L 1800 860 L 1800 330 L 1890 330" />

        <!-- SVM -> Defoliation -->
        <path class="wire-g" d="M 2100 400 L 2100 470" />

    </svg>

    <!-- UI Nodes (HTML Overlay) -->

    <!-- INPUTS -->
    <div class="node img-box" style="left:40px; top:180px;">
        <img src="data:image/jpeg;base64,{b64_pre}">
        <div class="img-title">PRE-DEFOLIATION</div>
        <div class="img-sub">Highly Camouflaged Canopy</div>
    </div>
    <div class="node img-box" style="left:40px; top:680px;">
        <img src="data:image/jpeg;base64,{b64_post}">
        <div class="img-title">POST-DEFOLIATION</div>
        <div class="img-sub">Exposed Harvest Topology</div>
    </div>

    <!-- PHASE I: CV -->
    <div class="node n-cv" style="left:470px; top:190px; width:460px; height:120px;">
        <div class="title">Morphological Top-Hat</div>
        <div class="math"><i>T<sub>w</sub>(I) = I - (I ∘ b)</i></div>
    </div>
    <div class="node n-cv" style="left:470px; top:390px; width:460px; height:120px;">
        <div class="title">HSV Masking & CLAHE</div>
        <div class="math"><i>M = {{ p | S<sub>p</sub> &lt; &tau;<sub>s</sub> &and; V<sub>p</sub> &gt; &tau;<sub>v</sub> }}</i></div>
    </div>
    <div class="node n-cv" style="left:470px; top:560px; width:460px; height:120px;">
        <div class="title">Contour Distance Transform</div>
        <div class="math"><i>D(p) = min<sub>q &isin; B</sub> d(p, q)</i></div>
    </div>

    <div class="node n-feat" style="left:470px; top:750px; width:460px; height:120px;">
        <div class="title">GLCM Textural Matrices</div>
        <div class="math"><i>P(i, j | d, &theta;)</i></div>
    </div>
    <div class="node n-feat" style="left:470px; top:930px; width:460px; height:120px;">
        <div class="title">Spectral Indices</div>
        <div class="math"><i>ExG = 2G - R - B</i></div>
    </div>

    <!-- PHASE II: QUANTUM -->
    <div class="node n-quant" style="left:1060px; top:220px; width:700px; height:120px;">
        <div class="title">Mutual Information Filter Pruning</div>
        <div class="math"><i>I(X; Y) = &sum; p(x,y) log [p(x,y) / p(x)p(y)]</i></div>
    </div>

    <div class="node n-quant" style="left:1060px; top:440px; width:330px; height:140px;">
        <div class="title">Data Encoding</div>
        <div class="math"><i>|&Phi;(x)&lang; = U<sub>&Phi;</sub> H<sup>&otimes;n</sup> |0&lang;</i></div>
        <div class="code">ZZFeatureMap</div>
    </div>
    <div class="node n-quant" style="left:1440px; top:440px; width:320px; height:140px;">
        <div class="title">Parameterized Ansatz</div>
        <div class="math"><i>U<sub>A</sub>(&theta;) |&Phi;(x)&lang;</i></div>
        <div class="code">RealAmplitudes</div>
    </div>

    <div class="node n-quant" style="left:1250px; top:630px; width:320px; height:140px; border-color:#EAB308;">
        <div class="title" style="color:#FDE047;">Classical Optimizer</div>
        <div class="math"><i>min<sub>&theta;</sub> L(&theta;)</i></div>
        <div class="code" style="border-left-color:#EAB308; color:#EAB308;">COBYLA</div>
    </div>

    <div class="node n-quant" style="left:1060px; top:690px; width:510px; height:120px; z-index:5;">
        <div class="title">Combinatorial Subset Evaluator</div>
        <div class="math"><i>max<sub>S &sub; F, |S|=4</sub> A<sub>VQC</sub>(S)</i></div>
    </div>

    <!-- PHASE III: SYNTHESIS -->
    <div class="node n-out" style="left:1900px; top:260px; width:420px; height:140px;">
        <div class="title">Quantum Subset RBF SVM</div>
        <div class="math"><i>K(x, x') = exp(-&gamma; ||x-x'||<sup>2</sup>)</i></div>
        <div class="code" style="border-left-color:#F59E0B; color:#F59E0B">C(6,4) Optimal Manifold</div>
    </div>

    <div class="node" style="left:1900px; top:470px; width:420px; height:160px; background:#F59E0B; border:3px solid #FEF3C7; color:#111827;">
        <div class="title" style="color:#000;">Classification Vertex</div>
        <div class="math" style="color:#000;"><i>y&#770; &isin; {{Pre, Post}}</i></div>
    </div>

    <div class="node n-out" style="left:1900px; top:780px; width:420px; height:140px;">
        <div class="title" style="color:#38BDF8;">Spatial Yield Density Map</div>
        <div class="math"><i>&rho;(x,y) = &sum; &delta;(x-x<sub>i</sub>, y-y<sub>i</sub>)</i></div>
    </div>
    
    <div style="position:absolute; bottom:20px; right:40px; font-family:'Fira Code', monospace; font-size:18px; color:#475569;">
        Fig 1: Proposed QuantumHarvest Hybrid VAC-QFS Analytical Pipeline
    </div>
</div>
</body>
</html>
"""

with open("/Volumes/T9/CottonDefoliationApp/difflogic_architecture.html", "w") as f:
    f.write(html)
print("DiffLogic style diagram generated.")
