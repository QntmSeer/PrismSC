# scripts/log_utils.py
import sys
import os
import warnings

# Suppress all library warnings to keep logs clean and readable
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

def gradient_text(text, style="prism"):
    """Apply truecolor 24-bit RGB gradient to a string using styles."""
    if style == "worker":
        # cyan (0, 242, 254) to blue (79, 172, 254)
        start_rgb, end_rgb = (0, 242, 254), (79, 172, 254)
    else:
        # violet (139, 92, 246) to magenta (217, 70, 239)
        start_rgb, end_rgb = (139, 92, 246), (217, 70, 239)
        
    n = len(text)
    if n <= 1:
        return f"\033[38;2;{start_rgb[0]};{start_rgb[1]};{start_rgb[2]}m{text}\033[0m"
    result = []
    for i, char in enumerate(text):
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (n - 1))
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (n - 1))
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (n - 1))
        result.append(f"\033[38;2;{r};{g};{b}m{char}")
    return "".join(result) + "\033[0m"

def progress_gradient_text(text, fraction):
    """Interpolate gradient from cyan-blue (worker) at 0.0 to violet-magenta (prismsc) at 1.0."""
    start_rgb = (
        int(0 + (139 - 0) * fraction),
        int(242 + (92 - 242) * fraction),
        int(254 + (246 - 254) * fraction)
    )
    end_rgb = (
        int(79 + (217 - 79) * fraction),
        int(172 + (70 - 172) * fraction),
        int(254 + (239 - 254) * fraction)
    )
    
    n = len(text)
    if n <= 1:
        return f"\033[38;2;{start_rgb[0]};{start_rgb[1]};{start_rgb[2]}m{text}\033[0m"
    result = []
    for i, char in enumerate(text):
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * i / (n - 1))
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * i / (n - 1))
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * i / (n - 1))
        result.append(f"\033[38;2;{r};{g};{b}m{char}")
    return "".join(result) + "\033[0m"

def log_info(module, message, style="prism"):
    prefix = gradient_text(f"[{module}]", style=style)
    print(f"{prefix} {message}")
    sys.stdout.flush()

def log_success(module, message, style="prism"):
    prefix = gradient_text(f"[{module}]", style=style)
    print(f"{prefix} \033[1;32m[SUCCESS]\033[0m {message}")
    sys.stdout.flush()

def log_warn(module, message, style="prism"):
    prefix = gradient_text(f"[{module}]", style=style)
    print(f"{prefix} \033[1;33m[WARN]\033[0m {message}")
    sys.stdout.flush()

def log_error(module, message, style="prism"):
    prefix = gradient_text(f"[{module}]", style=style)
    print(f"{prefix} \033[1;31m[ERROR]\033[0m {message}")
    sys.stdout.flush()

def print_logo():
    logo = r"""
                  /\
                 /  \
                /    \
               /  /\  \
 =============/==/==\==\=============
             /  /    \  \  . · :  [RNA-seq]
            /  /______\  \  : .   [ATAC-seq]
           /_____________\   · .  [WNN-Graph]

   ____       _               ____   ____ 
  / __ \_____(_)________ ___ / ___/  / ___/ 
 / /_/ / ___/ / ___/ __ `__ \\___ \ / /     
/ ____/ /  / (__  ) / / / / /___/ // /___   
/_/   /_/  /_/____/_/ /_/ /_//____/ \____/  
"""
    for line in logo.strip("\n").split("\n"):
        print(gradient_text(line, style="prism"))
    print()
    sys.stdout.flush()


