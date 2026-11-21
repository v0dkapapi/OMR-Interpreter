import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def build_template():
    page_size = {
        "width": config.CANONICAL_WIDTH,
        "height": config.CANONICAL_HEIGHT
    }
    
    alignment_anchors = {
        "top_left": {"x": 60, "y": 150, "w": 55, "h": 55},
        "top_right": {"x": 1570, "y": 110, "w": 50, "h": 30},
        "bottom_right": {"x": 1570, "y": 2160, "w": 50, "h": 30},
        "student_id_anchor": {"x": 1235, "y": 195, "w": 40, "h": 45}
    }
    
    handwritten_rois = {
        "name": {"x": 650, "y": 245, "w": 500, "h": 65},
        "module_code": {"x": 650, "y": 340, "w": 500, "h": 75},
        "seat_number": {"x": 650, "y": 445, "w": 220, "h": 65},
        "date": {"x": 940, "y": 445, "w": 250, "h": 65},
        "student_number_written": {"x": 1250, "y": 195, "w": 310, "h": 45}
    }
    
    # 8-column Student Number grid
    student_col_centers = [1258, 1298, 1336, 1376, 1415, 1454, 1492, 1531]
    student_row_centers = [238, 271, 303, 334, 365, 398, 430, 463, 495, 529] # 0 to 9
    
    student_number_grid = {}
    for col_idx, cx in enumerate(student_col_centers, start=1):
        col_key = str(col_idx)
        student_number_grid[col_key] = {}
        for digit, cy in enumerate(student_row_centers):
            student_number_grid[col_key][str(digit)] = {
                "x": int(cx - 14),
                "y": int(cy - 12),
                "w": 28,
                "h": 24
            }
            
    # 30 question row centers Y (6 blocks of 5 questions)
    row_centers_y = [
        # Block 1
        711, 751, 790, 830, 870,
        # Block 2
        976, 1014, 1052, 1090, 1127,
        # Block 3
        1231, 1270, 1310, 1348, 1386,
        # Block 4
        1494, 1532, 1572, 1611, 1650,
        # Block 5
        1760, 1800, 1840, 1878, 1918,
        # Block 6
        2027, 2065, 2105, 2144, 2185
    ]
    
    # Options A-E centers X for each of the 4 columns
    col_option_xs = [
        {"A": 215, "B": 248, "C": 288, "D": 327, "E": 373},   # Col 1: Q1-30
        {"A": 607, "B": 637, "C": 675, "D": 712, "E": 752},   # Col 2: Q31-60
        {"A": 988, "B": 1020, "C": 1059, "D": 1099, "E": 1138}, # Col 3: Q61-90
        {"A": 1378, "B": 1411, "C": 1450, "D": 1489, "E": 1529} # Col 4: Q91-120
    ]
    
    bubble_rois = {}
    for col_idx in range(4):
        opt_xs = col_option_xs[col_idx]
        for row_idx in range(30):
            q_num = col_idx * 30 + row_idx + 1
            cy = row_centers_y[row_idx]
            q_key = str(q_num)
            bubble_rois[q_key] = {}
            for opt in ["A", "B", "C", "D", "E"]:
                cx = opt_xs[opt]
                bubble_rois[q_key][opt] = {
                    "x": int(cx - 14),
                    "y": int(cy - 12),
                    "w": 28,
                    "h": 24
                }
                
    scoring = {
        "marks_per_correct": 1.0,
        "negative_marks": 0.0,
        "multi_mark_policy": "zero", # "zero" or "negative"
        "unanswered_marks": 0.0,
        "total_questions": 120,
        "options_per_question": 5
    }
    
    template = {
        "name": "UCD_S3231_Standard",
        "description": "Canonical template for S3231 120-question multiple-choice answer sheet",
        "page_size": page_size,
        "alignment_anchors": alignment_anchors,
        "handwritten_rois": handwritten_rois,
        "student_number_grid": student_number_grid,
        "bubble_rois": bubble_rois,
        "scoring": scoring
    }
    
    with open(config.TEMPLATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2)
        
    print(f"Generated template.json with {len(bubble_rois)} questions and {len(student_number_grid)} student ID columns.")
    print(f"Saved to: {config.TEMPLATE_PATH}")

if __name__ == "__main__":
    build_template()
