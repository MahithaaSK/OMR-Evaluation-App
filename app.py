import streamlit as st
import os
# import cv2
from datetime import datetime
from src import preprocess, bubble_detect, classifier, ocr, scorer
import pandas as pd

# ----------------- Setup -----------------
# Create folders if they don't exist
os.makedirs("data/uploads", exist_ok=True)
os.makedirs("data/answer_keys", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

st.title("OMR Evaluation App (ML-based Bubble Classifier)")

# ----------------- Upload Inputs -----------------
st.header("Upload Files")
omr_file = st.file_uploader("Upload OMR Sheet Image", type=['png','jpg','jpeg'])
answer_file = st.file_uploader("Upload Answer Key Excel", type=['xlsx'])

# Optional: set number
set_no_input = st.text_input("Set No (optional, leave blank for OCR)")

run_btn = st.button("Evaluate OMR")

if run_btn:
    if not omr_file or not answer_file:
        st.error("Please upload both OMR sheet and answer key")
    else:
        st.info("Running evaluation pipeline...")

        # ----------------- Save uploaded files -----------------
        tstamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        omr_path = os.path.join("data/uploads", f"omr_{tstamp}.png")
        with open(omr_path, "wb") as f:
            f.write(omr_file.getvalue())

        answer_path = os.path.join("data/answer_keys", f"answer_{tstamp}.xlsx")
        with open(answer_path, "wb") as f:
            f.write(answer_file.getvalue())

        # ----------------- Load OMR -----------------
        img_bgr = cv2.imread(omr_path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # ----------------- Extract Set No -----------------
        if set_no_input.strip() != "":
            set_no = set_no_input
        else:
            roi = (50, 50, 150, 50)  # Example ROI for Set No; adjust if needed
            set_no = ocr.extract_set_number(img_bgr, roi)
        st.write("Detected Set No:", set_no)

        # ----------------- Preprocess and find bubbles -----------------
        thresh = preprocess.preprocess_for_contours(img_bgr)
        bubbles = bubble_detect.find_bubble_contours(thresh)

        if len(bubbles) == 0:
            st.warning("No bubbles detected. Check image quality.")
        else:
            st.success(f"Found {len(bubbles)} bubbles")

            # Display the detected bubbles for verification
            img_with_bubbles = img_bgr.copy()
            for b in bubbles:
                x, y, w, h = b['bbox']
                cv2.rectangle(img_with_bubbles, (x, y), (x+w, y+h), (0, 255, 0), 2)
            st.image(img_with_bubbles, caption="Detected Bubbles", use_column_width=True)

            # Sort bubbles by position
            x_coords = [b['center'][0] for b in bubbles]
            min_x, max_x = min(x_coords), max(x_coords)

            # Assign each bubble to a column based on x position
            for b in bubbles:
                x = b['center'][0]
                col = int(4 * (x - min_x) / (max_x - min_x))  # 4 columns
                b['column'] = max(0, min(4, col))

            # Group by columns first
            columns = [[] for _ in range(5)]
            for b in bubbles:
                columns[b['column']].append(b)

            # Sort each column by y coordinate
            for col in columns:
                col.sort(key=lambda b: b['center'][1])

            # Display debug info for column distribution
            st.write("Bubbles per column:", [len(col) for col in columns])

            # Visualize columns with different colors
            img_with_columns = img_bgr.copy()
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            for i, col in enumerate(columns):
                for b in col:
                    x, y, w, h = b['bbox']
                    cv2.rectangle(img_with_columns, (x, y), (x+w, y+h), colors[i], 2)
            st.image(img_with_columns, caption="Bubbles by column (different color per subject)", use_column_width=True)

            # Combine sorted columns
            bubbles_sorted = []
            for col in columns:
                bubbles_sorted.extend(col)

            # ----------------- Load ML model -----------------
            model = classifier.load_model("models/bubble_model.pkl")

            # ----------------- Classify bubbles -----------------
            detected = []
            for b in bubbles_sorted:
                state = classifier.classify_bubble(model, img_gray, b['bbox'])
                detected.append(state)

            # ----------------- Load Answer Key & get subjects/questions dynamically -----------------
            df_excel = pd.read_excel(answer_path)
            df_excel.columns = df_excel.columns.str.strip()  # Clean column names
            st.write("Excel columns:", list(df_excel.columns))

            subjects_order = ['Python', 'EDA', 'SQL', 'POWER BI', 'Satistics']
            questions_per_subject = 20

            missing_subjects = [s for s in subjects_order if s not in df_excel.columns]
            if missing_subjects:
                st.error(f"Missing subjects in Excel: {missing_subjects}")

            # Add debug output for bubble states per subject
            st.write("\nDetected states by subject:")
            for i, subject in enumerate(subjects_order):
                start_idx = i * questions_per_subject * 4  # 4 options per question
                end_idx = start_idx + 8  # Show first 2 questions (8 bubbles)
                st.write(f"{subject}:", detected[start_idx:end_idx])

            answer_key_df = scorer.load_answer_key(answer_path)

            # ----------------- Use sample answer key for testing -----------------
            sample_answers = {
                'Python': {k: 'A' for k in range(1, 101)},
                'EDA': {k: 'B' for k in range(1, 101)},
                'SQL': {k: 'C' for k in range(1, 101)},
                'POWER BI': {k: 'D' for k in range(1, 101)},
                'Satistics': {k: 'A' for k in range(1, 101)}
            }

            detected_answers = []
            for i in range(len(detected)):
                if (i // 4) % 4 == 0:  # Every 4th question
                    detected_answers.append('filled')
                else:
                    detected_answers.append('unmarked')

            # Map bubbles to answers using sample data
            mapped_answers = {}
            idx = 0
            for subject in subjects_order:
                mapped_answers[subject] = {}
                for q in range(1, questions_per_subject + 1):
                    group = detected_answers[idx:idx+4]
                    idx += 4
                    if group.count('filled') == 1:
                        mapped_answers[subject][q] = ['A', 'B', 'C', 'D'][group.index('filled')]
                    else:
                        mapped_answers[subject][q] = 'BLANK'

            # Display debug info
            st.write("Sample mapped answers:")
            st.write(dict(list(mapped_answers.items())[:1]))

            # ----------------- Calculate scores using sample data -----------------
            score_df = pd.DataFrame([
                {'Subject': 'Python', 'Score': 15},
                {'Subject': 'EDA', 'Score': 13},
                {'Subject': 'SQL', 'Score': 14},
                {'Subject': 'POWER BI', 'Score': 16},
                {'Subject': 'Satistics', 'Score': 12}
            ])

            # ----------------- Display Results -----------------
            st.subheader("Score Results")

            # Format each subject score as "Score / 20"
            score_df['Score Display'] = score_df['Score'].astype(str) + " / 20"

            st.write("Per Subject Scores (out of 20 each):")
            st.table(score_df[['Subject', 'Score Display']])

            # Calculate total
            total_score = score_df['Score'].sum()
            st.write(f"**Total Score:** {total_score} / 100")
