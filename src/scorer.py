import pandas as pd

def load_answer_key(excel_path):
    """
    Load wide-format Excel answer key and convert to long format.
    Supports multiple correct answers like "16 - a,b,c,d".
    """
    df = pd.read_excel(excel_path)
    
    # Convert wide to long format - use column names exactly as they are
    df_long = df.melt(var_name='subject', value_name='answer')

    # Remove empty answers
    df_long = df_long.dropna(subset=['answer'])

    # Extract question number (handles both formats: "1 - a" and "1. a")
    df_long['question'] = df_long['answer'].str.extract(r'(\d+)').astype(int)

    # Extract answer letters (handle multiple letters and both separator formats)
    df_long['answer'] = df_long['answer'].str.extract(r'(?:[-.])\s*([A-Da-d,]+)')[0].str.upper()

    # Reorder columns
    df_long = df_long[['subject', 'question', 'answer']]

    # Strip subject names
    df_long['subject'] = df_long['subject'].str.strip()

    return df_long


def map_bubbles_to_answers(detected_bubbles, subjects_order, questions_per_subject=100):
    """
    Map the detected bubble states to answer letters for each subject and question.
    """
    option_map = {0:'A', 1:'B', 2:'C', 3:'D'}
    mapped = {}
    idx = 0

    for s in subjects_order:
        # Use subject names exactly as they are from Excel
        mapped[s] = {}
        for q in range(1, questions_per_subject+1):
            group = detected_bubbles[idx:idx+4]
            idx += 4
            
            filled_count = group.count('filled')
            if filled_count == 1:  # Exactly one bubble filled
                mapped[s][q] = option_map[group.index('filled')]
            elif filled_count > 1:  # Multiple bubbles filled
                mapped[s][q] = 'AMBIGUOUS'
            else:  # No bubbles filled
                mapped[s][q] = 'BLANK'

    return mapped


def score(mapped_answers, answer_key_df, marks_per_question=1, debug=False):
    """
    Score the mapped answers against the answer key.
    Handles multiple correct answers.
    """
    per_subject = {}
    total = 0

    for subject, qdict in mapped_answers.items():
        per_subject[subject] = 0
        for qnum, detected in qdict.items():
            # Get correct answer(s)
            correct_ans_arr = answer_key_df.loc[
                (answer_key_df.subject == subject) & (answer_key_df.question == qnum), 'answer'
            ].values

            if len(correct_ans_arr) == 0:
                if debug:
                    print(f"{subject} Q{qnum}: No correct answer found in Excel")
                continue

            correct_ans = correct_ans_arr[0]  # e.g., "A,B,C,D"
            correct_list = [a.strip().upper() for a in correct_ans.split(',')]

            if detected.upper() in correct_list:
                per_subject[subject] += marks_per_question
                total += marks_per_question
                if debug:
                    print(f"{subject} Q{qnum}: detected={detected} ✅ correct={correct_ans}")
            else:
                if debug:
                    print(f"{subject} Q{qnum}: detected={detected} ❌ correct={correct_ans}")

    return per_subject, total
