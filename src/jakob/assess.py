import pandas as pd
from sklearn.metrics import mean_absolute_error
import sys

FIRST_UNKNOWN_SUB_ID = 10

def check_ids(unmasked_data: pd.DataFrame, predicted_data: pd.DataFrame):
    unmasked_id_sub_id = unmasked_data[['id', 'sub_id']]
    predicted_id_sub_id = predicted_data[['id', 'sub_id']]
    if not unmasked_id_sub_id.equals(predicted_id_sub_id):
        print("Error: The id's and sub_id's are not identical in the revealed/answer and suggested datasets")
        sys.exit(1)
    print("The id's and sub_id's are identical in the revealed/answer and suggested datasets")


def check_columns(unmasked_power_unknowns:pd.DataFrame, predicted_data:pd.DataFrame):
    required_columns = ['id', 'sub_id', 'power']
    if not all(column in unmasked_power_unknowns.columns for column in required_columns):
        print("Error: The revealed/answer dataset is missing the required columns")
        sys.exit(1)
    if not all(column in predicted_data.columns for column in required_columns):
        print("Error: The suggested dataset is missing the required columns")
        sys.exit(1)
    print(f"The revealed/answer and suggested datasets have the required columns: {required_columns}")



if __name__ == "__main__":
    print("Comparing the two datasets, revealed(answer) and predicted(suggested)")
    if len(sys.argv) != 3:
        print("Error: Missing arguments\nUsage: python assess.py <answer.parquet> <predicted.parquet>")
        sys.exit(1)

    unmasked_data = pd.read_parquet(sys.argv[1])
    predicted_data = pd.read_parquet(sys.argv[2])
    unmasked_power_unknowns = unmasked_data[unmasked_data.sub_id >= FIRST_UNKNOWN_SUB_ID]
    predicted_data = predicted_data[predicted_data.sub_id >= FIRST_UNKNOWN_SUB_ID]

    print("Checking the datasets for identical id's and sub_id's and required columns")
    check_columns(unmasked_power_unknowns, predicted_data)
    check_ids(unmasked_power_unknowns, predicted_data)

    print("Now comparing the predicted power values with the unmasked power values")
    mae = mean_absolute_error(unmasked_power_unknowns['power'], predicted_data['power'])
    print("MAE: ", mae)



