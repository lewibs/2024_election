import pandas as pd

# Load the CSV file
data = pd.read_csv("./anes_timeseries_2020_csv_20220210.csv", low_memory=False)

# Filter for rows where V201001 equals 2
responses_with_spanish = data[data['V201001'] == 2]
responses_for_trump_in_spanish = data[(data['V201001'] == 2) & (data['V202073'] == 2)]
responses_for_biden_in_spanish = data[(data['V201001'] == 2) & (data['V202073'] == 1)]

# Define a function to print the sum of responses for a given variable
def print_response_sums(variable_name):
    response_sums = data[variable_name].value_counts(dropna=False)
    print(f"Response sums for {variable_name}:")
    print(response_sums)
    print("\n")

# Calculate the percentage
responses_for_trump = (len(responses_for_trump_in_spanish) / len(responses_with_spanish)) * 100
responses_for_biden =  (len(responses_for_biden_in_spanish) / len(responses_with_spanish)) * 100
responses_with_spanish = (len(responses_with_spanish) / len(data)) * 100

print(f"Percentage of people who answered in spanish: {responses_with_spanish:.2f}%")
print(f"Percentage of people who voted for trump: {responses_for_trump:.2f}%")
print(f"Percentage of people who voted for biden: {responses_for_biden:.2f}%")
