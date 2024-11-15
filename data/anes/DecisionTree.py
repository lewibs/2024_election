from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

xpairs = [
    ["V202001", "SELECT LANGUAGE", True],
    ["V202008", "DID PEOPLE TELL YOU TO VOTE", True],
    ["V202009", "DID YOU TELL PEOPLE WHY THEY SHOULD VOTE", True],
    ["V202013", "DID YOU PARTICIPATE IN ANY ONLINE MEETING, RALLIE, SPEACHES, OR FUNDRAISERS", True],
    ["V202014", "DID YOU GO TO ANY IN ANY IN PERSON MEETING, RALLIE, SPEACHES, OR FUNDRAISERS", True],
    ["V202015", "DO YOU WEAR ANY CAMPAIN MERCH", True],
    ["V202016", "DID YOU DO ANY WORK FOR A CANDIDATE", True],
    ["V202017", "DID YOU DONATE TO A CANDIDATE", True],
    ["V202019", "DID YOU DONATE ANY MONEY TO A PARTY", True],
    ["V202022", "DO YOU DISCUSS POLITICS WITH FAMILY OR FRIENDS", True],
    ["V202023", "HOW OFTEN DO YOU DISCUSS POLOTICS IN A WEEK", True],
    ["V202024", "HAVE YOU GOTTEN INTO A POLITICAL ARGUMENT IN THE LAST 12 MONTHS", True],
    ["V202025", "IN THE LAST 12 MONTHS HAVE YOU GONE TO A PROTEXT, MARCH, RALLY, OR DEMINSTRATION", True],
    ["V202026", "IN THE LAST 12 MONTHS HAVE YOU SIGNED A PETITION", True],
    ["V202027", "IN THE LAST 12 MONTHS HAVE YOU GIVEN MONEY TO A RELIGIOUS ORGINIZATION", True],
    ["V202028", "IN THE LAST 12 MONTHS HAVE YOU GIVEN MONEY TO ANY OTHER ORGINIZATION", True],
    ["V202029", "HAVE YOU POSTED ONLINE ABOUT POLOTICS IN THE LAST 12 MONTHS", True],
    ["V202030", "IN THE LAST 12 MONTHS HAVE YOU CONTACTED A MEMBER OF THE US SENATE OR HOUSE OF REP", True],
    ["V202034", "IN THE LAST 12 MONTHS HAVE YOU CONTACTED AN ELECTED OFFICIAL IN FEDERAL GOVT", True],
    ["V202036", "IN THE LAST 12 MONTHS HAVE YOU CONTECTED A NON-ELECTED OFFICAL IN FEDERAL GOVT", True],
    ["V202040", "IN THE LAST 12 MONTHS HAVE YOU CONTACTED A NON-ELECTED LOCAL OFFICAL", True],
    ["V202038", "IN THE LAST 12 MONTHS HAVE YOU CONTACTED AN ELECTED LOCAL/STATE OFFICAL", True],
    ["V202042", "HOW OFTEN BOUGHT OR BOYCOTTED PRODUCT/SERVICE FOR SOCIAL/POLITICAL REASONS", True],
    ["V202051", "REGISTED TO VOTE", True],
    ["V202054a", "WHAT STATE ARE YOU CURRENTLY IN", True],
    ["V202054b", "WHAT STATE ARE YOU VOTING IN", True],
    ["V202061", "HOW LONG HAVE YOU BEEN REGISTERED TO VOTE THERE", True],
    ["V202074", "DO YOU STRONGLY LIKE WHO YOU VOTED FOR", True],
    ["V202075", "HOW LONG BEFORE THE ELECTION DID YOU MAKE THE DECISION TO VOTE", True],
    ["V202118", "HOW DO YOU USUALLY VOTE", True],
    ["V202119", "HOW DIFFICULT WAS IT FOR YOU TO VOTE", True],
    #THEMEMITERS
    ["V202159", "CHRISTIAN VALUES", False],
    ["V202160", "FEMINISTS", False],
    ["V202161", "LIBERALS", False],
    ["V202162", "UNIONS", False],
    ["V202163", "BIG BUSINESS", False],
    ["V202164", "CONSERVATIVES", False],
    ["V202165", "THE US SUPREME COURT", False],
    ["V202166", "GAY MEN AND LESBIANS", False],
    # IT WOULD BE INTERESTING TO LOOK AT HOW THIS HAS CHANGED OVER THE YEARS
    ["V202167", "CONGRESS", False],
    ["V202168", "MUSLIMS", False],
    ["V202169", "CHRISTIANS", False],
    ["V202170", "JEWS", False],
    ["V202171", "POLICE", False],
    ["V202172", "TRANSGENDERS", False],
    ["V202173", "SCIENTISTS", False],
    ["V202174", "BLM", False],
    ["V202175", "HOURNALISTS", False],
    ["V202176", "NATO", False],
    ["V202177", "UN", False],
    ["V202178", "NRA", False],
    ["V202179", "SOCIALISTS", False],
    ["V202180", "CAPITALISTS", False],
    ["V202181", "FBI", False],
    ["V202182", "ICE", False],
    ["V202183", "#METOO", False],
    ["V202184", "RURAL AMERICANS", False],
    ["V202185", "PLANNED PARENTHOOD", False],
    ["V202186", "WHO", False],
    ["V202187", "CDC", False],
    #MOST IMPORTATNT PROBLEMS
    ["V202205y1", "MOST IMPORTANT PROBLEMS FACING THE COUNTRY", True],
    #OTHERS
    ["V202215", "HOW WELL DO YOU UNDERSTAND IMPORTANT ISSUES", True],
    ["V202220", "HOW IMPORTANT IS IT THAT MORE HISPANICS GET ELECTED", True],
    ["V202221", "HOW IMPORTANT IS IT THAT MORE BLACKS GET ELECTED", True],
    ["V202222", "HOW IMPORTANT IS IT THAT MORE ASIANS GET ELECTED", True],
    ["V202223", "HOW IMPORTANT IS IT THAT MORE LGBT GET ELECTED", True],
    ["V202224", "HOW IMPORTANT IS IT THAT MORE WOMEN GET ELECTED", True],
    ["V202225", "SHOULD THERE BE LIMITS ON CAMPAIN SPENDING", True],
    ["V202229", "LIMITS ON IMPORTS", True],
    ["V202232", "WHAT SHOULD IMIGRATION LEVELS BE", True],
    ["V202233", "HOW LIKELY IMMIGRATION WILL TAKE AWAY JOBS", True],
    ["V202234", "SHOULD WE ACCEPT REFUGEES", True],
    ["V202237", "WHAT IS THE EFFECT OF ILLEGAL IMMIGRATION ON CRIME RATE", True],
    ["V202243", "RETURN IMMIGRANTS TO THEIR COUNTRY", True],
    ["V202249", "DEI", True],
    ["V202253", "SHOULD WE HAVE LESS GOVERNMENT", True],
    ["V202257", "REDUCE INCOME INEQUALITY", True],
    ["V202260", "SOCIETY SHOULD ENFORCE EQUALITY", True],
    #SO MANY QUESTIONS.... PAUSED HERE I CAN ADD MORE LATER
]

ypairs = [
    ["V202064", "WHAT IS YOUR PARTY OF REGISTRATION", True],
    ["V202073", "WHO DID YOU VOTE FOR PRESIDENT", True],
]

def make_map(pairs, data):
    m = {}
    for [key, value, one_hot] in pairs:
        print(f"Processing key: {key}")
        
        # Initialize the dictionary entry for the key
        m[key] = [key, value, {}]
        
        if one_hot:
            # Get the unique values for the key/column in the dataset
            unique_values = data[key].unique()
            unique_values = unique_values.tolist()
            m[key][2]["length"] = len(unique_values)
            for i, value in enumerate(unique_values):
                m[key][2][value] = i
        
    return m


# Load data from CSV
data = pd.read_csv('./anes_timeseries_2020_csv_20220210.csv')

xkeys = make_map(xpairs, data)
ykeys = make_map(ypairs, data) 
# Print the headers
print(ykeys)


# # Assume the features are in columns 'Feature1' and 'Feature2', and 'Label' is the target column
# X = data[['Feature1', 'Feature2']].values  # Features
# y = data['Label'].values  # Target labels

# # Sample dataset
# X = np.array([[2.5, 'A'], [7.5, 'B'], [3.0, 'A'], [9.5, 'B'], [4.5, 'A'], [8.5, 'B'], [5.5, 'A'], [6.5, 'B']])
# y = np.array([0, 1, 0, 1, 0, 1, 0, 1])

# # Define preprocessor for different types of features
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', StandardScaler(), [0]),             # Standardize numerical features
#         ('cat', OneHotEncoder(), [1])               # One-hot encode categorical features
#     ])

# # Create a pipeline with preprocessing and decision tree
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('classifier', DecisionTreeClassifier(max_depth=3, random_state=42))
# ])

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the model
# pipeline.fit(X_train, y_train)

# # Make predictions
# predictions = pipeline.predict(X_test)
# print("Predictions:", predictions)
