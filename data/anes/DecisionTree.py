import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import export_text
import numpy as np
import pandas as pd

# Load data from CSV
data = pd.read_csv('./anes_timeseries_2020_csv_20220210.csv')

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
    ["V202159", "CHRISTIANS", False],
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
    ["V202175", "JOURNALISTS", False],
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
    # #OTHERS
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
    ["V202238", "EFFECT OF ILLEGAL IMMIGRATION ON CRIME RATE (STRENGTH)", True],
    ["V202243", "RETURN IMMIGRANTS TO THEIR COUNTRY", True],
    ["V202249", "DEI", True],
    ["V202253", "SHOULD WE HAVE LESS GOVERNMENT", True],
    ["V202257", "REDUCE INCOME INEQUALITY", True],
    ["V202260", "SOCIETY SHOULD ENFORCE EQUALITY", True],
    #SO MANY QUESTIONS.... PAUSED HERE I CAN ADD MORE LATER
]

ypairs = [
    # ["V202064", "WHAT IS YOUR PARTY OF REGISTRATION", True],
    ["V202073", "WHO DID YOU VOTE FOR PRESIDENT", True],
]

def make_map(pairs, data):
    m = {}
    for [key, value, one_hot] in pairs:
        
        # Initialize the dictionary entry for the key
        m[key] = [key, value, {}]
        
        if one_hot:
            # Get the unique values for the key/column in the dataset
            unique_values = data[key].unique()
            unique_values = unique_values.tolist()
            m[key][2]["length"] = len(unique_values)
            for i, value in enumerate(unique_values):
                m[key][2][value] = i
        else:
            m[key][2]["length"] = 0

    return m

def one_hot_encode(data, dataMap, exludeIfBadMap):
    encoded_data = []
    ignore_column = [False] * data.shape[0]
    index_map = {}
    index_end = 0

    for key in exludeIfBadMap:
        for idx, answer in enumerate(data[key]):
            if answer < 0:
                ignore_column[idx] = True
 
    for key in dataMap:
        col_data = []
        question = dataMap[key]
        length = question[2]["length"]

        if length:
            column = data[key]

            for idx, value in enumerate(column):
                encode = np.zeros(length)
                encode[dataMap[key][2][value]] = 1
                if ignore_column[idx] == False: 
                    col_data.append(encode)
        else:
            for idx, value in enumerate(data[key]):
                if ignore_column[idx] == False: 
                    col_data.append(np.array([value]))

        encoded_data.append(col_data)
        
        index_start = index_end 
        index_end = index_end + len(col_data[0])

        for i in range(index_start,index_end):
            index_map[i] = {
                "code":question[0],
                "question":question[1],
                "options":question[2],
                "range_start":index_start,
                "range_end":index_end,
                "range_position":i-index_start
            }
    
    encoded_data = list(zip(*encoded_data))
    encoded_data = [np.concatenate(row) for row in encoded_data]

    return encoded_data, index_map

def answer_to_one_hot(question, answer, questionMap):
    question_key = questionMap[question][2]
    one_hot = [0] * question_key["length"]
    one_hot[question_key[answer]] = 1
    return one_hot

def decode_prediction(answer, output, labelMap):
    encoding_map = labelMap[answer][2]
    decoding_map = {value: key for key, value in encoding_map.items() if key != "length"}
    return decoding_map[np.argmax(output == 1)]

def make_prediction(predictor, xMap, yMap, is_temperature, question_code, answer_code, question_answer):
    if is_temperature:
        one_hot = [[question_answer]]
    else:
        one_hot = [answer_to_one_hot(question_code, question_answer, xMap)]
    answer_one_hot = predictor.predict(one_hot)[0]
    answer_index = np.argmax(answer_one_hot == 1)
    answer_prediction = predictor.predict_proba(one_hot)[answer_index][0]
    answer = decode_prediction(answer_code, answer_one_hot, yMap)
    return answer, answer_prediction

def make_clasifier(forest_name, xpairs, ypairs, data, print_info=True):
    xMap = make_map(xpairs, data)
    yMap = make_map(ypairs, data)

    # Print the headers
    X, input_keys = one_hot_encode(data, xMap, yMap)
    y, _ = one_hot_encode(data, yMap, yMap)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=3)

    # Train the ensemble model
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Evaluate the model
    if print_info:
        print(forest_name)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

        importances = clf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]

        added_questions = {}
        questions = []

        for i in sorted_indices:
            if input_keys[i]["question"] not in added_questions: 
                added_questions[input_keys[i]["question"]] = input_keys[i]
                questions.append([importances[i], input_keys[i]])

        print("Ranked Question Importance:")
        [print(question[1]["question"], question[0]) for question in questions]
        print("")


    return clf, xMap, yMap
    

main_clasifier = make_clasifier("All Questions", xpairs, ypairs, data, True)

for pair in xpairs:
    answer_key = "V202073"

    response_map = {
        -9: "Refused",
        -8: "Don’t know",
        -7: "No post-election data, deleted due to incomplete interview",
        -6: "No post-election interview",
        -1: "Inapplicable",
        1: "Joe Biden",
        2: "Donald Trump",
        3: "Jo Jorgensen",
        4: "Howie Hawkins",
        5: "Other candidate {SPECIFY}",
        7: "Specified as Republican candidate",
        8: "Specified as Libertarian candidate",
        11: "Specified as don’t know",
        12: "Specified as refused"
    }

    single_question_clasifier, xMap, yMap = make_clasifier(pair[0] + " " + pair[1], [pair], ypairs, data, False)
    question_key = pair[0]
    question = pair[1]
    is_temperature = not pair[2]

    print(question_key, question)
    if is_temperature:
        print(f"{'Temp':<10}{'Prediction':<20}{'Confidence':<10}")
        for temp in range(100):
            prediction, confidence = make_prediction(single_question_clasifier, xMap, yMap, is_temperature, question_key, answer_key, temp)
            print(f"{temp:<10}{response_map[prediction]:<20}{max(confidence):<10.2f}")
    else:
        print(f"{'Answer':<10}{'Prediction':<20}{'Confidence':<10}")
        for persons_answer in xMap[question_key][2]:
            if persons_answer != "length" and int(persons_answer) >= 0:
                prediction, confidence = make_prediction(single_question_clasifier, xMap, yMap, is_temperature, question_key, answer_key, persons_answer)
                print(f"{persons_answer:<10}{response_map[prediction]:<20}{max(confidence):<10.2f}")

    print("")

