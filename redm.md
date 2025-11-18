
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. डेटा लोड करना
df = pd.read_csv('Book2.csv')

# 2. फीचर्स (X) और टारगेट (y) को अलग करना
X = df[['Hours_Studied', 'Attendance', 'Previous_Score']]  # independent variables
y = df['Final_Score']  # dependent variable

# 3. ट्रेन और टेस्ट सेट में बाँटना
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# यहाँ test_size=0.2 का मतलब है 20% डेटा टेस्ट के लिए, और 80% ट्रेन के लिए

# 4. मॉडल बनाना और ट्रेन करना
model = LinearRegression()  # मॉडल की instance
model.fit(X_train, y_train)  # ट्रेनिंग

# 5. टेस्ट पर prediction करना
y_pred = model.predict(X_test)

# 6. मॉडल को evaluate करना
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)

# 7. मॉडल के coefficients देखना (slope) और intercept
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. डेटा लोड करना
df = pd.read_csv('Book2.csv')

# 2. Class label बनाना (binary classification)
# मान लो अगर Final_Score >= 60 → Pass (1), वरना Fail (0)
df['Pass'] = (df['Final_Score'] >= 60).astype(int)

# 3. Features (X) और Target (y) चुनना
X = df[['Hours_Studied', 'Attendance', 'Previous_Score']]
y = df['Pass']

# 4. ट्रेन-टेस्ट विभाजन
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Logistic Regression मॉडल बनाना और ट्रेन करना
model = LogisticRegression(max_iter=1000)  # max_iter ज़रूरत से बढ़ाया, ताकि converge हो सके
model.fit(X_train, y_train)

# 6. Prediction करना
y_pred = model.predict(X_test)

# 7. मॉडल का मूल्यांकन करना
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 8. Probability देखना (optional)
y_pred_proba = model.predict_proba(X_test)  # यह दिखाएगा हर class की probability
print("Prediction Probabilities:\n", y_pred_proba)



Logistic Regression






Importing the libraries
1>
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
2>## Importing the datase
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

3
print(X_train)

4>print(y_train)
5>print(x_test)
5>print(y_test)

6>from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

7>print(X_train)
print(X_test)
Training the Logistic Regression model on the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
LogisticRegression(random_state=0)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.


LogisticRegression
?Documentation for LogisticRegressioniFitted
LogisticRegression(random_state=0)






Predicting a new result
classifier.fit(X_train, y_train)
print(classifier.predict(sc.transform([[30,87000]])))
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
## Making the Confusion Matrix

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


 3> Ordinary Least Squares

 1>import statsmodels.api as sm 
import pandas as pd          
import matplotlib.pyplot as plt 
import numpy as np

 2> data = pd.read_csv('Salary_Data.csv')


3>
x = data['YearsExperience'].tolist()  
y = data['Salary'].tolist()


4>x = sm.add_constant(x)

5>
result = sm.OLS(y, x).fit()

print(result.summary())


6plt.scatter(data['YearsExperience'], data['Salary'], color='blue', label='Data Points')

x_range = np.linspace(data['YearsExperience'].min(), data['Salary'].max(), 100)
y_pred = result.params[0] + result.params[1] * x_range 

plt.plot(x_range, y_pred, color='red', label='Regression Line')
plt.xlabel('Independent Variable (X)')
plt.ylabel('Dependent Variable (Y)')
plt.title('OLS Regression Fit')
plt.legend()
plt.show()





# -------------------------------
#  Linear Regression For ANY CSV Dataset
#  Beginner Friendly Code
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------
#  STEP 1: Load Your DATASET
# ---------------------------------------------
# 👉 Just change the CSV file name here
data = pd.read_csv("your_dataset.csv")

print("Dataset Loaded Successfully!")
print(data.head())


# ---------------------------------------------
#  STEP 2: Choose Features & Target Column
# ---------------------------------------------
# Example:
# X = data[["Size"]]        # input column (independent variable)
# y = data["Price"]         # output column (target variable)

# 👉 Change this according to your dataset
X = data.iloc[:, :-1]      # all columns except last = input
y = data.iloc[:, -1]       # last column = target


# ---------------------------------------------
#  STEP 3: Split Dataset for Training & Testing
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------
#  STEP 4: Train the Linear Regression Model
# ---------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nTraining Complete!")
print("Coefficient (m):", model.coef_)
print("Intercept (c):", model.intercept_)


# ---------------------------------------------
#  STEP 5: Make Predictions
# ---------------------------------------------
y_pred = model.predict(X_test)

print("\nPredictions on Test Data:")
print(y_pred)


# ---------------------------------------------
#  STEP 6: Model Accuracy
# ---------------------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)


# ---------------------------------------------
#  STEP 7: Plot (Only if dataset has 1 feature)
# ---------------------------------------------
if X.shape[1] == 1:
    plt.scatter(X_test, y_test, label="Actual Data")
    plt.plot(X_test, y_pred, label="Best Fit Line")
    plt.xlabel("Input Feature")
    plt.ylabel("Target Value")
    plt.title("Linear Regression")
    plt.legend()
    plt.show()
else:
    print("\nPlot not shown (dataset has multiple columns).")




Naive Bayes






Importing the librar
1>import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


3>from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


4>print(y_train)


5>print(X_test)

6>print(y_test)


7>from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


8>print(X_train)


print(X_test)









# ----------------------------------------------------
#  SUPPORT VECTOR MACHINE (SVM) FOR ANY CSV DATASET
#  Beginner Friendly - Copy Paste Ready
# ----------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC      # SVM Classifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------
# STEP 1: Load CSV Dataset
# ----------------------------------------------------
# 👉 बस अपना dataset नाम डालो
data = pd.read_csv("your_dataset.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# ----------------------------------------------------
# STEP 2: Select Features (X) & Target (y)
# ----------------------------------------------------
# Example:
# X = data[["Age", "Salary"]]   # input
# y = data["Purchased"]         # target

# 👉 नीचे automatic code है (last column = target)
X = data.iloc[:, :-1]   # all columns except last = inputs
y = data.iloc[:, -1]    # last column = target

# If your dataset has categorical values then encode them
from sklearn.preprocessing import LabelEncoder
if y.dtype == "object":
    le = LabelEncoder()
    y = le.fit_transform(y)

# ----------------------------------------------------
# STEP 3: Split Train/Test
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------
# STEP 4: Create SVM Model
# ----------------------------------------------------
model = SVC(kernel="linear")  
# kernels → "linear", "rbf", "poly", "sigmoid"

model.fit(X_train, y_train)

print("\nTraining Completed!")

# ----------------------------------------------------
# STEP 5: Prediction
# ----------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------
# STEP 6: Accuracy & Report
# ----------------------------------------------------
print("\nACCURACY:", accuracy_score(y_test, y_pred))

print("\nCONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))

# ----------------------------------------------------
# STEP 7 (Optional): Plot (Only when 2 features)
# ----------------------------------------------------
if X.shape[1] == 2:
    print("\nPlotting decision boundary...")

    # convert to numpy
    X_np = X_test.values

    # define boundary
    x_min, x_max = X_np[:, 0].min() - 1, X_np[:, 0].max() + 1
    y_min, y_max = X_np[:, 1].min() - 1, X_np[:, 1].max() + 1
    
    # mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # predictions for each point
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y_test)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("SVM Decision Boundary")
    plt.show()
else:
    print("Plot skipped (features > 2).")




# ------------------------------------------------------------
#   DECISION TREE CLASSIFICATION For ANY CSV Dataset
#   Beginner-Friendly | Copy-Paste Ready Code
# ------------------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# STEP 1: Load Your Dataset
# ------------------------------------------------------------
# 👉 बस यहाँ अपने dataset की file का नाम बदल देना
data = pd.read_csv("your_dataset.csv")

print("Dataset Loaded Successfully!")
print(data.head())

# ------------------------------------------------------------
# STEP 2: Select Input (X) & Output (y)
# ------------------------------------------------------------
# Example:
# X = data[["Age", "Salary"]]
# y = data["Purchased"]

# 👉 Below code makes last column as target
X = data.iloc[:, :-1]     # all columns except last
y = data.iloc[:, -1]      # last column = target

# If target contains text labels → Convert to numbers
from sklearn.preprocessing import LabelEncoder
if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

# ------------------------------------------------------------
# STEP 3: Split Train/Test
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------
# STEP 4: Create Decision Tree Model
# ------------------------------------------------------------
model = DecisionTreeClassifier(
    criterion="gini",      # or "entropy"
    max_depth=None,        # increase/decrease for tuning
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ------------------------------------------------------------
# STEP 5: Prediction
# ------------------------------------------------------------
y_pred = model.predict(X_test)

# ------------------------------------------------------------
# STEP 6: Accuracy & Evaluation Report
# ------------------------------------------------------------
print("\nACCURACY:", accuracy_score(y_test, y_pred))
print("\nCONFUSION MATRIX:\n", confusion_matrix(y_test, y_pred))
print("\nCLASSIFICATION REPORT:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# STEP 7 (Optional): Visualize Decision Tree
# ------------------------------------------------------------
plt.figure(figsize=(18, 10))
plot_tree(model, feature_names=X.columns, class_names=True, filled=True)
plt.title("Decision Tree Visualization")
plt.show()








import 'package:flutter/material.dart';

void main() {
  runApp(const CalculatorApp());
} 

class CalculatorApp extends StatefulWidget {
  const CalculatorApp({super.key});


  @override
  State<CalculatorApp> createState() => CalculatorAppState();
}

class CalculatorAppState extends State<CalculatorApp> {
  
   String display = "0";
   String expression = "";
   bool shouldReset = false;

  
 
  void onButtonPress(String value) {
    setState(() {
      if (value == "AC") {
         display = "0";
         expression = "";
      } else if (value == "CE") {
        display = "0"; 
      } else if (value == "<") {
        if (display.length > 1) {                            
          display = display.substring(0, display.length - 1);
        }                                                      
        else {
          display = "0";
        }
      } else if (value == "=") {
        expression += display;
        try {
          final result = _evaluate(expression);
          display = result.toString();  
        } catch (e) {
          display = "Error";
        }
        expression = "";
        shouldReset = true;
      } else if ("+-*/%".contains(value))  {
        expression += display + value;
        shouldReset = true;
      }
       else {
        if (display == "0" || shouldReset) {
          display = value;
          shouldReset = false;
        } else {
          display += value;
        }
      }
    });
  }

  double _evaluate(String exp) {
    exp = exp.replaceAll("×", "*").replaceAll("÷", "/");

    List<String> operators = ["+", "-", "*", "/", "%"];
                             
    double parseExpression(String expr) {
       
       for (String op in ["*", "/", "%"]) {
        int index = expr.lastIndexOf(op);
        if (index != -1) {
          double left = parseExpression(expr.substring(0, index));
          double right = double.parse(expr.substring(index + 1));  
                                                                     
          switch (op) {                        
            case "*":                   
              return left * right;     
            case "/":                
              return left / right;   
            case "%":               
              return left % right;
          }                                       
        }                                                          
       }

      for (String op in ["+", "-"]) {
        int index = expr.lastIndexOf(op);
        if (index != -1) {
          double left = parseExpression(expr.substring(0, index));
          double right = double.parse(expr.substring(index + 1));
        return op == "+"
              ? left + right
              : left - right;
        }
      }
      return double.parse(expr);
    }

    return parseExpression(exp);
  }
                    
Widget button(String text) {
    return Expanded(
        child: InkWell(                                         
        onTap: () => onButtonPress(text),
        child: Container(
          height: 70,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            border: Border.all(color: const Color.fromARGB(255, 113, 31, 31)),
            ),
          child: Text(                                         
            text,                                                       
            style :     const TextStyle(fontSize: 25),                      
          ),                                                        
        ),                                                     
      ),                                                        
    );                                                       
  }
  
@override
 Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: const Text("Calculator")
          ),
        body: Column(
          children: [
            Expanded( 
              child: Container(
                width: double.infinity,
                color: const Color.fromARGB(255, 235, 8, 8),
                alignment: Alignment.bottomRight,
                padding: const EdgeInsets.all(20),
                child: Text(display,
                    style: const TextStyle(fontSize: 45, color: Colors.white)),
                ),
            ),
            Row(children  : [button("AC"), button("CE"), button("<"), button("%")]),
            Row(children  : [button("7"), button("8"), button("9"), button("/")]),
            Row(children  : [button("4"), button("5"), button("6"), button("*")]),
            Row(children  : [button("1"), button("2"), button("3"), button("-")]),  
            Row(children  : [button("0"), button("."), button("="), button("+")]),
          ],
        ),
      ),
    );
  }
}


