import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# --- STEP 1: CREATE THE DATA INSIDE THE CODE ---
csv_data = """Make,Volume,Doors,Style
Toyota,102,4,Sedan
Kia,121,5,SUV
Mazda,113,4,Sedan
Porshe,134,5,SUV
Chevrolet,134,5,SUV
Chevrolet,300,5,Van
Mercedes,114,5,SUV
Cadilac,167,5,SUV
Honda,120,4,Sedan
Ford,96,2,Sedan
Toyota,100,4,Sedan
Ford,130,4,Pickup
Jeep,110,4,Jeep
Honda,150,4,Van
Tesla,97,4,Sedan
Nissan,105,4,SUV
Chevrolet,135,4,Pickup
Toyota,108,4,SUV
Ford,115,4,SUV
Honda,95,4,Sedan
Lexus,107,4,SUV
Nissan,92,4,SUV
Nissan,105,4,SUV
Porsche ,89,4,SUV
Jeep,70,2,Jeep
Toyota,97,4,SUV
Toyota,88,4,Sedan
Ford,120,4,Pickup
Tesla,104,4,Sedan
Honda,97,4,Sedan
Toyota,100,4,Sedan
Land Rover,115,4,SUV
Jeep,95,4,Jeep
Nissan,105,4,SUV
Honda,97,4,Sedan
Ford,132,4,Pickup
Volkswagen,94,4,Sedan
Tesla,99,4,Sedan
Kia,110,4,SUV
Toyota,103,4,Sedan
Tesla,135,4,SUV
Jeep,72,4,Jeep
Toyota,101,4,Sedan
Porsche,145,4,SUV
Jeep,53,2,Jeep
Porsche,137,4,SUV
Audi,100,4,Sedan
BMW,131,4,SUV
Toyota,98,4,Pickup
Volkswagen,94,4,Sedan
Honda,192,5,Van
Honda,106,5,SUV
Toyota,105,4,Sedan
Kia,96,4,Sedan
Kia,168,5,SUV
Nissan,116,4,Sedan
BMW,112,4,Sedan
Nissan,137,5,SUV
Ram,132,4,Pickup
Hyundai,113,4,Sedan
Lexus,145,4,SUV
Ram,132,4,Pickup
Nissan,171,4,SUV
Toyota,99,4,SUV
Audi,96,4,SUV
Tesla,106,4,SUV
BMW,136,4,SUV
Mercedes,134,4,SUV
Honda,103,4,SUV
Chevrolet,168,4,SUV
Honda,112,4,Sedan
Nissan,110,4,Sedan
Chevrolet,167,5,SUV
Buick,144,5,SUV
Mazda,114,4,Sedan
Hyundai,166,5,SUV
Nissan,118,4,Sedan
Honda,110,2,Sedan
Toyota,100,4,Pickup
Infiniti,112,4,Sedan
Honda,122,4,Sedan
Ford,132,4,Pickup
Jeep,135,5,Jeep
Honda,196,5,Van
Tesla,112,4,Sedan
Nissan,136,5,SUV
Toyota,115,4,Sedan
Chevrolet,134,4,Pickup
Dodge,190,5,Van
Ford,152,5,SUV
BMW,128,4,SUV
Honda,106,4,SUV
Mercedes,111,4,SUV
Ford,106,4,SUV
Tesla,128,4,SUV
Audi,109,4,SUV
Jeep,104,4,Jeep
Mercedes,112,4,Sedan
Tesla,128,4,SUV
BMW,105,4,Sedan
Jeep,160,4,SUV
Volkswagen,110,4,Sedan
Chrysler,155,4,Sedan
Nissan,140,4,SUV
Ford,170,4,Pickup
Acura,118,4,Sedan
Mercedes,110,4,Sedan
BMW,88,4,Sedan
Audi,94,4,Sedan
Nissan,110,5,Sedan
Toyota,38,4,SUV
Volvo,12,4,Sedan
Mercedes-Bends,17,4,SUV
Jeep,36,4,SUV
Nissan,25,4,SUV
Volvo,11,2,Sedan
Nissan,14,4,Sedan
Volkswagen,14,4,Sedan
Toyota,13,4,Sedan
Ford,34,4,SUV
Jeep,144,5,SUV
Tesla,97,4,Sedan
Ford,82,2,Sedan
Nissan,136,5,SUV
Chrysler,165,5,Van
Toyota,100,4,Sedan
Ford,132,4,Pickup
Honda,160,5,Van
Jeep,104,3,Jeep
Chevrolet,135,4,Pickup
Chevrolet,94,4,Sedan
Mazda,94,4,SUV
Porsche,134,4,SUV
Mercedes,122,4,SUV
Fiat,85,2,Sedan
Toyota,115,4,Sedan
Toyota,102,4,Sedan
Lexus,101,4,Sedan
Kia,121,4,Sedan
Volvo,117,4,SUV
Chevrolet,122,5,SUV
Ford,172,5,SUV
Tesla,76,5,SUV
Jeep,105,5,SUV
Ford,122,4,Pickup
Nissan,106,5,SUV
Chevrolet,120,4,Pickup
Lexus,120,5,SUV
Toyota,122,4,Pickup
Ford,105,5,SUV
Infiniti Q50,100,4,Sedan
Toyota RAV4 XSE,136,4,SUV
Honda Accord V6,106,4,Sedan
Mercedes GLA 250,123,4,SUV
Chevrolet Equinox LT,144,4,SUV
Chevrolet Trailblazer,121,4,SUV
Audi A5,99,4,Sedan
Land Rover Discovery,170,4,SUV
Ford Focus SE,95,4,Sedan
Ford Bronco,136,4,Jeep"""

# Save this data to a real file so you can submit it
with open("MyCars.csv", "w") as f:
    f.write(csv_data)
print("SUCCESS: Generated 'MyCars.csv' from the data provided.")

# --- STEP 2: LOAD AND PROCESS ---
# Now we read the file we just created
df = pd.read_csv("MyCars.csv")

# Select features and target
X = df[['Volume', 'Doors']]
y = df['Style']

# Normalize (0-1 scale)
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=['Volume', 'Doors'])

# Recombine for splitting
processed_data = X_normalized.copy()
processed_data['Style'] = y

# --- STEP 3: SPLIT (80/20) ---
train_df, test_df = train_test_split(processed_data, test_size=0.2, random_state=42)

# Save split files
train_df.to_csv('Training.csv', index=False)
test_df.to_csv('Testing.csv', index=False)
print("SUCCESS: Created 'Training.csv' and 'Testing.csv'")

# --- STEP 4: TRAIN KNN AND FIND BEST K ---
X_train = train_df[['Volume', 'Doors']]
y_train = train_df['Style']
X_test = test_df[['Volume', 'Doors']]
y_test = test_df['Style']

accuracies = []
best_k = 1
best_accuracy = 0
best_model = None

for k in range(1, len(X_train) + 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    predictions = knn.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    accuracies.append({'K': k, 'Accuracy': acc})
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_k = k
        best_model = knn

pd.DataFrame(accuracies).to_csv('Accuracy.csv', index=False)
print(f"SUCCESS: Created 'Accuracy.csv'. Best K was: {best_k}")

# --- STEP 5: FINAL PREDICTION ---
final_predictions = best_model.predict(X_test)
confidence_scores = np.max(best_model.predict_proba(X_test), axis=1)

# Add results to dataframe
test_df_final = test_df.copy()
test_df_final['Prediction'] = final_predictions
test_df_final['Confidence'] = confidence_scores

# Update the testing file
test_df_final.to_csv('Testing.csv', index=False)
print("SUCCESS: Updated 'Testing.csv' with final predictions.")
