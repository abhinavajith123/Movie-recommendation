import torch
import pandas as pd
from sklearn import preprocessing
from recommendation import RecSysModel  # import your model class definition

# Load the same dataset to get label encoders
df = pd.read_csv("ratings.csv")

# Recreate the same label encoders as training
lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()
df.userId = lbl_user.fit_transform(df.userId.values)
df.movieId = lbl_movie.fit_transform(df.movieId.values)

# Load the trained model
model = RecSysModel(
    n_users=len(lbl_user.classes_),
    n_movies=len(lbl_movie.classes_)
)
model.load_state_dict(torch.load("recsys_model.pt",weights_only=True))
model.eval()

# Example: Predict for a specific user and movie
user_id = int(input("Enter user id:") )   # encoded user index
movie_id = int(input("Enter movie id:") )  # encoded movie index

with torch.no_grad():
    pred = model(torch.tensor([user_id]), torch.tensor([movie_id]))
    print(f"Predicted rating: {pred.item():.2f}")

# Optional: show actual rating if it exists
true_rows = df[(df.userId == user_id) & (df.movieId == movie_id)]
if not true_rows.empty:
    print(f"True rating: {true_rows.iloc[0].rating:.2f}")
else:
    print("No true rating found for this user–movie pair.")
