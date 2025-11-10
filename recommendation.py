import os
import pandas as pd
from sklearn import model_selection, preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error


# ---------------------------
# Dataset Class
# ---------------------------
class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.movies)

    def __getitem__(self, idx):
        user = self.users[idx]
        movie = self.movies[idx]
        rating = self.ratings[idx]
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(movie, dtype=torch.long),
            torch.tensor(rating, dtype=torch.long),
        )


# ---------------------------
# Model Class
# ---------------------------
class RecSysModel(nn.Module):
    def __init__(self, n_users, n_movies, n_embeddings=32):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, n_embeddings)
        self.movie_embed = nn.Embedding(n_movies, n_embeddings)
        self.out = nn.Linear(n_embeddings * 2, 1)

    def forward(self, users, movies):
        user_embeds = self.user_embed(users)
        movie_embeds = self.movie_embed(movies)
        x = torch.cat([user_embeds, movie_embeds], dim=1)
        x = self.out(x)
        return x


# ---------------------------
# Main Function
# ---------------------------
def main():
    # Load dataset
    df = pd.read_csv("ratings.csv")
    print(f"Unique Users: {df.userId.nunique()}, Unique Movies: {df.movieId.nunique()}")

    # Encode IDs
    lbl_user = preprocessing.LabelEncoder()
    lbl_movie = preprocessing.LabelEncoder()
    df.userId = lbl_user.fit_transform(df.userId.values)
    df.movieId = lbl_movie.fit_transform(df.movieId.values)

    # Split dataset
    df_train, df_test = model_selection.train_test_split(
        df, test_size=0.2, random_state=42, stratify=df.rating.values
    )

    # Create datasets
    train_dataset = MovieDataset(
        users=df_train.userId.values,
        movies=df_train.movieId.values,
        ratings=df_train.rating.values,
    )
    test_dataset = MovieDataset(
        users=df_test.userId.values,
        movies=df_test.movieId.values,
        ratings=df_test.rating.values,
    )

    # Data loaders
    BATCH_SIZE = 4
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model, optimizer, and loss
    model = RecSysModel(
        n_users=len(lbl_user.classes_), n_movies=len(lbl_movie.classes_)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    MODEL_PATH = "recsys_model.pt"

# If model already exists, load it
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(" Loaded existing trained model.")
    else:
    # Training loop
        model.train()
        for epoch in range(1):
            for users, movies, ratings in train_loader:
                optimizer.zero_grad()
                y_pred = model(users, movies)
                y_true = ratings.unsqueeze(dim=1).float()
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
        print(f"Epoch [{epoch+1}] Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
        print(" Model saved successfully!")
    # Evaluation
    
if __name__ == "__main__":
    main()
