from preprocessing import *
from save_model import save_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
# I put this program in the same folder as MLGame/games/arkaonid/ml
# you can edit path to get log folder
if __name__ == "__main__":
    # preprocessing
    data_set = get_dataset()
    X, y = combine_multiple_data(data_set)

    # %% training
    model = KNeighborsClassifier(n_neighbors=3)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    print("model:", model)
    print(accuracy_score(y_predict, y_test))

    # %% save the model
    save_model(model, "model.pickle")