import pickle

def load_and_print_artifacts_dict(path):
    artifacts_dict = pickle.load(open(path, "rb"))

    print("Target encoder mapping:")
    print([ac for ac in artifacts_dict["encoder"].mapping])

    print("Columns to train:")
    print([ac for ac in artifacts_dict["columns_to_score"]])

if __name__ == "__main__":
    load_and_print_artifacts_dict("./Artifacts/artifacts_dict_file.pkl")



