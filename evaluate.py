import os
import pandas as pd
import matplotlib.pyplot as plt

def extract_data_from_report(file_path):
    data = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                key, value = parts
                try:
                    if key == "Final Accuracy":
                        data[key] = float(value)
                    elif key in ["Epochs", "Batch Size", "LSTM Units", "Text Length used", "Number of Words used"]:
                        data[key] = int(value)
                    elif key == "Learning Rate":
                        data[key] = float(value)
                except ValueError:
                    print(f"Skipping malformed entry in file: {file_path}")
                    return None
    return data if len(data) == 7 else None

def main():
    input_folder = "results"
    output_csv = "training_results.csv"
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    headers = ["Epochs", "Batch Size", "LSTM Units", "Learning Rate", "Text Length used", "Number of Words used", "Final Accuracy"]
    results = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            extracted_data = extract_data_from_report(file_path)
            if extracted_data:
                results.append(extracted_data)

    if not results:
        print("No valid data found. Exiting.")
        return

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"CSV file '{output_csv}' created successfully with {len(df)} entries.")

    plt.figure()
    df.groupby("Learning Rate")["Final Accuracy"].mean().plot(kind='bar', rot=45)
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Accuracy")
    plt.title("Influence of Learning Rate on Accuracy")
    plt.savefig(os.path.join(plot_dir, "learning_rate_vs_accuracy.png"))
    plt.close()

    plt.figure()
    df.groupby("Epochs")["Final Accuracy"].mean().plot(kind='bar', rot=45)
    plt.xlabel("Epochs")
    plt.ylabel("Final Accuracy")
    plt.title("Influence of Epochs on Accuracy")
    plt.savefig(os.path.join(plot_dir, "epochs_vs_accuracy.png"))
    plt.close()

    plt.figure()
    df.groupby("Batch Size")["Final Accuracy"].mean().plot(kind='bar', rot=45)
    plt.xlabel("Batch Size")
    plt.ylabel("Final Accuracy")
    plt.title("Influence of Batch Size on Accuracy")
    plt.savefig(os.path.join(plot_dir, "batch_size_vs_accuracy.png"))
    plt.close()

    plt.figure()
    df.groupby("LSTM Units")["Final Accuracy"].mean().plot(kind='bar', rot=45)
    plt.xlabel("LSTM Units")
    plt.ylabel("Final Accuracy")
    plt.title("Influence of LSTM Units on Accuracy")
    plt.savefig(os.path.join(plot_dir, "lstm_units_vs_accuracy.png"))
    plt.close()

    plt.figure()
    df.groupby("Text Length used")["Final Accuracy"].mean().plot(kind='bar', rot=45)
    plt.xlabel("Text Length Used")
    plt.ylabel("Final Accuracy")
    plt.title("Influence of Text Length on Accuracy")
    plt.savefig(os.path.join(plot_dir, "text_length_vs_accuracy.png"))
    plt.close()

if __name__ == "__main__":
    main()
