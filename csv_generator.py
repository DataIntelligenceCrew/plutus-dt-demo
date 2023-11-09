import random
import csv

def generate_random_csv_file(filename, n_rows, schema, weights):
  """Generates a random CSV file with the given schema and number of rows."""

  with open(filename, "w", encoding="UTF-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(schema.keys())

    for i in range(n_rows):
      row = []
      for column_name, column_type in schema.items():
        if column_name == "a":
          # Generate a random value between 0 and 1.
          row.append(random.choices((0,1), weights["a"], k=1)[0])
        elif column_name == "b":
          # Generate a random value between 0, 1, and 2.
          row.append(random.choices((0,1,2), weights["b"], k=1)[0])
        elif column_name == "c":
          # Generate a random value between 0, 1, 2, and 3.
          row.append(random.choices((0,1,2,3), weights["c"], k=1)[0])
        else:
          raise ValueError(f"Unknown column type: {column_type}")

      writer.writerow(row)

def generate_n_random_csv_files(n, schema, weights, output_dir="."):
  """Generates `n` random CSV files with the given schema, with values chosen from different distributions."""
  print(weights)

  for i in range(n):
    filename = f"{output_dir}/random_csv_{i}.csv"
    generate_random_csv_file(filename, 100000, schema, weights[i])

if __name__ == "__main__":
  schema = {
    "a": "int",
    "b": "int",
    "c": "int"
  }
  weights = [
    {
      "a": [1, 2],
      "b": [3, 1, 4],
      "c": [4, 6, 9, 0]
    },
    {
      "a": [1, 2],
      "b": [1, 5, 6],
      "c": [1, 15, 1, 2]
    },
    {
      "a": [1, 5],
      "b": [7, 8, 1],
      "c": [10, 7, 3, 3]
    },
    {
      "a": [2, 5],
      "b": [4, 1, 3],
      "c": [1, 1, 1, 0.2]
    },
    {
      "a": [3, 5],
      "b": [1, 6, 5],
      "c": [10, 5, 7, 3]
    },
  ]

  # Generate 10 random CSV files.
  generate_n_random_csv_files(5, schema, weights)
